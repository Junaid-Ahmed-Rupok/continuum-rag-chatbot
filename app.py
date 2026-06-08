"""
Continuum RAG Chatbot - Main Streamlit Application
A persistent memory chatbot with RAG, memory decay, and local LLM support.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

import streamlit as st
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)

from utils import (
    ContinuumConfig,
    ConversationBuffer,
    RAGMemory,
    extract_facts,
    get_config,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="Continuum AI - Persistent Memory Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Custom CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2.5rem; }
    .main-header p  { color: #e0e0e0; margin: 0.5rem 0 0 0; }
    .fact-item {
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 5px;
        margin: 5px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Runtime config (mutable, UI-adjustable — separate from frozen ContinuumConfig)
# ============================================================================

@dataclass
class RuntimeOverrides:
    """
    Mutable UI overrides that shadow specific ContinuumConfig fields.
    Avoids mutating the frozen ContinuumConfig dataclass at runtime.
    """
    top_k: int
    decay_lambda: float

    @classmethod
    def from_config(cls, config: ContinuumConfig) -> "RuntimeOverrides":
        return cls(top_k=config.top_k, decay_lambda=config.decay_lambda)

# ============================================================================
# Session State Initialization
# ============================================================================

def init_session_state() -> None:
    """Initialize all session state variables on first run."""
    defaults = {
        "memory_initialized": False,
        "config": get_config(),
        "memory": None,
        "conv_buffer": None,
        "llm": None,
        "tokenizer": None,
        "messages": [],
        "model_loaded": False,
        "device": None,
        "overrides": None,  # RuntimeOverrides, set after config is available
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Lazily initialize overrides once config exists
    if st.session_state.overrides is None and st.session_state.config is not None:
        st.session_state.overrides = RuntimeOverrides.from_config(st.session_state.config)

# ============================================================================
# LLM Loading
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_llm():
    """
    Load Phi-3-mini model; cached across reruns.

    NOTE: st.cache_resource will cache a (None, None, None) failure result,
    preventing automatic retries. If loading fails, the user must restart
    the Streamlit server to force a reload.
    """
    model_id = "microsoft/Phi-3-mini-4k-instruct"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        if torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
            device = "GPU"
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                device_map="cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            device = "CPU"

        model.eval()
        logger.info("Model loaded on %s", device)
        return model, tokenizer, device

    except Exception as e:
        logger.error("Failed to load model: %s", e)
        return None, None, None

# ============================================================================
# Response Generation  (pure generator — no side effects on memory/buffer)
# ============================================================================

def stream_response(
    message: str,
    memories_for_prompt: list,
    conv_buffer: ConversationBuffer,
    llm,
    tokenizer,
    config: ContinuumConfig,
    overrides: RuntimeOverrides,
):
    """
    Yield incremental response chunks from the LLM.

    Memory decay, retrieval, and reinforcement are intentionally kept OUT of
    this generator. They are performed by the caller (generate_reply) before
    streaming begins, so a mid-stream exception cannot leave memory in a
    partially mutated state while the buffer has no matching assistant entry.
    """
    system_prompt = (
        "You are Continuum, a helpful AI assistant with persistent memory. "
        "You remember facts about the user across conversations. "
        "Weave memories naturally into responses — never say 'according to my memory'. "
        "Be warm, concise, and helpful. Keep responses under 200 words unless asked."
    )
    if memories_for_prompt:
        facts = "\n".join(
            f"- {m.text} (relevance: {m.score:.2f})" for m in memories_for_prompt
        )
        system_prompt += f"\n\nWhat you remember about this user:\n{facts}"

    messages_for_llm = [
        {"role": "system", "content": system_prompt},
        *conv_buffer.format_for_llm(),
        {"role": "user", "content": message},
    ]

    prompt = tokenizer.apply_chat_template(
        messages_for_llm, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=config.max_tokens,
        temperature=config.temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    thread = Thread(target=llm.generate, kwargs=gen_kwargs)
    thread.start()

    response = ""
    for chunk in streamer:
        if chunk:
            response += chunk
            yield response

    thread.join()


def generate_reply(
    message: str,
    memory: RAGMemory,
    conv_buffer: ConversationBuffer,
    llm,
    tokenizer,
    config: ContinuumConfig,
    overrides: RuntimeOverrides,
):
    """
    Orchestrate memory operations and LLM streaming for a single turn.

    Memory side effects (decay, retrieval, reinforcement) happen here —
    before streaming — so that stream_response remains a pure generator.
    Buffer writes happen only in persist_turn, after streaming succeeds,
    preventing orphaned user messages on mid-stream failures.
    """
    # --- Memory side effects BEFORE streaming ---
    memory.decay_all()
    memories = memory.retrieve(message, top_k=overrides.top_k)
    for mem in memories:
        memory.reinforce(mem.id)

    # Delegate pure token streaming to the generator
    yield from stream_response(
        message, memories, conv_buffer, llm, tokenizer, config, overrides
    )


def persist_turn(
    user_msg: str,
    assistant_msg: str,
    memory: RAGMemory,
    conv_buffer: ConversationBuffer,
) -> None:
    """
    Persist a completed conversation turn: save both messages to the buffer,
    then extract and store facts from the exchange.

    Both buffer writes happen together here (not split around the stream) so
    the buffer always contains matched user/assistant pairs.

    NOTE: extract_facts may be slow if it calls an external model or heavy
    NLP pipeline. Consider running it in a background thread if it causes
    noticeable UI lag.
    """
    conv_buffer.add("user", user_msg)
    conv_buffer.add("assistant", assistant_msg)
    conv_buffer.save()

    for fact in extract_facts(user_msg, assistant_msg):
        memory.add_memory(fact, {"source": "conversation"})

# ============================================================================
# Sidebar helpers
# ============================================================================

def render_stats(memory: RAGMemory) -> None:
    stats = memory.get_stats()
    strength_pct = int(stats["avg_strength"] * 100)
    st.markdown(f"""
### 📊 Memory Statistics
| Metric | Value |
|--------|-------|
| Total Memories | **{stats['total']}** |
| Avg Strength | **{stats['avg_strength']:.2f}** ({strength_pct}%) |
| Session Added | +{stats['session_added']} |
| Session Pruned | -{stats['session_pruned']} |
| Storage Size | {stats['size_kb']:.1f} KB |
""")


def render_top_facts(memory: RAGMemory) -> None:
    facts = memory.get_top_facts(5)
    if not facts:
        st.markdown("✨ No memories yet. Start chatting!")
        return

    st.markdown("### 🧠 Strongest Memories")
    for i, fact in enumerate(facts, 1):
        filled = int(fact["strength"] * 10)
        bar = "█" * filled + "░" * (10 - filled)
        st.markdown(
            f"**{i}.** {fact['text'][:80]}\n\n"
            f"   `{bar}` {fact['strength']:.2f}"
        )


def render_status(model_loaded: bool, memory_initialized: bool, device: str | None) -> None:
    st.markdown("### 📡 Status")
    col1, col2, col3 = st.columns(3)

    model_icon  = "🟢" if model_loaded else "🔴"
    memory_icon = "🟢" if memory_initialized else "🔴"
    device_label = device or "—"

    with col1:
        st.markdown(f"{model_icon} **Model**\n\n{device_label}")
    with col2:
        memory_status = "Active" if memory_initialized else "Off"
        st.markdown(f"{memory_icon} **Memory**\n\n{memory_status}")
    with col3:
        st.markdown("🟢 **Stream**\n\nOn" if model_loaded else "⚫ **Stream**\n\nOff")

# ============================================================================
# Main
# ============================================================================

def main() -> None:
    init_session_state()

    st.markdown("""
    <div class="main-header">
        <h1>🧠 Continuum AI</h1>
        <p>Persistent Memory · RAG · Local LLM</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Initialize memory system once ---
    if not st.session_state.memory_initialized:
        with st.spinner("🔧 Initializing memory system..."):
            config = st.session_state.config
            st.session_state.memory = RAGMemory(config)
            st.session_state.conv_buffer = ConversationBuffer(
                Path(config.data_dir),
                max_turns=config.ctx_window_turns,
            )
            st.session_state.memory_initialized = True

    # --- Load LLM once ---
    # FIX: Read device from session state rather than a local variable that
    # stays None on reruns where the model is already loaded.
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading Phi-3-mini... (3–5 min on first run)"):
            llm, tokenizer, device = load_llm()
        if llm is None:
            st.error("❌ Model failed to load. Check logs and restart the server.")
            st.info(
                "💡 Ensure you have an internet connection for first-time download. "
                "If the problem persists, restart Streamlit to clear the model cache."
            )
            st.stop()
        st.session_state.llm = llm
        st.session_state.tokenizer = tokenizer
        st.session_state.model_loaded = True
        st.session_state.device = device
        # FIX: Removed unnecessary st.rerun() here. The rest of main() renders
        # correctly in the same run; rerunning just added a redundant cycle.

    col_chat, col_sidebar = st.columns([2, 1])

    # ── Chat column ──────────────────────────────────────────────────────────
    with col_chat:
        st.markdown("### 💬 Conversation")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Type your message here..."):
            # Show user message immediately
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # FIX: Do NOT write user message to conv_buffer here. Both the user
            # and assistant messages are written together in persist_turn after
            # streaming succeeds. This prevents an orphaned user entry in the
            # buffer if streaming fails mid-way.

            # Stream assistant response
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""

                try:
                    for chunk in generate_reply(
                        prompt,
                        st.session_state.memory,
                        st.session_state.conv_buffer,
                        st.session_state.llm,
                        st.session_state.tokenizer,
                        st.session_state.config,
                        st.session_state.overrides,
                    ):
                        full_response = chunk
                        placeholder.markdown(full_response + "▌")

                    placeholder.markdown(full_response)

                except Exception as e:
                    full_response = f"❌ Generation error: {e}"
                    placeholder.markdown(full_response)
                    logger.error("Generation failed: %s", e)

            # Persist the completed turn only on success (side effects here,
            # not in the generator, and not split around the stream).
            if full_response and not full_response.startswith("❌"):
                persist_turn(
                    prompt,
                    full_response,
                    st.session_state.memory,
                    st.session_state.conv_buffer,
                )

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.rerun()

    # ── Sidebar column ───────────────────────────────────────────────────────
    with col_sidebar:
        st.markdown("### 🧠 Memory Status")
        render_stats(st.session_state.memory)

        st.divider()
        render_top_facts(st.session_state.memory)

        st.divider()

        with st.expander("⚙️ Settings"):
            st.markdown("**Retrieval Settings**")
            overrides = st.session_state.overrides

            overrides.top_k = st.slider(
                "Top-K Retrieval", 1, 10, overrides.top_k,
                help="Number of memories to retrieve per query",
            )
            overrides.decay_lambda = st.slider(
                "Decay Rate (λ)", 0.01, 0.5, overrides.decay_lambda, step=0.01,
                help="Higher = faster forgetting",
            )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🗑️ Reset Memory", use_container_width=True):
                st.session_state.memory.reset()
                st.session_state.conv_buffer.clear()
                st.session_state.messages = []
                st.success("✅ Memory and chat reset!")
                st.rerun()

        with col_b:
            if st.button("💾 Export Memory", use_container_width=True):
                try:
                    path = st.session_state.memory.export_json()
                    st.success(f"✅ Exported to: {path}")
                except Exception as e:
                    st.error(f"❌ Export failed: {e}")

        st.divider()
        # FIX: Read device from session state so it's correct on all reruns,
        # not just the one where the model first loaded.
        render_status(
            st.session_state.model_loaded,
            st.session_state.memory_initialized,
            st.session_state.get("device"),
        )


if __name__ == "__main__":
    main()
