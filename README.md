markdown
<div align="center">

# 🧠 CONTINUUM AI
### *Persistent Memory RAG Chatbot That Never Forgets*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://continuum-rag-chatbot.streamlit.app)
[![GitHub stars](https://img.shields.io/github/stars/Junaid-Ahmed-Rupok/continuum-rag-chatbot?style=social)](https://github.com/Junaid-Ahmed-Rupok/continuum-rag-chatbot)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

## ✨ Overview

**Continuum AI** is a conversational AI assistant with **persistent memory** that never forgets. Unlike traditional chatbots that lose context after each session, Continuum remembers facts about you across conversations, automatically decays old memories (just like humans!), and retrieves relevant information using RAG (Retrieval-Augmented Generation).

### 🎯 Why Continuum AI?

| Feature | Traditional Chatbots | Continuum AI |
|---------|---------------------|--------------|
| Memory Across Sessions | ❌ No | ✅ Yes |
| Automatic Fact Extraction | ❌ No | ✅ Yes |
| Memory Decay (Forgetting) | ❌ No | ✅ Yes |
| Reinforcement Learning | ❌ No | ✅ Yes |
| Local LLM (No API Keys) | ❌ No | ✅ Yes |
| Persistent Storage | ❌ No | ✅ Yes |

---

## 🚀 Live Demo

🔗 **Try it now:** [continuum-rag-chatbot.streamlit.app](https://continuum-rag-chatbot.streamlit.app)

> **Note:** First load may take 3–5 minutes as the model downloads (~2.3GB). Subsequent loads are instant!

---

## 🌟 Key Features

### 🧠 Persistent Memory
- Stores memories in ChromaDB vector database
- Remembers facts across multiple conversations
- Never forgets what you tell it

### 📉 Ebbinghaus Memory Decay
- Memories naturally fade over time (just like humans!)
- Configurable decay rate (λ = 0.01 to 0.5)
- Frequently accessed memories get reinforced

### 🔍 RAG (Retrieval-Augmented Generation)
- Retrieves relevant memories before responding
- Uses cosine similarity for semantic search
- Top-K configurable (1–10 memories)

### 💪 Reinforcement Learning
- Strengthens memories when accessed
- `strength = min(1.0, old_strength + 0.3)`
- Frequently discussed topics stay strong

### 🔧 100% Free & Local
- No API keys required
- Runs Microsoft Phi-3-mini locally
- All processing on your machine

### 🎨 Beautiful UI
- Dark theme with neon accents
- Real-time memory statistics
- Streaming responses
- Mobile responsive

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Web UI framework |
| **LLM** | Microsoft Phi-3-mini (3.8B params) | Text generation |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Semantic search |
| **Vector DB** | ChromaDB | Persistent memory storage |
| **Framework** | PyTorch / Transformers | Model inference |
| **Language** | Python 3.10+ | Core logic |

---

## 📦 Installation

### Prerequisites
- Python 3.10 or higher
- 8GB+ RAM (16GB recommended)
- 5GB free disk space

### Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/Junaid-Ahmed-Rupok/continuum-rag-chatbot.git
cd continuum-rag-chatbot

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

---

## 🎮 Usage Guide

### Basic Commands

| Action | How To |
|--------|--------|
| Send message | Type in chat box → Press Enter |
| Reset all memory | Click "🗑️ Reset Memory" |
| Export memories | Click "💾 Export Memory" |
| Adjust Top-K | Settings panel → Slider |
| Change decay rate | Settings panel → Slider |

### Example Conversations

```
User: "Hi, my name is Alex"
Continuum: "Nice to meet you, Alex! How can I help you today?"
[Memory saved: "User's name is Alex"]

--- Next Session ---

User: "What's my name?"
Continuum: "Your name is Alex! You told me that in our previous conversation."
[Memory retrieved and reinforced]
```

### Memory Management Tips

1. **Reinforce important memories** by discussing them frequently
2. **Use the export feature** to back up your memories
3. **Adjust decay rate**:
   - Low (0.05): Very slow forgetting
   - Medium (0.1): Default, natural forgetting
   - High (0.5): Fast forgetting, short-term focus

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       USER INTERFACE                        │
│                      (Streamlit UI)                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      QUERY PROCESSING                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Decay All  │→ │  Retrieve   │→ │    Reinforce        │ │
│  │  Memories   │  │  Memories   │  │  Retrieved Ones     │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY SYSTEM                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  ChromaDB   │←→│  Embeddings │←→│  Sentence           │ │
│  │  Vector DB  │  │  (384-dim)  │  │  Transformer        │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                         LLM LAYER                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           Phi-3-mini (3.8B parameters)              │   │
│  │         Context: System + Memories + History        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      POST PROCESSING                        │
│  ┌─────────────┐  ┌─────────────────────────────────────┐  │
│  │  Save to    │  │  Extract Facts & Store to Memory    │  │
│  │  Buffer     │  │                                     │  │
│  └─────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Memory Decay Formula

Continuum implements the **Ebbinghaus Forgetting Curve**:

```
new_strength = old_strength × e^(-λ × days_since_reinforcement)

Where:
- λ = decay rate (default: 0.1)
- days_since = time since last access
- min_strength = pruning threshold (default: 0.05)
```

### Decay Visualization

| Days | λ=0.05 | λ=0.1 | λ=0.2 | λ=0.5 |
|------|--------|-------|-------|-------|
| 0    | 1.00   | 1.00  | 1.00  | 1.00  |
| 7    | 0.70   | 0.50  | 0.25  | 0.03  |
| 14   | 0.50   | 0.25  | 0.06  | 0.001 |
| 30   | 0.22   | 0.05  | 0.002 | 0.000 |

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TOP_K` | Number of memories to retrieve | 5 |
| `TEMPERATURE` | LLM creativity (0.0–1.0) | 0.7 |
| `DECAY_LAMBDA` | Memory decay rate | 0.1 |
| `CONTINUUM_DATA_DIR` | Data storage location | `./continuum_data` |

### Streamlit Configuration (`.streamlit/config.toml`)

```toml
[theme]
base = "dark"
primaryColor = "#00e5ff"
backgroundColor = "#0a0e1a"
secondaryBackgroundColor = "#151c2c"
textColor = "#f0f3ff"
```

---

## 🧪 Testing

```bash
python -m pytest tests/
```

Test coverage includes:
- ✅ Directory structure validation
- ✅ ChromaDB read/write operations
- ✅ Embedding dimension verification
- ✅ Decay formula accuracy
- ✅ Top-K retrieval respect
- ✅ Reinforcement functionality
- ✅ Fact extraction accuracy
- ✅ JSON export validity

---

## 📁 Project Structure

```
continuum-rag-chatbot/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
│
├── .streamlit/
│   └── config.toml             # Streamlit configuration
│
├── utils/
│   ├── __init__.py             # Package exports
│   ├── config.py               # Configuration management
│   ├── memory.py               # RAGMemory core class
│   ├── conversation.py         # Conversation buffer
│   └── helpers.py              # Fact extraction utilities
│
└── continuum_data/             # Auto-created on first run
    ├── db/                     # ChromaDB storage
    ├── exports/                # JSON exports
    └── config/                 # Session statistics
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model won't load | Check internet connection, restart Streamlit |
| Slow responses | Streamlit Cloud has limited CPU — run locally for better speed |
| Memory not persisting | Check `continuum_data/` folder permissions |
| Import errors | Run `pip install -r requirements.txt` |
| Port already in use | `streamlit run app.py --server.port 8502` |

### Getting Help

- 📧 Open an [Issue](https://github.com/Junaid-Ahmed-Rupok/continuum-rag-chatbot/issues)
- ⭐ Star the repo for updates
- 🔔 Watch for new releases

---

## 🗺️ Roadmap

### Version 1.0 ✅ (Current)
- [x] Persistent memory with ChromaDB
- [x] Ebbinghaus decay curves
- [x] Fact extraction
- [x] Streamlit UI
- [x] Local Phi-3-mini integration

### Version 1.1 (Planned)
- [ ] Multi-user support
- [ ] Memory categories/tags
- [ ] Export to CSV/PDF
- [ ] Voice input support

### Version 2.0 (Exploring)
- [ ] Fine-tuning on user conversations
- [ ] Multi-modal support (images)
- [ ] Web search integration
- [ ] Mobile app

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** changes: `git commit -m 'Add amazing feature'`
4. **Push** to branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

### Development Setup

```bash
git clone https://github.com/YOUR_USERNAME/continuum-rag-chatbot.git
cd continuum-rag-chatbot
pip install -r requirements.txt
pip install black pytest
black .
pytest
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for details.

---

## 🙏 Acknowledgments

- **Microsoft** for [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct)
- **Hugging Face** for the Transformers library
- **ChromaDB** for vector database
- **Streamlit** for the UI framework
- **Sentence Transformers** for embeddings

---

## 📞 Contact

**Junaid Ahmed Rupok**

- GitHub: [@Junaid-Ahmed-Rupok](https://github.com/Junaid-Ahmed-Rupok)
- Project: [continuum-rag-chatbot](https://github.com/Junaid-Ahmed-Rupok/continuum-rag-chatbot)
- Live Demo: [continuum-rag-chatbot.streamlit.app](https://continuum-rag-chatbot.streamlit.app)

---

<div align="center">

**Built with 🧠 by Junaid Ahmed Rupok**

*Never forget. Always remember.*

⭐ If you found this helpful, give it a star on GitHub!

</div>
```

