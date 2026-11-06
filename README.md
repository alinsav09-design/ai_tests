# 🔒 Private Document Intelligence Platform (Local RAG Architecture)

A production-grade, offline-first Retrieval-Augmented Generation (RAG) platform that allows users to upload private data assets (PDFs, text data, or code matrices) and securely query them using local hardware acceleration. 

This project demonstrates a complete local AI Engineering ecosystem, running entirely within an accelerated infrastructure layer without sending telemetry data to external third-party cloud APIs.

## 🚀 Key Architectural Capabilities
* **Local Hardware Acceleration:** Uses PyTorch CUDA bindings to offload re-ranking and generative tasks to an on-premise **NVIDIA Quadro T2000 GPU**.
* **Isolated Vector Ingestion Layer:** Slices document segments into recursive token blocks and indexes them using a local `ChromaDB` vector storage layer.
* **Deterministic Tool Routing:** Intercepts real-time queries (such as system time or external weather API calls) using an intent routing agent layer to eliminate LLM hallucinations.
* **Real-time Interface Streaming:** Built with `Streamlit` to deliver sub-second text token streaming directly into a clean web layout interface.

## 🛠️ System Technology Stack
* **UI Frontend & State Management:** Python `Streamlit`
* **Local LLM Engine Provider:** `Ollama` framework orchestration
* **Linguistic Generative Processing Core:** `Llama 3.2 (1B)` (Instruct Quantized Matrix)
* **Mathematical Vector Embeddings Layer:** `nomic-embed-text`
* **Orchestration Framework:** `LangChain Ecosystem` / `Transformers`

## ⚙️ How It Works (Pipeline Topology)
1. **Document Ingestion:** A `PyPDFLoader` engine reads uploaded file payloads.
2. **Text Fragmentation:** Documents are broken down into token sequences using `RecursiveCharacterTextSplitter` (chunk sizes: 600, overlap: 100).
3. **Matrix Generation:** Text arrays are converted into dense vector layers via the local embedding engine.
4. **Context Synthesis:** ChromaDB queries semantic vector indexes using cosine similarity retrieval models (`k=3`), passes data variables into a strict markdown boundary prompt block (`### GROUND TRUTH DATA`), and lets the model stream the final validation output block.
