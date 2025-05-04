# FAQ Chatbot Using RAG for Customer Support

A Retrieval-Augmented Generation (RAG) chatbot for e-commerce customer support, leveraging open-source LLMs and semantic search for accurate, context-aware answers.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [System Architecture](#system-architecture)
4. [Project Structure](#project-structure)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Evaluation & Results](#evaluation--results)
8. [Troubleshooting](#troubleshooting)
9. [Future Work](#future-work)
10. [Credits & References](#credits--references)

---

## Project Overview

**Domain:** Customer Support Automation  
**Corpus:** E-commerce FAQs (Hugging Face: NebulaByte/E-Commerce_FAQs or local CSV)  
**Goal:** Retrieve relevant responses from a knowledge base and generate user-friendly, context-aware answers to customer queries.

**Key Objectives:**
- Automate responses to common customer support questions
- Use RAG to combine semantic retrieval with LLM-based generation
- Support multiple languages and model sizes for broad accessibility

**Value Proposition:**
- Reduces manual workload for support agents
- Delivers faster, more accurate answers to customers
- Minimizes hallucination by grounding responses in real company FAQs

---

## Features
- **Retrieval-Augmented Generation:** Combines semantic search (Sentence-BERT + FAISS) with LLMs for grounded answers
- **Multiple LLMs Supported:** Phi-2, TinyLlama-1.1B, Mistral-7B (selectable based on hardware)
- **Multilingual:** English, Spanish, French (automatic translation of queries and responses)
- **Web Interface:** Streamlit-based, with chat history, configuration sidebar, and feedback system
- **Performance Metrics:** Real-time display of retrieval/generation times and memory usage
- **User Feedback:** 1-5 rating and comments for continuous improvement
- **Sample Questions:** Quick testing with common e-commerce queries
- **Resource Adaptation:** Runs on CPU or GPU, with memory and speed optimizations

---

## System Architecture

1. **User Query:** Entered via web interface
2. **Preprocessing:** Clean, translate, and augment query if needed
3. **Embedding & Retrieval:** Query embedded (Sentence-BERT), similar FAQs retrieved from FAISS index
4. **Context Construction:** Top FAQs formatted as context for the LLM
5. **Response Generation:** LLM generates answer using context and query
6. **Display & Feedback:** Response, relevant FAQs, and metrics shown; user can rate and comment

**Main Components:**
- Data Processing (`src/data_processing.py`)
- Embedding & Retrieval (`src/embedding.py`)
- LLM Response Generation (`src/llm_response.py`)
- Utilities & Evaluation (`src/utils.py`)
- Web Interface (`app.py`)

---

## Project Structure

```
faq-rag-chatbot/
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
├── README.md              # User documentation manual
├── data/                  # FAQ datasets (CSV, JSON)
├── src/                   # Source code modules
│   ├── data_processing.py # Data loading, cleaning, augmentation
│   ├── embedding.py       # Embedding generation, FAISS retrieval
│   ├── llm_response.py    # LLM loading, prompt, response
│   ├── utils.py           # Evaluation, metrics, memory utils
│   └── __init__.py
├── embeddings/            # Persisted embedding model/index
├── docs/                  # Project documentation (reports, guides)
├── venv/                  # Python virtual environment
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+
- 16GB+ RAM (32GB+ recommended for larger models)
- Optional: CUDA-compatible GPU with 8GB+ VRAM (15GB+ for Mistral-7B)
- Internet connection (for downloading models/datasets)
- Swap space (4GB) recommended for <32GB RAM

### Installation Steps

1. **Clone the Repository**
   ```bash
    git clone https://github.com/jitendra147/faq-rag-chatbot.git
    cd faq-rag-chatbot
    ```
2. **Create and Activate a Virtual Environment**
   - Linux/macOS:
     ```bash
     python -m venv venv
     source venv/bin/activate
     ```
   - Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **(Optional) Download NLTK Resources**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
   ```
5. **(Optional) Add Swap Space (Linux, for low memory systems)**
   ```bash
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

### Running the Application

```bash
streamlit run app.py
```

Access the web interface at [http://localhost:8501](http://localhost:8501).

---

## Usage Guide

### Interface Overview
- **Main Chat Area:** Interact with the chatbot
- **Retrieved Information:** See FAQs used for the answer
- **Configuration Sidebar:** Choose model, language, data source, and more
- **Sample Questions:** Click to test common queries
- **Performance Metrics:** View retrieval/generation times and memory usage
- **Feedback:** Rate responses and leave comments

### Configuration Options
- **Dataset Source:** Hugging Face (NebulaByte/E-Commerce_FAQs) or local CSV
- **FAQ Augmentation:** Enable for paraphrased questions (slower startup, better coverage)
- **Language:** English, Spanish, French
- **LLM Model:** Phi-2 (balanced), TinyLlama-1.1B (fastest), Mistral-7B (highest quality, GPU only)
- **Memory Usage Display:** See current RAM/VRAM utilization

### Interacting with the Chatbot
1. Type your question and click "Ask"
2. View the chatbot's response and the retrieved FAQs
3. Rate the answer (1-5) and leave feedback if desired
4. Try sample questions for quick demos

### Performance Tips
- Use TinyLlama for low-memory systems
- Enable FAQ augmentation for better retrieval (if you have enough RAM)
- Preload embeddings for faster responses
- Use GPU for best performance with larger models

---

## Evaluation & Results

### Retrieval Performance
- **RAG (Sentence-BERT) outperforms keyword-based search**
  - Precision@1: 0.82 (RAG) vs. 0.64 (TF-IDF)
  - Recall@3: 0.78 (RAG) vs. 0.51 (TF-IDF)
- **Dense embeddings** handle paraphrased and semantically similar queries much better than keyword search

### Response Quality
- **BLEU/ROUGE-L/Word Overlap**: RAG+LLM consistently outperforms baselines
- **Human Ratings**: 4.2/5 (Phi-2+RAG), 3.8/5 (TinyLlama+RAG), 4.4/5 (Mistral-7B+RAG)
- **Multilingual**: Maintains ~90% of English performance in Spanish/French

### System Performance
- **Retrieval time:** ~0.03s (FAISS)
- **Generation time:** 1-4s (GPU), 20-100s (CPU, large models)
- **Memory usage:** 3-32GB RAM depending on model

---

## Troubleshooting

**Out of Memory Errors:**
- Switch to TinyLlama
- Disable FAQ augmentation
- Add swap space
- Reduce embedding batch size

**Slow Response Time:**
- Use smaller model
- Preload embeddings
- Use GPU if available

**Model Loading Failures:**
- Check internet connection
- Ensure sufficient disk space
- Update `transformers` library

**CUDA Errors:**
- Update GPU drivers
- Check CUDA compatibility
- Set `CUDA_VISIBLE_DEVICES=0` if needed

For more help, see the [GitHub issues](https://github.com/your-repo/issues) or consult library docs.

---

## Future Work
- **Hybrid Retrieval:** Combine dense and sparse (BM25) search
- **Advanced Embeddings:** Try E5, BGE, or domain-tuned models
- **Chunking Strategies:** Test sentence/paragraph/semantic chunking
- **Model Fine-tuning:** PEFT/LoRA, instruction tuning for support data
- **Conversation Management:** Multi-turn context, proactive clarifications
- **UI/UX Enhancements:** Mobile support, voice input/output, accessibility
- **Business Integration:** Ticketing system integration, agent handoff, analytics
- **Scaling:** Automated knowledge base updates, microservices, monitoring

---

## Credits & References

**Team:**
- Deshik Sastry Yarlagadda
- Sai Jitendra Chowdary Katragadda

**Key References:**
- Lewis et al. (2020) "Retrieval-augmented generation for knowledge-intensive NLP tasks"
- Reimers & Gurevych (2019) "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- Hugging Face Datasets: NebulaByte/E-Commerce_FAQs
- See `comprehensive-project-report.md` for full citations



