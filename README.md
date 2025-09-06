---

# ⚡ LightningServe: Production-Ready Vision-Language RAG Inference Stack

🔗 **GitHub Repo:** [LightningServe](https://github.com/hasti1126/lightningserve)
🌐 **Live Demo:** [LightningServe on Render](https://lightning-serve-icnn7mvuxzzmwyllbuubvn.streamlit.app/)

---

## 💡 Motivation

LightningServe was built to demonstrate **low-latency, scalable, and reliable inference infrastructure** for multimodal foundation models.
It highlights **systems engineering + AI integration skills** — combining **state-of-the-art embeddings (Cohere Embed v4)** and **Google Gemini 2.5 Flash** for real-time retrieval and reasoning, wrapped in a serving stack with **observability, caching, and benchmarking**.

This project aligns with real-world needs for building **production-ready inference pipelines** in AI-first companies.

---

## ✨ Features

* **FastAPI Inference API** – Low-latency endpoints for document upload, query, and streaming retrieval
* **Cohere Embed v4** – High-performance embeddings optimized for retrieval ([details](https://cohere.com/blog/embed-4))
* **Google Gemini 2.5 Flash** – LLM for reasoning and response generation via Google AI Studio
* **Redis Caching** – Speeds up repeated queries with in-memory caching
* **Prometheus Observability** – Metrics on queries, latency, errors, cache hits, and system usage
* **Benchmarking Utilities** – Measure throughput, latency, and reliability under real-world workloads
* **Streamlit Dashboard** – Interactive UI for upload, querying, analytics, and performance monitoring

---

## 🏗️ System Architecture

```mermaid
graph TD
    A[User / Client] -->|Query| B[FastAPI Backend]
    B --> C[Cohere Embed v4 for Indexing & Retrieval]
    B --> D[Redis Cache]
    B --> E[Google Gemini 2.5 Flash LLM]
    B --> F[Prometheus Metrics]
    B --> G[System Monitor (psutil)]
    B --> H[Streamlit Frontend]
```

---

## 🚀 Quick Start

### 1️⃣ Clone the repo

```bash
git clone https://github.com/your-username/lightningserve.git
cd lightningserve
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Set API Keys

```bash
export COHERE_API_KEY=your_cohere_key
export GOOGLE_API_KEY=your_google_key
```

### 4️⃣ Run the backend (FastAPI)

```bash
uvicorn app:app --reload
```

API available at: [https://lightning-serve.onrender.com/docs](https://lightning-serve.onrender.com/docs)

### 5️⃣ Run the frontend (Streamlit Dashboard)

```bash
streamlit run streamlit_app.py
```

Or simply try the **live hosted version** 👉 [LightningServe on Render](https://lightning-serve-icnn7mvuxzzmwyllbuubvn.streamlit.app/)

---

## 📡 API Endpoints

| Method | Endpoint                | Description                                    |
| ------ | ----------------------- | ---------------------------------------------- |
| POST   | `/rag/upload-documents` | Upload and index documents (Cohere Embed v4)   |
| POST   | `/rag/query`            | Query documents (retrieval + Gemini 2.5 Flash) |
| GET    | `/metrics`              | Prometheus metrics for observability           |
| GET    | `/stats`                | System stats (CPU, memory, disk)               |
| POST   | `/rag/benchmark`        | Run benchmark tests on queries                 |

---

## 📊 Monitoring & Benchmarking

* **Prometheus Metrics** → Track cache hits, latency distribution, query counts
* **Benchmarking Pipeline** → Evaluate throughput and reliability with fixed queries
* **Streamlit Analytics** → Visualize usage, response times, and system health

---

## 🛠️ Tech Stack

* **Backend:** FastAPI, Redis, Uvicorn
* **LLM & Embeddings:** Cohere Embed v4, Google Gemini 2.5 Flash (via Google AI Studio)
* **Observability:** Prometheus, psutil
* **Frontend:** Streamlit
* **Deployment:** Render (PaaS)

---

## 🔮 Future Improvements

* Distributed deployment with Kubernetes
* Autoscaling for high-concurrency inference
* GPU acceleration for embeddings + retrieval



⚡ **LightningServe = Low-latency serving + observability + reliability for multimodal AI.**

---
