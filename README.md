# Smart Logistics RAG Search Engine 🔍

A Retrieval-Augmented Generation (RAG) system designed for logistics teams. It allows users to upload Standard Operating Procedures (SOPs) or configuration guides and ask plain-English questions to get accurate, grounded answers.

This project demonstrates how to connect a Large Language Model (LLM) to internal, private data without fine-tuning, preventing hallucinations.

## 🚀 Live Demo

### [Open the Logistics SOP Search Demo →](https://logistics-sop-search.onrender.com/)

### 12-second walkthrough

![YZA Logistics SOP Search demo showing a sample query, grounded evidence, and custom SOP upload](assets/yza-logistics-search-demo.gif)

Try the bundled **YZA Logistics Operations SOP Demo** with ready-made questions, or upload/paste your own `.txt` or `.md` SOP and search it directly in the browser. Custom documents stay inside the browser tab and are not uploaded or permanently stored.

> **Portfolio disclaimer:** The bundled YZA SOP is fictional sample data created for this demonstration. It is not an official YZA document and contains no proprietary company information.

> The free Render instance can take a short time to open after inactivity. Once the page loads, searching runs entirely in the browser with no backend wake-up required.

> **Part of the Logistics AI Suite**
> This project pairs with the [Logistics RCA Agent](../logistic-support). Together, they demonstrate both structured data extraction (RCA Agent) and semantic search + RAG (this project).

## 🌟 Features

- **100% Local Embeddings**: Uses HuggingFace's `all-MiniLM-L6-v2` to convert text into vectors locally (no API key required, keeps data private).
- **Local Vector Database**: Uses ChromaDB to store and retrieve document chunks on your laptop.
- **Dual LLM Support**: 
  - **Cloud**: Google Gemini 2.0 Flash (fast, smart)
  - **Local**: Ollama (100% offline, privacy-first)
- **FastAPI Backend**: Clean, typed REST API using Pydantic schemas.

## 🏗️ How It Works (The RAG Pipeline)

1. **Ingestion**: A user uploads an SOP (PDF or TXT). LangChain splits it into ~500-character chunks.
2. **Embedding**: Each chunk is converted into a vector and saved to ChromaDB.
3. **Retrieval**: The user asks a question. The question is embedded, and ChromaDB finds the top 3 most similar chunks.
4. **Generation**: The retrieved chunks and the question are sent to the LLM with a strict prompt: *"Answer using ONLY the provided context."*

## 🚀 Quick Start

### 1. Install Dependencies

It's highly recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Note: The first time you run this, `sentence-transformers` will download a ~90MB model to your cache).*

### 2. Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` to add your Gemini API key (if using Gemini) or switch `LLM_PROVIDER=ollama` (if using local Ollama).

### 3. Run the Server

```bash
uvicorn app.main:app --reload
```

Visit the interactive API docs: **http://localhost:8000/docs**

## 🧪 Testing the API

We've included a mock SOP (`data/sample_sop.txt`) that covers customs delays, webhook failures, and RTOs.

### Step 1: Ingest the SOP

```bash
curl -X 'POST' \
  'http://localhost:8000/ingest' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "file_path": "data/sample_sop.txt"
}'
```

### Step 2: Ask a Question

```bash
curl -X 'POST' \
  'http://localhost:8000/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "question": "What should we do if a shipment is stuck at customs for 3 days?",
  "top_k": 3
}'
```

**Example Response:**
```json
{
  "answer": "If a shipment is stuck at customs for 3 days (which is more than the 48-hour trigger), you should first verify that the Commercial Invoice and AWB match the shipment contents in the logistics portal. Then, contact the designated customs broker via email at customs-broker@yza-logistics.test with the AWB number. Finally, send the 'International Delay Notice' email template to the customer.",
  "sources": [
    "## 1. Customs Warehouse Delays\n\n### 1.1 Trigger\nA shipment crossing international borders has a status stuck at \"Customs Clearance\" for more than 48 hours without update.\n\n### 1.2 Procedure\n- **Initial Verification**: Verify the Commercial Invoice and AWB match the shipment contents in the logistics portal.\n- **Action**: Contact the designated customs broker via email (customs-broker@yza-logistics.test) with the AWB number."
  ]
}
```

## 🧠 Why RAG instead of Fine-Tuning?

For a logistics company like YZA, internal policies (like carrier APIs or RTO rules) can change frequently.
- **Fine-tuning** requires retraining the model every time a policy changes (expensive and slow).
- **RAG** allows you to just upload a new text file. The LLM instantly knows the new rules because it searches the live database before answering. 
