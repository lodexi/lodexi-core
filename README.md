<div align="center">
  <br />
  <h1>LODEXI Core Engine</h1>
  <p>
    <strong>High-Performance, Multi-Tenant Knowledge Indexing & Grounded Retrieval Engine</strong>
  </p>
</div>

<br />

## Overview

**LODEXI Core** is the intelligence backend of the LODEXI ecosystem. It is a decoupled, ultra-fast RAG (Retrieval-Augmented Generation) engine built with **Python**, **FastAPI**, and **Qdrant Vector Database**.

It provides semantic search, document ingestion, and grounded conversational QA capabilities for the `lodexi-portal` and any other external client applications.

---

## Key Features

1. **Strict Multi-Tenant Isolation**: Enforces tenant boundary partitioning via `X-API-Key` validation and Qdrant payload filtering. Zero cross-tenant data leakage.
2. **Dual-Mode Serving**:
   - **`POST /v1/search`**: Pure semantic vector retrieval returning ranked chunks, scores, and metadata without LLM synthesis.
   - **`POST /v1/ask`**: Grounded conversational synthesis providing factual answers with verifiable citations and source links.
3. **Decoupled & Language-Agnostic**: Easily connect any web app (Next.js, Laravel, Go, Node.js) via simple JSON REST requests.
4. **Visual Dashboard**: Includes a built-in `dashboard.html` visualizer matching the LODEXI branding aesthetics.
5. **Flexible Runtime**: Runs 100% locally with embedded Qdrant (zero Docker required) or scaled via Docker Compose.

---

## Project Architecture

```text
lodexi-core/
├── requirements.txt            # Python dependencies
├── .env.example                # Configuration template
├── src/
│   ├── main.py                 # FastAPI application entrypoint
│   ├── config.py               # Pydantic environment settings
│   ├── core/                   # Security, Vector Store, LLM configs
│   ├── models/                 # Request/Response Pydantic schemas
│   ├── templates/              # Visual UI Dashboard
│   └── api/                    # v1 REST Endpoints
└── tests/                      # Automated Pytest suite
```

---

## 🛠️ Quickstart Guide

### 1. Setup Environment

```bash
git clone https://github.com/lodexi/core.git lodexi-core
cd lodexi-core

# Create and activate virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment settings
cp .env.example .env
```

### 2. Run the Engine

```bash
uvicorn src.main:app --reload --port 8000
```

- **Visual Dashboard**: [http://localhost:8000](http://localhost:8000)
- **Swagger API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Run Automated Tests

```bash
pytest
```

---

## API Integration Examples

### Example: LODEXI Portal (PHP/Laravel)
```php
// Call LODEXI Core Grounded QA
$response = Http::withHeaders([
    'X-API-Key' => 'key_portal_secret_123',
])->post('http://localhost:8000/v1/ask', [
    'question' => 'How to setup multi-tenancy?',
    'limit' => 4,
]);

$answer = $response->json()['answer'];
$citations = $response->json()['citations'];
```

---

## License

Copyright © 2026 LODEXI. All rights reserved.

This software is proprietary. You may not use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software without explicit written permission.
