# RecipeRAG

A Retrieval-Augmented Generation chatbot that answers natural language questions
about recipes using semantic search over a vector database.

## Tech Stack

| Component        | Tool                              | Why                                    |
| ---------------- | --------------------------------- | -------------------------------------- |
| Language         | Python 3.10+                      | ML/AI ecosystem standard               |
| RAG Framework    | LangChain                         | Chains embeddings → retrieval → LLM    |
| Vector Database  | PostgreSQL + pgvector (Docker)    | Production-grade, real SQL filtering   |
| Embeddings       | all-MiniLM-L6-v2 (local)         | Free, no API needed, 384-dim vectors   |
| LLM              | OpenRouter free tier              | OpenAI-compatible API, $0 cost         |
| UI               | Streamlit                         | Chat UI in pure Python                 |

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Git

## Quick Start

### 1. Clone and enter the project
```bash
git clone <your-repo-url>
cd RecipeRAG
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### 5. Start PostgreSQL + pgvector
```bash
docker compose up -d
```

### 6. Verify everything works
```bash
python -m src.verify_setup
```

You should see "ALL CHECKS PASSED."

## Project Structure

```
RecipeRAG/
├── src/                    # Core RAG pipeline
│   ├── init.py
│   └── verify_setup.py    # Phase 1 verification
├── data_processing/        # Data cleaning scripts
│   └── init.py
├── app/                    # Streamlit chat UI
│   └── init.py
├── data/                   # Raw + processed data (gitignored)
├── docker-compose.yml      # PostgreSQL + pgvector
├── requirements.txt        # Pinned dependencies
├── .env.example            # Environment variable template
├── .gitignore
└── README.md
```
## Useful Commands
```bash
# Start the database
docker compose up -d

# Stop the database (data is preserved)
docker compose down

# Stop and DELETE all data
docker compose down -v && rm -rf pgdata/

# View database logs
docker compose logs -f db

# Connect to the database directly
docker exec -it reciperag-db psql -U reciperag -d reciperag
```
