You are helping me build a personal learning project called "RecipeRAG" — a private AI chatbot that lets me query a recipe dataset using natural language. This is a practice project to learn the RAG pipeline before I build a similar system for a client using support ticket data.

**What the bot should do:**
- Semantic search over recipes (e.g., "Find me a quick chicken dinner under 30 minutes")
- Analytical/summary questions (e.g., "What are the most common spices in Indian recipes?")
- Metadata filtering from natural language (e.g., user says "vegetarian" → auto-filter on dietary tags)
- Cite which recipes it used to generate each answer

**Tech stack (mandatory — everything must be free):**
- Python
- Pandas & Regex for data cleaning
- LangChain for RAG orchestration
- PostgreSQL with pgvector extension for vector storage (run locally via Docker)
- HuggingFace sentence-transformers (all-MiniLM-L6-v2) for embeddings — runs locally, free
- OpenRouter (free tier) as the LLM gateway — use a free model like `mistralai/mistral-7b-instruct:free` or `meta-llama/llama-3-8b-instruct:free` (check OpenRouter's free model list at runtime and pick the best available one)
- Streamlit for the chat UI

**Why PostgreSQL + pgvector instead of ChromaDB:**
- I want to learn pgvector because my client project will likely use a managed PostgreSQL or enterprise vector DB
- pgvector gives me real SQL filtering on metadata (WHERE clauses) which is more powerful and production-realistic than ChromaDB's filter syntax
- It also teaches me schema design, indexing, and database management — skills that transfer directly

**OpenRouter setup:**
- Use the OpenRouter API (https://openrouter.ai/api/v1) which is OpenAI-compatible
- Integrate via LangChain's ChatOpenAI class with `base_url="https://openrouter.ai/api/v1"` and the OpenRouter API key
- Store the API key in a `.env` file, never hardcode it
- Handle rate limits and free-tier constraints gracefully (retry logic, fallback model)

**Dataset:**
- Use the Food.com Recipes dataset from Kaggle (or any publicly available recipe dataset with fields like: name, ingredients, steps, tags, cook time, nutrition)
- Start with a subset of ~500–1,000 recipes to keep things fast

**Project structure:**
```
/src              → Core pipeline (embeddings, retrieval, LLM chain)
/data_processing  → Cleaning scripts
/app              → Streamlit UI
/data             → Raw + processed data (gitignored)
docker-compose.yml → PostgreSQL + pgvector container
requirements.txt
.env.example      → Template with OPENROUTER_API_KEY=your_key_here
.gitignore
README.md
```

**Build this in 4 phases:**

Phase 1 — Environment setup: Initialize the repo structure, virtual environment, requirements.txt, .env.example, and a docker-compose.yml that spins up PostgreSQL with the pgvector extension. Write a README with full setup instructions. Verify by connecting to the database and confirming pgvector is enabled.

Phase 2 — Data cleaning: Download and clean the dataset. Strip HTML, normalize ingredient formats, parse cook/prep times into numeric minutes, remove duplicates, and create a combined `recipe_text` field (name + ingredients + steps) optimized for embedding. Export as clean CSV.

Phase 3 — Vector pipeline: Design a PostgreSQL schema with a recipes table that has columns for all metadata (recipe_name, cuisine, cook_time_minutes, tags, calories) plus a vector column for the embedding. Generate embeddings from `recipe_text` using sentence-transformers locally. Bulk insert into PostgreSQL. Create an IVFFlat or HNSW index on the vector column. Verify by running a raw similarity search query.

Phase 4 — Chatbot: Wire up the RAG chain with LangChain — query → generate embedding → SQL similarity search with optional WHERE filters → pass retrieved recipes as context to the LLM via OpenRouter → generate answer. Build a Streamlit chat UI with message history and recipe citations shown under each answer. Add natural-language-to-metadata-filter logic so the bot auto-detects time, cuisine, and dietary constraints from the query and translates them into SQL WHERE clauses.

**Important guidelines:**
- Walk me through each phase step by step. Don't dump all the code at once.
- Explain your decisions as you go so I learn the "why," not just the "how."
- After each phase, give me a way to verify it works before moving on.
- Keep the code clean, commented, and modular so I can later swap pgvector for Vertex AI Vector Search and OpenRouter for Vertex AI's Gemini when I move to the client project.
- Every tool and service used must be free. No paid tiers, no trial-that-expires. If something costs money, flag it and suggest a free alternative.

Start with Phase 1.
