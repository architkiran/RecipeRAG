"""
db_schema.py — Database schema definition and management.

Defines the recipes table with:
- Metadata columns (name, cuisine, cook time, etc.) for SQL filtering
- A vector(384) column for semantic similarity search
- An HNSW index for fast approximate nearest-neighbor queries

Supports both Neon (via DATABASE_URL) and local PostgreSQL.

Usage:
    python -m src.db_schema          # Creates/resets the table
    python -m src.db_schema --drop   # Drops the table entirely
"""

import sys
import psycopg2
from urllib.parse import urlparse
from dotenv import load_dotenv
import os

load_dotenv()


def parse_database_url(url: str) -> dict:
    """Parse a PostgreSQL connection URL into psycopg2 config dict."""
    parsed = urlparse(url)
    return {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip("/"),
        "user": parsed.username,
        "password": parsed.password,
    }


# Try to use Neon DATABASE_URL first, fallback to individual POSTGRES_* variables
DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL:
    DB_CONFIG = parse_database_url(DATABASE_URL)
else:
    # Fallback for local PostgreSQL
    DB_CONFIG = {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname": os.getenv("POSTGRES_DB", "reciperag"),
        "user": os.getenv("POSTGRES_USER", "reciperag"),
        "password": os.getenv("POSTGRES_PASSWORD", "reciperag_dev_password"),
    }

# ── Embedding dimensions (must match your model) ──
# all-MiniLM-L6-v2 outputs 384-dimensional vectors
EMBEDDING_DIM = 384

# ═══════════════════════════════════════════════════
# SCHEMA DEFINITION
# ═══════════════════════════════════════════════════

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS recipes (
    -- ── Identity ──────────────────────────────────
    id              INTEGER PRIMARY KEY,        -- Original Food.com recipe ID
    name            TEXT NOT NULL,               -- Recipe name (for display)
    description     TEXT,                        -- Short description

    -- ── Ingredients & Steps (stored as text) ─────
    ingredients     TEXT,                        -- Stringified list of ingredients
    steps           TEXT,                        -- Stringified list of steps

    -- ── Filterable Metadata ──────────────────────
    -- These columns support WHERE clauses for metadata filtering.
    -- When a user says "vegetarian Italian under 30 min", we translate
    -- that into: WHERE dietary LIKE '%vegetarian%'
    --            AND cuisine = 'italian'
    --            AND minutes <= 30
    minutes         INTEGER,                    -- Cook time in minutes
    n_steps         INTEGER,                    -- Number of steps
    n_ingredients   INTEGER,                    -- Number of ingredients
    calories        REAL,                       -- Calorie count
    protein_pdv     REAL,                       -- Protein % daily value
    cuisine         TEXT,                       -- e.g., 'italian', 'mexican'
    dietary         TEXT,                       -- e.g., 'vegetarian, low-fat'
    meal_type       TEXT,                       -- e.g., 'dinner', 'dessert'
    tags            TEXT,                       -- All tags, comma-separated

    -- ── The Text That Was Embedded ───────────────
    recipe_text     TEXT NOT NULL,              -- The concatenated text field

    -- ── Vector Embedding ─────────────────────────
    -- This is the 384-dimensional vector from all-MiniLM-L6-v2.
    -- pgvector stores it as a native vector type for fast similarity search.
    embedding       vector({EMBEDDING_DIM})     -- The semantic embedding
);
"""

# ── HNSW Index ─────────────────────────────────────
# HNSW = Hierarchical Navigable Small World graph
#
# Parameters:
#   m = 16         → Each node connects to 16 neighbors. Higher = more accurate
#                    but more memory. 16 is the default and works well up to ~1M rows.
#   ef_construction = 64  → How many candidates to consider while building the graph.
#                           Higher = slower build but better quality. 64 is default.
#
# vector_cosine_ops → Tells pgvector to use cosine distance for this index.
#                     This matches how we'll query (using <=> operator).
#
# WHY NOT CREATE THE INDEX FIRST AND THEN INSERT?
# For HNSW, it's actually better to insert data FIRST, then create the index.
# Building the index with data present produces a better graph structure.
# (This is opposite to B-tree indexes where you usually create first.)

CREATE_INDEX_SQL = f"""
CREATE INDEX IF NOT EXISTS idx_recipes_embedding
ON recipes
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
"""

# ── Metadata Indexes ──────────────────────────────
# These speed up WHERE clause filtering on common columns.
CREATE_METADATA_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_recipes_cuisine ON recipes (cuisine);
CREATE INDEX IF NOT EXISTS idx_recipes_meal_type ON recipes (meal_type);
CREATE INDEX IF NOT EXISTS idx_recipes_minutes ON recipes (minutes);
CREATE INDEX IF NOT EXISTS idx_recipes_calories ON recipes (calories);
"""

DROP_TABLE_SQL = "DROP TABLE IF EXISTS recipes CASCADE;"


def get_connection():
    """Create and return a database connection."""
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True
    return conn


def create_schema():
    """Create the recipes table and indexes."""
    print("Creating database schema...")

    conn = get_connection()
    cur = conn.cursor()

    # Enable pgvector
    cur.execute(CREATE_EXTENSION_SQL)
    print("  [✓] pgvector extension enabled")

    # Create the table
    cur.execute(CREATE_TABLE_SQL)
    print(f"  [✓] recipes table created (embedding dimension: {EMBEDDING_DIM})")

    # Create metadata indexes
    cur.execute(CREATE_METADATA_INDEXES_SQL)
    print("  [✓] Metadata indexes created (cuisine, meal_type, minutes, calories)")

    # Note: We DON'T create the HNSW index here.
    # It's better to insert data first, then build the index.
    # The load_recipes.py script will create it after insertion.

    cur.close()
    conn.close()
    print("  Schema ready!")


def drop_schema():
    """Drop the recipes table entirely."""
    print("Dropping recipes table...")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(DROP_TABLE_SQL)
    cur.close()
    conn.close()
    print("  [✓] Table dropped")


def create_hnsw_index():
    """Create the HNSW index AFTER data has been inserted."""
    print("Creating HNSW index on embedding column...")
    print("  (This may take 10-30 seconds depending on data size)")

    conn = get_connection()
    cur = conn.cursor()
    cur.execute(CREATE_INDEX_SQL)
    cur.close()
    conn.close()
    print("  [✓] HNSW index created")


def main():
    if "--drop" in sys.argv:
        drop_schema()
    else:
        create_schema()

    # Verify by checking if the table exists
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = 'recipes'
        ORDER BY ordinal_position;
    """)
    columns = cur.fetchall()

    if columns:
        print(f"\n  Table 'recipes' has {len(columns)} columns:")
        for col_name, col_type in columns:
            print(f"    - {col_name}: {col_type}")
    else:
        print("\n  Table 'recipes' does not exist (was it dropped?)")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()