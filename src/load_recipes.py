"""
load_recipes.py — Load cleaned recipes into PostgreSQL with embeddings.

This is the main pipeline script for Phase 3. It:
1. Reads the cleaned CSV from Phase 2
2. Generates embeddings using all-MiniLM-L6-v2
3. Bulk inserts everything into PostgreSQL
4. Creates the HNSW index for fast similarity search

Usage:
    python -m src.load_recipes

Prerequisites:
    - PostgreSQL + pgvector running (docker compose up -d)
    - Schema created (python -m src.db_schema)
    - Clean data exists (data/recipes_clean.csv)
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from dotenv import load_dotenv
import os
import time

from src.embeddings import RecipeEmbedder
from src.db_schema import get_connection, create_schema, create_hnsw_index

load_dotenv()

CLEAN_DATA_PATH = Path("data/recipes_clean.csv")

# ── How many recipes to insert per SQL statement ──
# execute_values sends multiple rows in a single INSERT, which is
# MUCH faster than individual INSERT statements.
# 100 rows per batch is a good balance of speed vs memory.
INSERT_BATCH_SIZE = 100


def load_csv(path: Path) -> pd.DataFrame:
    """Load the cleaned CSV and prepare it for insertion."""
    print(f"Loading cleaned data from {path}...")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} recipes with {len(df.columns)} columns")

    # Replace NaN with None (PostgreSQL expects NULL, not NaN)
    df = df.where(pd.notna(df), None)

    return df


def generate_embeddings(df: pd.DataFrame) -> list[list[float]]:
    """Generate embeddings for all recipe texts."""
    embedder = RecipeEmbedder()

    # Get all recipe texts as a list of strings
    texts = df["recipe_text"].tolist()

    # Generate embeddings in batches
    start_time = time.time()
    embeddings = embedder.embed_batch(texts)
    elapsed = time.time() - start_time

    print(f"  Generated {len(embeddings)} embeddings in {elapsed:.1f} seconds")
    print(f"  ({len(embeddings)/elapsed:.0f} recipes/sec)")
    print(f"  Each embedding: {len(embeddings[0])} dimensions")

    return embeddings


def insert_recipes(df: pd.DataFrame, embeddings: list[list[float]]):
    """
    Bulk insert recipes and embeddings into PostgreSQL.

    WHY execute_values INSTEAD OF execute?
    - execute: sends one INSERT per row → 1,000 round trips to the database
    - execute_values: sends N rows per INSERT → ~10 round trips
    - This is 50-100x faster for bulk loading.

    WHY NOT COPY?
    - COPY is even faster but doesn't handle the vector type well.
    - execute_values is fast enough for 1,000 rows (~2 seconds).
    """
    print(f"Inserting {len(df)} recipes into PostgreSQL...")

    conn = get_connection()
    cur = conn.cursor()

    # Clear existing data (idempotent — safe to re-run)
    cur.execute("DELETE FROM recipes;")
    print("  Cleared existing recipes")

    # Prepare the INSERT statement
    insert_sql = """
        INSERT INTO recipes (
            id, name, description, ingredients, steps,
            minutes, n_steps, n_ingredients,
            calories, protein_pdv,
            cuisine, dietary, meal_type, tags,
            recipe_text, embedding
        ) VALUES %s
        ON CONFLICT (id) DO NOTHING;
    """

    # Build rows as tuples
    def safe_int(val):
        """Convert to int, returning None for NaN/None."""
        try:
            if val is None or pd.isna(val):
                return None
            return int(val)
        except (ValueError, TypeError):
            return None

    def safe_float(val):
        """Convert to float, returning None for NaN/None."""
        try:
            if val is None or pd.isna(val):
                return None
            return float(val)
        except (ValueError, TypeError):
            return None

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        rows.append((
            safe_int(row["id"]) or i,
            row["name"],
            row.get("description"),
            row.get("ingredients"),
            row.get("steps"),
            safe_int(row.get("minutes")),
            safe_int(row.get("n_steps")),
            safe_int(row.get("n_ingredients")),
            safe_float(row.get("calories")),
            safe_float(row.get("protein_pdv")),
            row.get("cuisine"),
            row.get("dietary"),
            row.get("meal_type"),
            row.get("tags_str"),
            row["recipe_text"],
            str(embeddings[i]),  # pgvector expects string format: '[0.1, 0.2, ...]'
        ))

    # Bulk insert in batches
    start_time = time.time()
    total_inserted = 0

    for batch_start in range(0, len(rows), INSERT_BATCH_SIZE):
        batch = rows[batch_start:batch_start + INSERT_BATCH_SIZE]
        execute_values(cur, insert_sql, batch, page_size=INSERT_BATCH_SIZE)
        total_inserted += len(batch)
        print(f"  Inserted {total_inserted}/{len(rows)} recipes...", end="\r")

    elapsed = time.time() - start_time
    print(f"  Inserted {total_inserted} recipes in {elapsed:.1f} seconds       ")

    cur.close()
    conn.close()


def verify_insertion():
    """Verify the data was inserted correctly."""
    print("\nVerifying insertion...")

    conn = get_connection()
    cur = conn.cursor()

    # Count rows
    cur.execute("SELECT COUNT(*) FROM recipes;")
    count = cur.fetchone()[0]
    print(f"  [✓] Total recipes in database: {count}")

    # Check a sample recipe
    cur.execute("SELECT id, name, cuisine, minutes FROM recipes LIMIT 3;")
    samples = cur.fetchall()
    print(f"  [✓] Sample recipes:")
    for recipe_id, name, cuisine, minutes in samples:
        print(f"      - {name} (cuisine: {cuisine}, {minutes} min)")

    # Check embedding dimensions
    cur.execute("SELECT embedding FROM recipes LIMIT 1;")
    sample_embedding = cur.fetchone()[0]
    # pgvector returns the embedding as a string; count the commas + 1
    dim = len(sample_embedding.strip("[]").split(","))
    print(f"  [✓] Embedding dimension: {dim}")

    # Check for NULL embeddings
    cur.execute("SELECT COUNT(*) FROM recipes WHERE embedding IS NULL;")
    null_count = cur.fetchone()[0]
    if null_count == 0:
        print(f"  [✓] No NULL embeddings")
    else:
        print(f"  [⚠] {null_count} recipes have NULL embeddings!")

    cur.close()
    conn.close()


def main():
    print("=" * 60)
    print("  RecipeRAG — Phase 3: Vector Pipeline")
    print("=" * 60)
    print()

    # Check prerequisites
    if not CLEAN_DATA_PATH.exists():
        print(f"ERROR: {CLEAN_DATA_PATH} not found!")
        print("Run Phase 2 first: python -m data_processing.clean_recipes")
        return

    # Step 1: Create schema (idempotent)
    create_schema()
    print()

    # Step 2: Load the cleaned CSV
    df = load_csv(CLEAN_DATA_PATH)
    print()

    # Step 3: Generate embeddings
    embeddings = generate_embeddings(df)
    print()

    # Step 4: Insert into PostgreSQL
    insert_recipes(df, embeddings)

    # Step 5: Create HNSW index (AFTER insertion — this is intentional)
    print()
    create_hnsw_index()

    # Step 6: Verify
    verify_insertion()

    print()
    print("=" * 60)
    print("  DONE! Vector pipeline complete.")
    print("  Run the search test: python -m src.search_test")
    print("=" * 60)


if __name__ == "__main__":
    main()