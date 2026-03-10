"""
verify_setup.py — Phase 1 verification script.

Run this after `docker compose up -d` to confirm:
  1. PostgreSQL is reachable
  2. pgvector extension is enabled
  3. Vector operations work (insert + similarity search)

Usage:
    python -m src.verify_setup
"""

import sys
import psycopg2
from dotenv import load_dotenv
import os

# ── Load environment variables from .env ───────────
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "dbname": os.getenv("POSTGRES_DB", "reciperag"),
    "user": os.getenv("POSTGRES_USER", "reciperag"),
    "password": os.getenv("POSTGRES_PASSWORD", "reciperag_dev_password"),
}


def verify_connection(cur):
    """Step 1: Can we talk to the database at all?"""
    cur.execute("SELECT version();")
    version = cur.fetchone()[0]
    print(f"[PASS] Connected to PostgreSQL")
    print(f"       Version: {version[:60]}...")


def verify_pgvector(cur):
    """Step 2: Is the pgvector extension available and enabled?"""
    # Create the extension if it doesn't exist yet.
    # This is idempotent — safe to run multiple times.
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # Confirm it's listed in installed extensions
    cur.execute("SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';")
    result = cur.fetchone()

    if result:
        print(f"[PASS] pgvector extension is enabled (version {result[1]})")
    else:
        print("[FAIL] pgvector extension not found!")
        sys.exit(1)


def verify_vector_operations(cur):
    """Step 3: Can we create a table with vectors, insert data, and run similarity search?"""

    # Clean up from any previous test run
    cur.execute("DROP TABLE IF EXISTS _test_vectors;")

    # Create a test table with a 3-dimensional vector column
    # In Phase 3 we'll use 384 dimensions (all-MiniLM-L6-v2 output size)
    cur.execute("""
        CREATE TABLE _test_vectors (
            id SERIAL PRIMARY KEY,
            label TEXT,
            embedding vector(3)
        );
    """)

    # Insert three test vectors
    test_data = [
        ("apple",  [1.0, 0.0, 0.0]),
        ("banana", [0.9, 0.1, 0.0]),
        ("car",    [0.0, 0.0, 1.0]),
    ]
    for label, vec in test_data:
        cur.execute(
            "INSERT INTO _test_vectors (label, embedding) VALUES (%s, %s);",
            (label, str(vec))
        )

    # Run a cosine similarity search: "what's closest to apple?"
    query_vector = [1.0, 0.0, 0.0]  # Same as "apple"
    cur.execute("""
        SELECT label, 1 - (embedding <=> %s::vector) AS cosine_similarity
        FROM _test_vectors
        ORDER BY embedding <=> %s::vector
        LIMIT 3;
    """, (str(query_vector), str(query_vector)))

    results = cur.fetchall()
    print(f"[PASS] Vector similarity search works!")
    print(f"       Query: {query_vector}")
    print(f"       Results (closest first):")
    for label, similarity in results:
        print(f"         - {label}: similarity = {similarity:.4f}")

    # Clean up test table
    cur.execute("DROP TABLE _test_vectors;")
    print(f"[PASS] Cleaned up test table")


def main():
    print("=" * 55)
    print("  RecipeRAG — Phase 1 Environment Verification")
    print("=" * 55)
    print()

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cur = conn.cursor()

        verify_connection(cur)
        print()
        verify_pgvector(cur)
        print()
        verify_vector_operations(cur)

        cur.close()
        conn.close()

        print()
        print("=" * 55)
        print("  ALL CHECKS PASSED — Phase 1 is complete!")
        print("  You're ready to move on to Phase 2.")
        print("=" * 55)

    except psycopg2.OperationalError as e:
        print(f"[FAIL] Cannot connect to PostgreSQL!")
        print(f"       Error: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Is Docker running?       → docker compose ps")
        print("  2. Is the container healthy? → docker compose logs db")
        print("  3. Is port 5432 open?        → lsof -i :5432")
        sys.exit(1)


if __name__ == "__main__":
    main()