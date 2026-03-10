"""
retriever.py — Semantic search over the recipes database.

This module is the heart of the RAG pipeline. It:
1. Embeds the user's query using the same model that embedded the recipes
2. Runs a pgvector similarity search with optional metadata filters
3. Returns ranked results with similarity scores

Usage:
    Called by the chatbot in Phase 4.
    Can also be used standalone — see search_test.py.
"""

import psycopg2
from src.embeddings import RecipeEmbedder
from src.db_schema import get_connection


class RecipeRetriever:
    """
    Retrieve recipes from PostgreSQL using semantic search + metadata filters.

    WHY A DEDICATED CLASS?
    - Holds a single instance of the embedding model (loaded once)
    - Manages database connection
    - Encapsulates the SQL query construction (including dynamic WHERE clauses)
    - Easy to swap out for a different vector DB later (just rewrite this class)
    """

    def __init__(self):
        """Initialize the retriever with an embedding model."""
        self.embedder = RecipeEmbedder()

    def search(
        self,
        query: str,
        top_k: int = 5,
        cuisine: str = None,
        dietary: str = None,
        meal_type: str = None,
        max_minutes: int = None,
        max_calories: float = None,
    ) -> list[dict]:
        """
        Search for recipes matching the query with optional filters.

        Parameters:
            query:        Natural language search query
            top_k:        Number of results to return (default: 5)
            cuisine:      Filter by cuisine (e.g., "italian")
            dietary:      Filter by dietary tag (e.g., "vegetarian")
            meal_type:    Filter by meal type (e.g., "dinner")
            max_minutes:  Maximum cook time in minutes
            max_calories: Maximum calories per serving

        Returns:
            List of dicts, each containing recipe data + similarity score.

        HOW THE SQL WORKS:
            1. We embed the query into a 384-dim vector
            2. The <=> operator computes cosine distance between the query
               vector and every recipe's embedding
            3. WHERE clauses filter on metadata BEFORE ranking
            4. ORDER BY distance ASC gives us closest matches first
            5. 1 - distance = cosine similarity (0 to 1 scale)

        WHY FILTER BEFORE RANKING?
        Applying WHERE clauses BEFORE the similarity search means PostgreSQL
        only computes distances for matching rows. This is faster and ensures
        ALL results satisfy the filters (not just the top-k).
        """
        # Step 1: Embed the query
        query_embedding = self.embedder.embed_single(query)

        # Step 2: Build the SQL query with dynamic filters
        conditions = []
        params = []

        if cuisine:
            conditions.append("cuisine = %s")
            params.append(cuisine.lower())

        if dietary:
            # Use LIKE for partial matching (e.g., "vegetarian" in "vegetarian, low-fat")
            conditions.append("dietary LIKE %s")
            params.append(f"%{dietary.lower()}%")

        if meal_type:
            conditions.append("meal_type = %s")
            params.append(meal_type.lower())

        if max_minutes:
            conditions.append("minutes <= %s")
            params.append(max_minutes)

        if max_calories:
            conditions.append("calories <= %s")
            params.append(max_calories)

        # Build WHERE clause
        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # The main similarity search query
        sql = f"""
            SELECT
                id,
                name,
                description,
                ingredients,
                steps,
                minutes,
                calories,
                cuisine,
                dietary,
                meal_type,
                recipe_text,
                1 - (embedding <=> %s::vector) AS similarity
            FROM recipes
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        # Add the embedding (used twice: once for SELECT, once for ORDER BY)
        # and top_k to the params
        all_params = [str(query_embedding)] + params + [str(query_embedding), top_k]

        # Wait — the params order matters! The embedding for the SELECT column
        # comes before the WHERE params, then the embedding for ORDER BY comes
        # after. Let me restructure:

        sql = f"""
            SELECT
                id,
                name,
                description,
                ingredients,
                steps,
                minutes,
                calories,
                cuisine,
                dietary,
                meal_type,
                recipe_text,
                1 - (embedding <=> %s::vector) AS similarity
            FROM recipes
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """

        # Build params in correct order:
        # %s (SELECT embedding) → WHERE params → %s (ORDER BY embedding) → %s (LIMIT)
        all_params = [str(query_embedding)] + params + [str(query_embedding), top_k]

        # Step 3: Execute the query
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(sql, all_params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        # Step 4: Format results
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "ingredients": row[3],
                "steps": row[4],
                "minutes": row[5],
                "calories": row[6],
                "cuisine": row[7],
                "dietary": row[8],
                "meal_type": row[9],
                "recipe_text": row[10],
                "similarity": round(float(row[11]), 4),
            })

        return results