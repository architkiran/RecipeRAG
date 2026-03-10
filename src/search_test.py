"""
search_test.py — Test semantic search against the recipes database.

Runs a series of test queries to verify that:
1. Basic similarity search works
2. Metadata filtering works
3. Results are sensible (chicken queries return chicken recipes, etc.)

Usage:
    python -m src.search_test
"""

from src.retriever import RecipeRetriever


def print_results(query: str, results: list[dict], filters: dict = None):
    """Pretty-print search results."""
    print(f"\n{'─' * 60}")
    print(f"  Query: \"{query}\"")
    if filters:
        filter_str = ", ".join(f"{k}={v}" for k, v in filters.items() if v)
        print(f"  Filters: {filter_str}")
    print(f"  Results: {len(results)}")
    print(f"{'─' * 60}")

    for i, r in enumerate(results):
        sim_bar = "█" * int(r["similarity"] * 20)
        print(f"\n  [{i+1}] {r['name']}")
        print(f"      Similarity: {r['similarity']:.4f} {sim_bar}")
        print(f"      Cuisine: {r['cuisine'] or 'N/A'} | "
              f"Time: {r['minutes'] or 'N/A'} min | "
              f"Calories: {r['calories'] or 'N/A'}")
        if r.get("dietary"):
            print(f"      Dietary: {r['dietary']}")


def main():
    print("=" * 60)
    print("  RecipeRAG — Phase 3: Search Verification")
    print("=" * 60)

    retriever = RecipeRetriever()

    # ── Test 1: Basic semantic search ──────────────
    print_results(
        "quick chicken stir fry",
        retriever.search("quick chicken stir fry", top_k=5)
    )

    # ── Test 2: Different cuisine ─────────────────
    print_results(
        "Italian pasta with tomato sauce",
        retriever.search("Italian pasta with tomato sauce", top_k=5)
    )

    # ── Test 3: Dessert search ────────────────────
    print_results(
        "chocolate cake for a birthday",
        retriever.search("chocolate cake for a birthday", top_k=5)
    )

    # ── Test 4: With time filter ──────────────────
    filters = {"max_minutes": 30}
    print_results(
        "easy weeknight dinner",
        retriever.search("easy weeknight dinner", top_k=5, **filters),
        filters
    )

    # ── Test 5: With dietary filter ───────────────
    filters = {"dietary": "vegetarian"}
    print_results(
        "healthy lunch ideas",
        retriever.search("healthy lunch ideas", top_k=5, **filters),
        filters
    )

    # ── Test 6: Combined filters ──────────────────
    filters = {"max_minutes": 45, "max_calories": 500}
    print_results(
        "light and healthy dinner",
        retriever.search("light and healthy dinner", top_k=5, **filters),
        filters
    )

    print(f"\n{'═' * 60}")
    print("  Search verification complete!")
    print("  If results look sensible, you're ready for Phase 4.")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()