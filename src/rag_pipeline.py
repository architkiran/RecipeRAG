"""
rag_pipeline.py — The complete RAG pipeline tying everything together.

Flow: User query → analyze → retrieve → format context → LLM → response

Usage:
    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    response = pipeline.ask("Find me a quick vegetarian dinner")
"""

from src.query_analyzer import QueryAnalyzer
from src.retriever import RecipeRetriever
from src.llm import RecipeLLM


class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation pipeline for recipe queries.

    Orchestrates:
    1. Query analysis (extract filters from natural language)
    2. Retrieval (semantic search + metadata filtering)
    3. Context formatting (turn recipes into readable prompt context)
    4. LLM generation (produce the final answer)
    """

    def __init__(self):
        """Initialize all pipeline components."""
        print("Initializing RAG Pipeline...")
        self.analyzer = QueryAnalyzer()
        self.retriever = RecipeRetriever()
        self.llm = RecipeLLM()
        print("  [✓] Pipeline ready!\n")

    def ask(self, query: str, top_k: int = 5) -> dict:
        """
        Process a user query through the full RAG pipeline.

        Parameters:
            query: Natural language question about recipes
            top_k: Number of recipes to retrieve

        Returns:
            dict with:
                answer: str — The LLM's response
                recipes: list — The retrieved recipes (for citations)
                filters: dict — The extracted filters (for transparency)
        """
        # ── Step 1: Analyze the query ──────────────
        filters = self.analyzer.analyze(query)

        # ── Step 2: Retrieve relevant recipes ──────
        recipes = self.retriever.search(
            query=filters["search_query"],
            top_k=top_k,
            cuisine=filters["cuisine"],
            dietary=filters["dietary"],
            meal_type=filters["meal_type"],
            max_minutes=filters["max_minutes"],
            max_calories=filters["max_calories"],
        )

        # ── Step 3: Format context for the LLM ────
        if not recipes:
            context = "No recipes found matching the query and filters."
        else:
            context = self._format_context(recipes)

        # ── Step 4: Generate answer with LLM ──────
        answer = self.llm.generate(query, context)

        return {
            "answer": answer,
            "recipes": recipes,
            "filters": filters,
        }

    def _format_context(self, recipes: list[dict]) -> str:
        """
        Format retrieved recipes into a readable context string for the LLM.

        WHY THIS FORMAT?
        The LLM needs to quickly understand each recipe to answer the question.
        We give it structured but readable text — not raw JSON or SQL rows.
        Including the recipe ID lets the LLM cite specific recipes.
        """
        parts = []
        for i, recipe in enumerate(recipes, 1):
            parts.append(f"""
Recipe {i}: {recipe['name']}
- Recipe ID: {recipe['id']}
- Cook time: {recipe['minutes'] or 'Unknown'} minutes
- Calories: {recipe['calories'] or 'Unknown'}
- Cuisine: {recipe['cuisine'] or 'Not specified'}
- Dietary: {recipe['dietary'] or 'None specified'}
- Meal type: {recipe['meal_type'] or 'Not specified'}
- Ingredients: {recipe['ingredients'][:300] if recipe['ingredients'] else 'Not available'}
- Similarity score: {recipe['similarity']}
""".strip())

        return "\n\n".join(parts)


def main():
    """Interactive test of the RAG pipeline."""
    pipeline = RAGPipeline()

    print("=" * 60)
    print("  RecipeRAG — Interactive Pipeline Test")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        print()
        query = input("  You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue

        result = pipeline.ask(query)

        # Show filters detected
        active_filters = {k: v for k, v in result["filters"].items()
                         if k != "search_query" and v is not None}
        if active_filters:
            print(f"\n  [Filters: {active_filters}]")

        # Show answer
        print(f"\n  RecipeRAG: {result['answer']}")

        # Show citations
        if result["recipes"]:
            print(f"\n  📚 Sources ({len(result['recipes'])} recipes):")
            for r in result["recipes"]:
                print(f"     - {r['name']} (similarity: {r['similarity']:.2f})")


if __name__ == "__main__":
    main()