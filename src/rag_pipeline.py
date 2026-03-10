"""
rag_pipeline.py — The complete RAG pipeline with conversation memory.

Flow: User query + history → analyze → retrieve → format context → LLM → response
"""

from src.query_analyzer import QueryAnalyzer
from src.retriever import RecipeRetriever
from src.llm import RecipeLLM


class RAGPipeline:
    """
    Complete RAG pipeline with conversation memory support.
    """

    def __init__(self):
        print("Initializing RAG Pipeline...")
        self.analyzer = QueryAnalyzer()
        self.retriever = RecipeRetriever()
        self.llm = RecipeLLM()
        print("  [✓] Pipeline ready!\n")

    def ask(self, query: str, top_k: int = 5, history: list[dict] = None) -> dict:
        """
        Process a user query through the full RAG pipeline.

        Parameters:
            query: Natural language question about recipes
            top_k: Number of recipes to retrieve
            history: Previous conversation messages for follow-up support

        Returns:
            dict with answer, recipes, and filters
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

        # ── Step 4: Generate answer with LLM + history ──
        answer = self.llm.generate(query, context, history=history)

        return {
            "answer": answer,
            "recipes": recipes,
            "filters": filters,
        }

    def _format_context(self, recipes: list[dict]) -> str:
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
    """Interactive test with conversation memory."""
    pipeline = RAGPipeline()

    print("=" * 60)
    print("  RecipeRAG — Interactive Pipeline Test (with memory)")
    print("  Type 'quit' to exit, 'clear' to reset history")
    print("=" * 60)

    history = []

    while True:
        print()
        query = input("  You: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if query.lower() == "clear":
            history = []
            print("  [History cleared]")
            continue
        if not query:
            continue

        result = pipeline.ask(query, history=history)

        active_filters = {k: v for k, v in result["filters"].items()
                         if k != "search_query" and v is not None}
        if active_filters:
            print(f"\n  [Filters: {active_filters}]")

        print(f"\n  RecipeRAG: {result['answer']}")

        if result["recipes"]:
            print(f"\n  📚 Sources ({len(result['recipes'])} recipes):")
            for r in result["recipes"]:
                print(f"     - {r['name']} (similarity: {r['similarity']:.2f})")

        # Add to conversation history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result["answer"]})


if __name__ == "__main__":
    main()