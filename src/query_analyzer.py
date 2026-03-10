"""
query_analyzer.py — Extract structured filters from natural language queries.

Parses user queries to detect:
- Cuisine preferences (e.g., "Italian", "Mexican", "Asian")
- Dietary restrictions (e.g., "vegetarian", "vegan", "gluten-free")
- Meal type (e.g., "breakfast", "dinner", "dessert")
- Time constraints (e.g., "under 30 minutes", "quick")
- Calorie constraints (e.g., "low calorie", "under 500 calories")

Usage:
    from src.query_analyzer import QueryAnalyzer
    analyzer = QueryAnalyzer()
    filters = analyzer.analyze("quick vegetarian pasta under 30 min")
"""

import re


class QueryAnalyzer:
    """
    Parse natural language queries into structured search filters.

    This is a rule-based approach using regex patterns.
    For production, you'd likely use an LLM to extract these,
    but regex is faster, free, and deterministic.
    """

    # ── Cuisine keywords ──────────────────────────
    CUISINE_MAP = {
        "italian": "italian",
        "mexican": "mexican",
        "chinese": "chinese",
        "indian": "indian",
        "japanese": "japanese",
        "thai": "thai",
        "french": "french",
        "greek": "greek",
        "mediterranean": "mediterranean",
        "korean": "korean",
        "vietnamese": "vietnamese",
        "spanish": "spanish",
        "american": "north-american",
        "north american": "north-american",
        "asian": "asian",
        "european": "european",
        "caribbean": "caribbean",
        "african": "african",
        "middle eastern": "middle-eastern",
    }

    # ── Dietary keywords ──────────────────────────
    DIETARY_KEYWORDS = [
        "vegetarian", "vegan", "gluten-free", "gluten free",
        "dairy-free", "dairy free", "low-carb", "low carb",
        "low-fat", "low fat", "low-sodium", "low sodium",
        "low-calorie", "low calorie", "sugar-free", "sugar free",
        "keto", "paleo", "whole30", "kosher", "halal",
    ]

    # ── Meal type keywords ────────────────────────
    MEAL_MAP = {
        "breakfast": "breakfast",
        "brunch": "brunch",
        "lunch": "lunch",
        "dinner": "dinner",
        "dessert": "desserts",
        "desserts": "desserts",
        "snack": "snacks",
        "snacks": "snacks",
        "appetizer": "appetizers",
        "appetizers": "appetizers",
        "side dish": "side-dishes",
        "side dishes": "side-dishes",
    }

    # ── Time patterns ─────────────────────────────
    TIME_PATTERNS = [
        # "under 30 minutes", "less than 30 min", "within 30 minutes"
        r"(?:under|less than|within|max|up to)\s+(\d+)\s*(?:min|minutes|mins)",
        # "30 minutes or less", "30 min or under"
        r"(\d+)\s*(?:min|minutes|mins)\s+(?:or less|or under|max)",
        # "quick" = 30 min, "fast" = 20 min
    ]

    QUICK_KEYWORDS = {
        "quick": 30,
        "fast": 20,
        "easy": 45,     # "easy" often implies not too long
        "simple": 45,
    }

    # ── Calorie patterns ─────────────────────────
    CALORIE_PATTERNS = [
        # "under 500 calories", "less than 300 cal"
        r"(?:under|less than|max|below)\s+(\d+)\s*(?:cal|calories|kcal)",
        # "500 calories or less"
        r"(\d+)\s*(?:cal|calories|kcal)\s+(?:or less|or under|max)",
    ]

    def analyze(self, query: str) -> dict:
        """
        Analyze a natural language query and extract structured filters.

        Returns a dict with:
            search_query: str   — The cleaned query for embedding search
            cuisine: str|None   — Detected cuisine
            dietary: str|None   — Detected dietary restriction
            meal_type: str|None — Detected meal type
            max_minutes: int|None — Maximum cook time
            max_calories: float|None — Maximum calories
        """
        query_lower = query.lower().strip()

        # Extract each filter type
        cuisine = self._extract_cuisine(query_lower)
        dietary = self._extract_dietary(query_lower)
        meal_type = self._extract_meal_type(query_lower)
        max_minutes = self._extract_time(query_lower)
        max_calories = self._extract_calories(query_lower)

        # Build a cleaned search query (remove filter keywords to
        # let the embedding focus on the actual food being searched)
        search_query = self._clean_query(query_lower, cuisine, dietary, meal_type)

        return {
            "search_query": search_query,
            "cuisine": cuisine,
            "dietary": dietary,
            "meal_type": meal_type,
            "max_minutes": max_minutes,
            "max_calories": max_calories,
        }

    def _extract_cuisine(self, query: str) -> str | None:
        """Find cuisine keywords in the query."""
        for keyword, value in self.CUISINE_MAP.items():
            if keyword in query:
                return value
        return None

    def _extract_dietary(self, query: str) -> str | None:
        """Find dietary restriction keywords in the query."""
        for keyword in self.DIETARY_KEYWORDS:
            if keyword in query:
                # Normalize to hyphenated form
                return keyword.replace(" ", "-")
        return None

    def _extract_meal_type(self, query: str) -> str | None:
        """Find meal type keywords in the query."""
        for keyword, value in self.MEAL_MAP.items():
            if keyword in query:
                return value
        return None

    def _extract_time(self, query: str) -> int | None:
        """Extract time constraints from the query."""
        # Try explicit patterns first ("under 30 minutes")
        for pattern in self.TIME_PATTERNS:
            match = re.search(pattern, query)
            if match:
                return int(match.group(1))

        # Try quick keywords ("quick", "fast")
        for keyword, minutes in self.QUICK_KEYWORDS.items():
            # Only match as a standalone word (not part of another word)
            if re.search(r'\b' + keyword + r'\b', query):
                return minutes

        return None

    def _extract_calories(self, query: str) -> float | None:
        """Extract calorie constraints from the query."""
        for pattern in self.CALORIE_PATTERNS:
            match = re.search(pattern, query)
            if match:
                return float(match.group(1))

        # "low calorie" without a number → default to 400
        if "low calorie" in query or "low-calorie" in query:
            return 400.0

        return None

    def _clean_query(self, query: str, cuisine, dietary, meal_type) -> str:
        """
        Remove filter keywords from the query so the embedding
        focuses on the actual food being searched.

        Example:
            "quick vegetarian Italian dinner under 30 min"
            → "dinner"  (after removing "quick", "vegetarian", "Italian", "under 30 min")

        Wait — that's too aggressive. Let's keep food-relevant words.
        Actually, we keep the full query for embedding because the model
        benefits from context. We only clean out the noise.
        """
        cleaned = query

        # Remove time phrases
        for pattern in self.TIME_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned)

        # Remove calorie phrases
        for pattern in self.CALORIE_PATTERNS:
            cleaned = re.sub(pattern, "", cleaned)

        # Remove filler words
        fillers = ["find me", "give me", "show me", "suggest", "recommend",
                   "i want", "i need", "looking for", "search for",
                   "can you", "please", "some", "a few"]
        for filler in fillers:
            cleaned = cleaned.replace(filler, "")

        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # If cleaning removed everything, fall back to original
        if len(cleaned) < 3:
            return query

        return cleaned


def main():
    """Test the query analyzer with sample queries."""
    analyzer = QueryAnalyzer()

    test_queries = [
        "Find me a quick chicken stir fry",
        "vegetarian pasta under 30 minutes",
        "low calorie Italian dinner",
        "easy Mexican breakfast ideas",
        "vegan dessert under 500 calories",
        "Thai curry for dinner",
        "gluten-free snacks that are quick to make",
        "chocolate cake for a birthday party",
        "healthy lunch under 20 minutes",
        "Korean BBQ recipes",
    ]

    print("=" * 60)
    print("  RecipeRAG — Query Analyzer Test")
    print("=" * 60)

    for query in test_queries:
        result = analyzer.analyze(query)
        print(f"\n  Query: \"{query}\"")
        print(f"  → search_query: \"{result['search_query']}\"")

        filters = {k: v for k, v in result.items()
                   if k != "search_query" and v is not None}
        if filters:
            print(f"  → filters: {filters}")
        else:
            print(f"  → filters: (none)")

    print(f"\n{'═' * 60}")


if __name__ == "__main__":
    main()