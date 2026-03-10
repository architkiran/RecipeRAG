"""
llm.py — LLM integration via OpenRouter.

Connects to OpenRouter's free tier using the OpenAI-compatible API.
Handles retries, rate limits, and model fallback.

Usage:
    from src.llm import RecipeLLM
    llm = RecipeLLM()
    response = llm.generate("What can I make with chicken and rice?", context="...")
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Primary and fallback models (all free on OpenRouter)
PRIMARY_MODEL = os.getenv("LLM_MODEL", "openrouter/free")
FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]

# ── System Prompt ──────────────────────────────────
# This tells the LLM how to behave. It's critical for getting good answers.
SYSTEM_PROMPT = """You are RecipeRAG, a helpful cooking assistant that answers questions about recipes.

RULES:
1. Base your answers ONLY on the recipes provided in the context below. Do not make up recipes.
2. If the context doesn't contain relevant recipes, say so honestly.
3. When referencing a recipe, always cite it by name.
4. Keep answers conversational, helpful, and concise (2-4 paragraphs max).
5. If asked about cook times, ingredients, or steps, be specific using the data provided.
6. Format recipe names in bold when mentioning them.

When listing recipes, use this format:
- **Recipe Name** (cook time, calories) — brief description
"""


class RecipeLLM:
    """
    LLM wrapper for generating answers from recipe context.

    Uses OpenRouter's free tier with automatic retry and model fallback.
    """

    def __init__(self):
        """Initialize the OpenRouter client."""
        if not OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY not found in .env file!\n"
                "Get a free key at: https://openrouter.ai/keys"
            )

        self.client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=OPENROUTER_API_KEY,
        )
        self.model = PRIMARY_MODEL
        print(f"  [✓] LLM initialized (model: {self.model})")

    def generate(self, user_query: str, context: str, max_retries: int = 3) -> str:
        """
        Generate an answer using the LLM with recipe context.

        Parameters:
            user_query: The user's original question
            context: Formatted recipe context from the retriever
            max_retries: Number of retry attempts on failure

        Returns:
            The LLM's response as a string
        """
        # Build the prompt with context
        user_message = f"""Based on the following recipes from our database, answer the user's question.

RECIPES FROM DATABASE:
{context}

USER QUESTION: {user_query}

Remember: Only use information from the recipes above. Cite recipe names when referencing them."""

        # Try with primary model, then fallbacks
        models_to_try = [self.model] + FALLBACK_MODELS

        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_message},
                        ],
                        max_tokens=1024,
                        temperature=0.7,  # Slightly creative but mostly factual
                    )
                    return response.choices[0].message.content

                except Exception as e:
                    error_msg = str(e)

                    # Rate limit — wait and retry
                    if "rate" in error_msg.lower() or "429" in error_msg:
                        wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 sec
                        print(f"  Rate limited. Waiting {wait_time}s... (attempt {attempt+1})")
                        time.sleep(wait_time)
                        continue

                    # Model not available — try next model
                    if "not available" in error_msg.lower() or "404" in error_msg:
                        print(f"  Model {model} unavailable, trying next...")
                        break  # Break inner retry loop, try next model

                    # Other error — retry
                    print(f"  LLM error (attempt {attempt+1}): {error_msg[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(1)

        return "I'm sorry, I couldn't generate a response right now. The free LLM tier may be temporarily unavailable. Please try again in a moment."


def main():
    """Test the LLM with a sample prompt."""
    print("Testing LLM connection via OpenRouter...")

    llm = RecipeLLM()

    test_context = """
Recipe 1: Cashew Chicken Stir Fry
- Cook time: 40 minutes
- Calories: 299
- Ingredients: chicken breast, cashews, soy sauce, garlic, ginger, vegetables
- Steps: Stir fry chicken, add vegetables, toss with sauce and cashews

Recipe 2: Quick Mushroom Supreme
- Cook time: 10 minutes
- Calories: 129
- Ingredients: mushrooms, cream, garlic, butter, parsley
- Steps: Sauté mushrooms in butter, add cream and garlic, simmer
"""

    test_query = "Which of these recipes would be good for a quick weeknight dinner?"

    print(f"\n  Query: {test_query}")
    print(f"  Generating response...\n")

    response = llm.generate(test_query, test_context)
    print(f"  Response:\n  {response}")


if __name__ == "__main__":
    main()