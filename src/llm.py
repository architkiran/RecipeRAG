"""
llm.py — LLM integration via OpenRouter (with conversation memory).

Connects to OpenRouter's free tier using the OpenAI-compatible API.
Handles retries, rate limits, model fallback, and conversation history.

Usage:
    from src.llm import RecipeLLM
    llm = RecipeLLM()
    response = llm.generate("What about a vegan version?", context="...", history=[...])
"""

import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

PRIMARY_MODEL = os.getenv("LLM_MODEL", "openrouter/free")
FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-coder:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]

# How many previous messages to include as context
# Each message pair (user + assistant) uses ~500-1000 tokens
# Free models typically have 4K-8K token limits, so we keep it short
MAX_HISTORY_MESSAGES = 6  # 3 pairs of user/assistant

SYSTEM_PROMPT = """You are RecipeRAG, a helpful cooking assistant that answers questions about recipes.

RULES:
1. Base your answers ONLY on the recipes provided in the context below. Do not make up recipes.
2. If the context doesn't contain relevant recipes, say so honestly.
3. When referencing a recipe, always cite it by name.
4. Keep answers conversational, helpful, and concise (2-4 paragraphs max).
5. If asked about cook times, ingredients, or steps, be specific using the data provided.
6. Format recipe names in bold when mentioning them.
7. If the user asks a follow-up question (like "what about vegetarian?" or "anything faster?"),
   use the conversation history to understand what they're referring to.

When listing recipes, use this format:
- **Recipe Name** (cook time, calories) — brief description
"""


class RecipeLLM:
    """
    LLM wrapper for generating answers from recipe context.
    Now supports conversation history for follow-up questions.
    """

    def __init__(self):
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

    def generate(
        self,
        user_query: str,
        context: str,
        history: list[dict] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Generate an answer using the LLM with recipe context and conversation history.

        Parameters:
            user_query: The user's current question
            context: Formatted recipe context from the retriever
            history: List of previous messages [{"role": "user"/"assistant", "content": "..."}]
            max_retries: Number of retry attempts on failure

        Returns:
            The LLM's response as a string
        """
        # Build the current user message with recipe context
        current_message = f"""Based on the following recipes from our database, answer the user's question.

RECIPES FROM DATABASE:
{context}

USER QUESTION: {user_query}

Remember: Only use information from the recipes above. Cite recipe names when referencing them."""

        # Build the full message list
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # Add conversation history (trimmed to last N messages)
        if history:
            # Only include the last MAX_HISTORY_MESSAGES messages
            recent_history = history[-MAX_HISTORY_MESSAGES:]
            for msg in recent_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Add the current query
        messages.append({"role": "user", "content": current_message})

        # Try models with retry logic
        models_to_try = [self.model] + FALLBACK_MODELS

        for model in models_to_try:
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=1024,
                        temperature=0.7,
                    )
                    return response.choices[0].message.content

                except Exception as e:
                    error_msg = str(e)

                    if "rate" in error_msg.lower() or "429" in error_msg:
                        wait_time = 2 ** attempt
                        print(f"  Rate limited. Waiting {wait_time}s... (attempt {attempt+1})")
                        time.sleep(wait_time)
                        continue

                    if "not available" in error_msg.lower() or "404" in error_msg:
                        print(f"  Model {model} unavailable, trying next...")
                        break

                    print(f"  LLM error (attempt {attempt+1}): {error_msg[:100]}")
                    if attempt < max_retries - 1:
                        time.sleep(1)

        return ("I'm sorry, I couldn't generate a response right now. "
                "The free LLM tier may be temporarily unavailable. "
                "Please try again in a moment.")


def main():
    print("Testing LLM with conversation memory...")

    llm = RecipeLLM()

    context = """
Recipe 1: Cashew Chicken Stir Fry
- Cook time: 40 minutes, Calories: 299
- Ingredients: chicken breast, cashews, soy sauce, garlic, ginger, vegetables

Recipe 2: Vegetable Tofu Stir Fry
- Cook time: 25 minutes, Calories: 180
- Ingredients: tofu, bell peppers, broccoli, soy sauce, sesame oil
- Dietary: vegetarian, vegan
"""

    # Simulate a conversation
    history = []

    query1 = "What stir fry recipes do you have?"
    print(f"\n  You: {query1}")
    response1 = llm.generate(query1, context, history)
    print(f"  Bot: {response1}")

    # Add to history
    history.append({"role": "user", "content": query1})
    history.append({"role": "assistant", "content": response1})

    # Follow-up question
    query2 = "What about a vegetarian version?"
    print(f"\n  You: {query2}")
    response2 = llm.generate(query2, context, history)
    print(f"  Bot: {response2}")


if __name__ == "__main__":
    main()