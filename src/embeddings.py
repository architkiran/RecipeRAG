"""
embeddings.py — Generate embeddings for recipe texts.

Uses the sentence-transformers library to run all-MiniLM-L6-v2 locally.
No API calls, no cost, no rate limits.

Usage:
    Called by load_recipes.py during the data loading pipeline.
    Can also be run standalone for testing:
        python -m src.embeddings
"""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Configuration ──────────────────────────────────
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = 64  # Process 64 texts at a time (balances speed vs memory)


class RecipeEmbedder:
    """
    Wrapper around sentence-transformers for generating recipe embeddings.

    WHY A CLASS?
    Loading the model takes 2-3 seconds. If we loaded it fresh for every
    embedding call, it would be incredibly slow. By wrapping it in a class,
    we load once and reuse across all calls.

    WHY NOT AN API?
    - Free: No API costs, no rate limits
    - Private: Your recipe data never leaves your machine
    - Fast: No network latency
    - Reliable: No downtime, no API key management
    """

    def __init__(self, model_name: str = MODEL_NAME):
        """Load the embedding model into memory."""
        print(f"Loading embedding model: {model_name}")
        print("  (First run downloads ~80MB model — subsequent runs use cache)")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  [✓] Model loaded — dimension: {self.dimension}")

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text string.

        Returns a list of floats (not numpy array) because
        psycopg2/pgvector expects Python lists.
        """
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()

    def embed_batch(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        WHY BATCH?
        The model processes multiple texts simultaneously on the CPU,
        which is much faster than one-at-a-time. With batch_size=64:
        - 1,000 recipes ≈ 16 batches ≈ 2–5 minutes on CPU
        - Without batching: ~15–20 minutes

        normalize_embeddings=True ensures all vectors have unit length (magnitude 1).
        This means cosine similarity equals dot product, which is faster to compute.
        """
        print(f"Generating embeddings for {len(texts)} texts (batch_size={BATCH_SIZE})...")

        embeddings = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # Unit vectors → cosine sim = dot product
        )

        # Convert numpy arrays to Python lists for database insertion
        return [emb.tolist() for emb in embeddings]


def main():
    """Test the embedding model with sample queries."""
    embedder = RecipeEmbedder()

    # Test with sample recipe-related texts
    test_texts = [
        "quick chicken stir fry with vegetables",
        "easy weeknight chicken dinner",
        "chocolate lava cake dessert",
        "vegetarian pasta with mushrooms",
        "how to change a car tire",  # Intentionally unrelated
    ]

    print(f"\nGenerating test embeddings...")
    embeddings = embedder.embed_batch(test_texts, show_progress=False)

    # Compute pairwise cosine similarities
    # Since vectors are normalized, cosine similarity = dot product
    print(f"\nPairwise Cosine Similarities:")
    print(f"{'':>45}", end="")
    for i in range(len(test_texts)):
        print(f"  [{i}]", end="")
    print()

    for i, text_i in enumerate(test_texts):
        label = text_i[:42] + "..." if len(text_i) > 45 else text_i
        print(f"  [{i}] {label:<42}", end="")
        for j in range(len(test_texts)):
            sim = np.dot(embeddings[i], embeddings[j])
            print(f" {sim:.2f}", end="")
        print()

    print(f"\nWhat to notice:")
    print(f"  - [0] and [1] should be highly similar (both chicken dinners)")
    print(f"  - [2] should be less similar to [0] (cake vs stir fry)")
    print(f"  - [4] should be very low with everything (cars vs food)")
    print(f"  - Diagonal is always 1.00 (each text is identical to itself)")


if __name__ == "__main__":
    main()