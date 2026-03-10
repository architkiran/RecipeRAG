"""
profile_data.py — Inspect the cleaned dataset to verify quality.

Usage:
    python -m data_processing.profile_data

Reads: data/recipes_clean.csv
"""

import pandas as pd
from pathlib import Path

CLEAN_DATA_PATH = Path("data/recipes_clean.csv")


def profile(df: pd.DataFrame):
    """Print a detailed profile of the cleaned dataset."""

    print("=" * 60)
    print("  RecipeRAG — Data Quality Profile")
    print("=" * 60)

    # ── Basic shape ──
    print(f"\n{'─' * 40}")
    print(f"SHAPE: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"{'─' * 40}")

    # ── Missing values ──
    print(f"\n{'─' * 40}")
    print("MISSING VALUES (per column):")
    print(f"{'─' * 40}")
    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = null_count / len(df) * 100
        status = "✓" if null_pct < 5 else "⚠" if null_pct < 20 else "✗"
        if null_count > 0:
            print(f"  {status} {col}: {null_count} missing ({null_pct:.1f}%)")
    all_complete = df.notna().all().all()
    if all_complete:
        print("  ✓ No missing values in any column!")

    # ── Cook time distribution ──
    if "minutes" in df.columns:
        print(f"\n{'─' * 40}")
        print("COOK TIME DISTRIBUTION:")
        print(f"{'─' * 40}")
        minutes = df["minutes"].dropna()
        print(f"  Count:  {len(minutes)}")
        print(f"  Min:    {minutes.min():.0f} min")
        print(f"  25th:   {minutes.quantile(0.25):.0f} min")
        print(f"  Median: {minutes.median():.0f} min")
        print(f"  75th:   {minutes.quantile(0.75):.0f} min")
        print(f"  Max:    {minutes.max():.0f} min")
        # Time buckets
        print(f"\n  Time buckets:")
        print(f"    Under 15 min:   {(minutes <= 15).sum()}")
        print(f"    15–30 min:      {((minutes > 15) & (minutes <= 30)).sum()}")
        print(f"    30–60 min:      {((minutes > 30) & (minutes <= 60)).sum()}")
        print(f"    1–2 hours:      {((minutes > 60) & (minutes <= 120)).sum()}")
        print(f"    2+ hours:       {(minutes > 120).sum()}")

    # ── Calories distribution ──
    if "calories" in df.columns:
        print(f"\n{'─' * 40}")
        print("CALORIE DISTRIBUTION:")
        print(f"{'─' * 40}")
        cals = df["calories"].dropna()
        print(f"  Min:    {cals.min():.0f}")
        print(f"  Median: {cals.median():.0f}")
        print(f"  Max:    {cals.max():.0f}")

    # ── Tag coverage ──
    print(f"\n{'─' * 40}")
    print("TAG COVERAGE:")
    print(f"{'─' * 40}")
    for col in ["cuisine", "dietary", "meal_type"]:
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = non_null / len(df) * 100
            print(f"  {col}: {non_null}/{len(df)} ({pct:.0f}%)")
            if col == "cuisine":
                print(f"    Top 5: {df[col].value_counts().head(5).to_dict()}")

    # ── recipe_text quality ──
    if "recipe_text" in df.columns:
        print(f"\n{'─' * 40}")
        print("RECIPE_TEXT FIELD (the embedding input):")
        print(f"{'─' * 40}")
        lengths = df["recipe_text"].str.len()
        word_counts = df["recipe_text"].str.split().str.len()
        print(f"  Character length — min: {lengths.min()}, avg: {lengths.mean():.0f}, max: {lengths.max()}")
        print(f"  Word count — min: {word_counts.min()}, avg: {word_counts.mean():.0f}, max: {word_counts.max()}")
        short = (lengths < 100).sum()
        if short > 0:
            print(f"  ⚠ {short} recipes have < 100 characters (may produce weak embeddings)")
        else:
            print(f"  ✓ All recipes have 100+ characters")

        # Show 3 example recipe_text entries
        print(f"\n  ── Sample recipe_text (first 3) ──")
        for i, text in enumerate(df["recipe_text"].head(3)):
            print(f"\n  [{i+1}] {text[:200]}...")

    print(f"\n{'═' * 60}")
    print("  Profile complete. Review the numbers above.")
    print(f"{'═' * 60}")


def main():
    if not CLEAN_DATA_PATH.exists():
        print(f"ERROR: {CLEAN_DATA_PATH} not found!")
        print("Run the cleaning pipeline first: python -m data_processing.clean_recipes")
        return

    df = pd.read_csv(CLEAN_DATA_PATH)
    profile(df)


if __name__ == "__main__":
    main()