"""
clean_recipes.py — Phase 2 data cleaning pipeline.

Takes the raw Food.com dataset and produces a clean CSV
ready for embedding and insertion into PostgreSQL.

Usage:
    python -m data_processing.clean_recipes

Input:  data/RAW_recipes.csv
Output: data/recipes_clean.csv
"""

import ast
import re
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path

# ── Configuration ──────────────────────────────────
RAW_DATA_PATH = Path("data/RAW_recipes.csv")
CLEAN_DATA_PATH = Path("data/recipes_clean.csv")
SAMPLE_SIZE = 1000          # Start small. Increase later if you want.
MAX_COOK_MINUTES = 360      # Cap at 6 hours — anything beyond is likely data entry error
RANDOM_SEED = 42            # For reproducible sampling


# ═══════════════════════════════════════════════════
# STEP 1: Load and Sample
# ═══════════════════════════════════════════════════

def load_and_sample(path: Path, sample_size: int) -> pd.DataFrame:
    """
    Load the raw CSV and take a random sample.

    WHY SAMPLE?
    - The full dataset is 230K+ recipes. Embedding all of them takes hours.
    - For development and learning, 1,000 recipes is plenty.
    - Once everything works, you can bump SAMPLE_SIZE to 5000 or remove sampling entirely.
    """
    print(f"Loading raw data from {path}...")
    df = pd.read_csv(path)
    print(f"  Raw dataset: {len(df):,} recipes, {len(df.columns)} columns")

    # Drop columns we don't need for RAG
    # contributor_id, submitted, and description are not useful for search
    cols_to_drop = ["contributor_id", "submitted"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Take a random sample
    df = df.sample(n=min(sample_size, len(df)), random_state=RANDOM_SEED)
    print(f"  Sampled: {len(df):,} recipes")

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════
# STEP 2: Parse String-Encoded Lists
# ═══════════════════════════════════════════════════

def safe_parse_list(value) -> list:
    """
    Parse a string like "['a', 'b', 'c']" into an actual Python list.

    WHY IS THIS NEEDED?
    The CSV stores lists as literal strings. Pandas reads them as:
        "['preheat oven', 'mix flour']"  ← this is a STRING, not a list

    We use ast.literal_eval to safely convert it to:
        ['preheat oven', 'mix flour']    ← this is an actual list

    ast.literal_eval is safe (unlike eval()) — it only parses literals,
    not arbitrary code.
    """
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def parse_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Parse all string-encoded list columns into real Python lists."""
    print("Parsing string-encoded lists...")

    list_columns = ["tags", "steps", "ingredients", "nutrition"]
    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(safe_parse_list)
            # Show a sample so you can verify it worked
            sample_val = df[col].iloc[0]
            print(f"  {col}: parsed → {type(sample_val).__name__} (e.g., {str(sample_val)[:80]}...)")

    return df


# ═══════════════════════════════════════════════════
# STEP 3: Clean Text Fields
# ═══════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Clean a single text string:
    1. Strip HTML tags (e.g., <br>, <b>, &amp;)
    2. Normalize whitespace (collapse multiple spaces/newlines into one space)
    3. Strip leading/trailing whitespace

    WHY BeautifulSoup?
    Some recipe descriptions contain HTML from the original website.
    Regex-based HTML stripping is fragile — BeautifulSoup handles edge cases
    like nested tags, malformed HTML, and HTML entities (&amp; → &).
    """
    if pd.isna(text):
        return ""
    text = str(text)
    # Strip HTML tags and decode entities
    text = BeautifulSoup(text, "html.parser").get_text()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_text_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning to all relevant string columns."""
    print("Cleaning text fields...")

    # Clean the recipe name
    df["name"] = df["name"].apply(clean_text)

    # Clean description (many are NaN — that's fine, we fill with "")
    if "description" in df.columns:
        df["description"] = df["description"].apply(clean_text)

    # Clean each step individually (they're lists now)
    df["steps"] = df["steps"].apply(
        lambda steps: [clean_text(s) for s in steps]
    )

    # Clean each ingredient individually
    df["ingredients"] = df["ingredients"].apply(
        lambda ings: [clean_text(i) for i in ings]
    )

    print("  Text fields cleaned (HTML stripped, whitespace normalized)")
    return df


# ═══════════════════════════════════════════════════
# STEP 4: Fix Cook Times
# ═══════════════════════════════════════════════════

def fix_cook_times(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap absurd cook times and handle zeros.

    WHY?
    The raw data has entries like:
      - 0 minutes (missing data, not a real cook time)
      - 1,051,200 minutes (2 years — clearly a data entry error)

    For RAG, cook time is a key filter ("find me something under 30 min").
    Bad values break this filter. We:
      1. Replace 0 with NaN (unknown is better than wrong)
      2. Cap anything above 360 min (6 hours) at 360
    """
    print("Fixing cook times...")

    original_stats = df["minutes"].describe()

    # Replace 0 with NaN (we don't know the actual time)
    df.loc[df["minutes"] == 0, "minutes"] = pd.NA

    # Cap outliers
    outlier_count = (df["minutes"] > MAX_COOK_MINUTES).sum()
    df.loc[df["minutes"] > MAX_COOK_MINUTES, "minutes"] = MAX_COOK_MINUTES

    print(f"  Capped {outlier_count} outlier cook times to {MAX_COOK_MINUTES} min")
    print(f"  Range: {df['minutes'].min()} – {df['minutes'].max()} minutes")

    return df


# ═══════════════════════════════════════════════════
# STEP 5: Extract Nutrition Info
# ═══════════════════════════════════════════════════

def extract_nutrition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the nutrition list into individual columns.

    The raw nutrition column is a list of 7 values:
    [calories, total_fat_%DV, sugar_%DV, sodium_%DV, protein_%DV, sat_fat_%DV, carbs_%DV]

    %DV = "Percent Daily Value" — how much of your daily recommended intake
    one serving provides. We only extract calories (most useful for search)
    and protein_%DV, but keep the raw list in case you want others later.

    WHY EXTRACT CALORIES?
    Users frequently search by calorie count: "low calorie dinner",
    "high protein meals". Having calories as a numeric column lets us
    add WHERE clauses like: WHERE calories < 300
    """
    print("Extracting nutrition info...")

    def get_nutrition_value(nutrition_list, index, default=None):
        """Safely get a value from the nutrition list by index."""
        try:
            if isinstance(nutrition_list, list) and len(nutrition_list) > index:
                return round(float(nutrition_list[index]), 1)
        except (ValueError, TypeError):
            pass
        return default

    df["calories"] = df["nutrition"].apply(lambda x: get_nutrition_value(x, 0))
    df["protein_pdv"] = df["nutrition"].apply(lambda x: get_nutrition_value(x, 4))

    non_null = df["calories"].notna().sum()
    print(f"  Extracted calories for {non_null}/{len(df)} recipes")
    print(f"  Calorie range: {df['calories'].min()} – {df['calories'].max()}")

    return df


# ═══════════════════════════════════════════════════
# STEP 6: Extract Useful Tags
# ═══════════════════════════════════════════════════

def extract_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract structured info from the tags list.

    Tags in the raw data look like:
    ['60-minutes-or-less', 'time-to-make', 'cuisine', 'north-american',
     'main-dish', 'vegetables', 'vegetarian', 'dietary', ...]

    We extract:
    - cuisine: The first cuisine-type tag found
    - dietary tags: vegetarian, vegan, gluten-free, etc.
    - meal_type: breakfast, lunch, dinner, dessert, snack

    WHY?
    These become WHERE clause filters in Phase 4. When a user says
    "find me a vegetarian Italian dinner", we need structured columns
    to filter on — not just free text search.
    """
    print("Extracting tags...")

    # Define tag categories for extraction
    CUISINE_TAGS = {
        "italian", "mexican", "chinese", "indian", "japanese", "thai",
        "french", "greek", "mediterranean", "korean", "vietnamese",
        "spanish", "middle-eastern", "african", "caribbean",
        "north-american", "south-american", "european", "asian",
        "south-west-pacific", "australian", "canadian"
    }

    DIETARY_TAGS = {
        "vegetarian", "vegan", "gluten-free", "dairy-free", "low-carb",
        "low-fat", "low-sodium", "low-calorie", "sugar-free",
        "kosher", "halal", "paleo", "keto", "whole30"
    }

    MEAL_TAGS = {
        "breakfast", "brunch", "lunch", "dinner", "desserts", "dessert",
        "snacks", "appetizers", "side-dishes", "main-dish",
        "beverages", "drinks", "cocktails"
    }

    def extract_first_match(tags_list, valid_set):
        """Return the first tag that matches the valid set, or None."""
        for tag in tags_list:
            tag_clean = tag.lower().strip()
            if tag_clean in valid_set:
                return tag_clean
        return None

    def extract_all_matches(tags_list, valid_set):
        """Return all matching tags as a comma-separated string."""
        matches = [t.lower().strip() for t in tags_list if t.lower().strip() in valid_set]
        return ", ".join(matches) if matches else None

    df["cuisine"] = df["tags"].apply(lambda t: extract_first_match(t, CUISINE_TAGS))
    df["dietary"] = df["tags"].apply(lambda t: extract_all_matches(t, DIETARY_TAGS))
    df["meal_type"] = df["tags"].apply(lambda t: extract_first_match(t, MEAL_TAGS))

    # Convert tags list to a comma-separated string for storage
    df["tags_str"] = df["tags"].apply(lambda t: ", ".join(t) if isinstance(t, list) else "")

    print(f"  Cuisines found: {df['cuisine'].notna().sum()}/{len(df)}")
    print(f"  Dietary tags found: {df['dietary'].notna().sum()}/{len(df)}")
    print(f"  Meal types found: {df['meal_type'].notna().sum()}/{len(df)}")
    print(f"  Top cuisines: {df['cuisine'].value_counts().head(5).to_dict()}")

    return df


# ═══════════════════════════════════════════════════
# STEP 7: Build the recipe_text Field
# ═══════════════════════════════════════════════════

def build_recipe_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a single text field that the embedding model will encode.

    THIS IS THE MOST IMPORTANT STEP IN THE ENTIRE PIPELINE.

    WHY?
    The embedding model (all-MiniLM-L6-v2) reads this text and converts it
    into a 384-dimensional vector. When a user searches "quick chicken dinner",
    the model encodes that query into a vector too, and we find recipes whose
    vectors are closest.

    If recipe_text is just the recipe name ("Chicken Parmesan"), the embedding
    captures that but misses that it takes 25 minutes and uses mozzarella.

    If recipe_text includes name + ingredients + steps + metadata, the embedding
    captures the FULL meaning of the recipe, and searches become much more accurate.

    FORMAT DECISIONS:
    - We use natural language, not JSON or pipe-delimited text
    - We put the recipe name first (highest signal for matching)
    - Ingredients come next (second highest signal)
    - Steps are summarized (they're long but contain useful verbs like "grill", "bake")
    - Metadata (time, tags) provides context for filter-style queries

    EXAMPLE OUTPUT:
    "Chicken Parmesan. Ingredients: chicken breast, marinara sauce, mozzarella cheese,
    breadcrumbs, parmesan, olive oil. Steps: pound chicken, bread and fry, top with
    sauce and cheese, bake until bubbly. Cook time: 45 minutes. Tags: italian, dinner,
    main-dish. Dietary: none."
    """
    print("Building recipe_text field...")

    def build_text(row):
        parts = []

        # Recipe name (most important — goes first)
        name = row.get("name", "")
        if name:
            parts.append(f"{name}.")

        # Description (if available and not empty)
        desc = row.get("description", "")
        if desc and len(desc) > 10:
            # Truncate long descriptions to keep embedding focused
            parts.append(desc[:200])

        # Ingredients (critical for search — "chicken", "tofu", etc.)
        ingredients = row.get("ingredients", [])
        if isinstance(ingredients, list) and ingredients:
            parts.append(f"Ingredients: {', '.join(ingredients)}.")

        # Steps (summarize — full steps are very long)
        steps = row.get("steps", [])
        if isinstance(steps, list) and steps:
            # Take first 5 steps to keep it concise
            step_summary = "; ".join(steps[:5])
            if len(steps) > 5:
                step_summary += f" ... ({len(steps)} steps total)"
            parts.append(f"Steps: {step_summary}.")

        # Cook time
        cook_time = row.get("minutes")
        if pd.notna(cook_time):
            parts.append(f"Cook time: {int(cook_time)} minutes.")

        # Tags/category info
        cuisine = row.get("cuisine", "")
        meal = row.get("meal_type", "")
        dietary = row.get("dietary", "")

        if cuisine:
            parts.append(f"Cuisine: {cuisine}.")
        if meal:
            parts.append(f"Meal type: {meal}.")
        if dietary:
            parts.append(f"Dietary: {dietary}.")

        return " ".join(parts)

    df["recipe_text"] = df.apply(build_text, axis=1)

    # Show stats
    text_lengths = df["recipe_text"].str.len()
    print(f"  recipe_text length — min: {text_lengths.min()}, "
          f"avg: {text_lengths.mean():.0f}, max: {text_lengths.max()}")

    # Show one example
    print(f"\n  ── Example recipe_text ──")
    print(f"  {df['recipe_text'].iloc[0][:300]}...")

    return df


# ═══════════════════════════════════════════════════
# STEP 8: Final Cleanup and Export
# ═══════════════════════════════════════════════════

def final_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing critical fields and select final columns.

    WHY DROP ROWS?
    A recipe without a name or ingredients is useless for search.
    Better to have 950 clean recipes than 1,000 with 50 garbage rows.
    """
    print("Final cleanup...")

    initial_count = len(df)

    # Drop recipes with no name or no ingredients
    df = df[df["name"].str.len() > 0]
    df = df[df["ingredients"].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
    df = df[df["recipe_text"].str.len() > 50]  # Must have enough text for a meaningful embedding

    dropped = initial_count - len(df)
    print(f"  Dropped {dropped} incomplete recipes ({len(df)} remaining)")

    # Select and order the final columns
    final_columns = [
        "id",               # Unique recipe ID (for citations)
        "name",             # Recipe name (for display)
        "description",      # Short description
        "ingredients",      # List of ingredients (stored as string for CSV)
        "steps",            # List of steps (stored as string for CSV)
        "minutes",          # Cook time in minutes
        "n_steps",          # Number of steps
        "n_ingredients",    # Number of ingredients
        "calories",         # Calorie count
        "protein_pdv",      # Protein % daily value
        "cuisine",          # Extracted cuisine tag
        "dietary",          # Extracted dietary tags
        "meal_type",        # Extracted meal type
        "tags_str",         # All tags as comma-separated string
        "recipe_text",      # THE KEY FIELD — used for embedding
    ]

    # Only keep columns that exist (in case some were dropped earlier)
    final_columns = [c for c in final_columns if c in df.columns]
    df = df[final_columns]

    # Convert list columns to strings for CSV storage
    # (We'll parse them back when inserting into PostgreSQL)
    for col in ["ingredients", "steps"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: str(x) if isinstance(x, list) else x
            )

    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  RecipeRAG — Phase 2: Data Cleaning Pipeline")
    print("=" * 60)
    print()

    # Verify input file exists
    if not RAW_DATA_PATH.exists():
        print(f"ERROR: {RAW_DATA_PATH} not found!")
        print(f"Please download the Food.com dataset and place RAW_recipes.csv in the data/ folder.")
        print(f"Download from: https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions")
        return

    # Run the pipeline
    df = load_and_sample(RAW_DATA_PATH, SAMPLE_SIZE)
    print()
    df = parse_list_columns(df)
    print()
    df = clean_text_fields(df)
    print()
    df = fix_cook_times(df)
    print()
    df = extract_nutrition(df)
    print()
    df = extract_tags(df)
    print()
    df = build_recipe_text(df)
    print()
    df = final_cleanup(df)

    # Save to CSV
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)

    print()
    print("=" * 60)
    print(f"  DONE! Cleaned dataset saved to: {CLEAN_DATA_PATH}")
    print(f"  Final shape: {df.shape[0]} recipes × {df.shape[1]} columns")
    print(f"  File size: {CLEAN_DATA_PATH.stat().st_size / 1024:.1f} KB")
    print("=" * 60)


if __name__ == "__main__":
    main()