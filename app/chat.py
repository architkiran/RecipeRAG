"""
chat.py — Streamlit chat UI for RecipeRAG.

A conversational interface for querying recipes using natural language.
Features:
- Chat-style message history
- Auto-detected filter badges (cuisine, dietary, time)
- Expandable recipe citations under each answer

Usage:
    streamlit run app/chat.py
"""

import streamlit as st
import sys
import os

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import RAGPipeline


# ═══════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════

st.set_page_config(
    page_title="RecipeRAG",
    page_icon="🍳",
    layout="centered",
)

# ── Custom CSS for a cleaner look ─────────────────
st.markdown("""
<style>
    .filter-badge {
        display: inline-block;
        padding: 2px 10px;
        margin: 2px 4px;
        border-radius: 12px;
        font-size: 0.8em;
        background-color: #f0f2f6;
        border: 1px solid #ddd;
    }
    .recipe-citation {
        padding: 8px 12px;
        margin: 4px 0;
        border-left: 3px solid #ff6b6b;
        background-color: #fafafa;
        border-radius: 0 4px 4px 0;
    }
    .similarity-bar {
        display: inline-block;
        height: 8px;
        background-color: #4CAF50;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# INITIALIZE SESSION STATE
# ═══════════════════════════════════════════════════

@st.cache_resource
def load_pipeline():
    """
    Load the RAG pipeline once and cache it.

    @st.cache_resource ensures the embedding model + DB connection
    are only initialized once, even if the user refreshes the page.
    Without this, every page refresh would reload the 80MB model.
    """
    return RAGPipeline()


# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []


# ═══════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════

st.title("🍳 RecipeRAG")
st.caption("Ask me anything about recipes — I'll search our database and find the best matches!")

# Help text in sidebar
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    **Try queries like:**
    - "Quick chicken dinner under 30 minutes"
    - "Vegetarian Italian pasta"
    - "Low calorie dessert ideas"
    - "What can I make with mushrooms and cream?"
    - "Healthy breakfast under 300 calories"

    **I auto-detect filters for:**
    - 🍽️ Cuisine (Italian, Mexican, Asian...)
    - 🥗 Dietary (vegetarian, vegan, gluten-free...)
    - ⏱️ Cook time ("quick", "under 30 min"...)
    - 🔥 Calories ("low calorie", "under 500 cal"...)
    """)

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "RecipeRAG uses semantic search over 1,000 recipes "
        "from Food.com, powered by pgvector + all-MiniLM-L6-v2 "
        "embeddings and a free LLM via OpenRouter."
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# ═══════════════════════════════════════════════════
# DISPLAY CHAT HISTORY
# ═══════════════════════════════════════════════════

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show filter badges for assistant messages
        if message["role"] == "assistant" and "filters" in message:
            filters = message["filters"]
            active = {k: v for k, v in filters.items()
                     if k != "search_query" and v is not None}
            if active:
                badges = " ".join(
                    f'<span class="filter-badge">{k}: {v}</span>'
                    for k, v in active.items()
                )
                st.markdown(f"**Filters applied:** {badges}", unsafe_allow_html=True)

        # Show recipe citations
        if message["role"] == "assistant" and "recipes" in message:
            recipes = message["recipes"]
            if recipes:
                with st.expander(f"📚 View {len(recipes)} source recipes"):
                    for r in recipes:
                        sim_pct = int(r["similarity"] * 100)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"**{r['name']}**")
                            details = []
                            if r.get("minutes"):
                                details.append(f"⏱️ {r['minutes']} min")
                            if r.get("calories"):
                                details.append(f"🔥 {r['calories']:.0f} cal")
                            if r.get("cuisine"):
                                details.append(f"🍽️ {r['cuisine']}")
                            if r.get("dietary"):
                                details.append(f"🥗 {r['dietary']}")
                            st.caption(" · ".join(details))
                        with col2:
                            st.caption(f"Match: {sim_pct}%")
                            st.progress(r["similarity"])


# ═══════════════════════════════════════════════════
# CHAT INPUT
# ═══════════════════════════════════════════════════

if prompt := st.chat_input("Ask about recipes..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching recipes and generating answer..."):
            pipeline = load_pipeline()
            result = pipeline.ask(prompt)

        # Display the answer
        st.markdown(result["answer"])

        # Show filter badges
        filters = result["filters"]
        active = {k: v for k, v in filters.items()
                 if k != "search_query" and v is not None}
        if active:
            badges = " ".join(
                f'<span class="filter-badge">{k}: {v}</span>'
                for k, v in active.items()
            )
            st.markdown(f"**Filters applied:** {badges}", unsafe_allow_html=True)

        # Show recipe citations
        recipes = result["recipes"]
        if recipes:
            with st.expander(f"📚 View {len(recipes)} source recipes"):
                for r in recipes:
                    sim_pct = int(r["similarity"] * 100)
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{r['name']}**")
                        details = []
                        if r.get("minutes"):
                            details.append(f"⏱️ {r['minutes']} min")
                        if r.get("calories"):
                            details.append(f"🔥 {r['calories']:.0f} cal")
                        if r.get("cuisine"):
                            details.append(f"🍽️ {r['cuisine']}")
                        if r.get("dietary"):
                            details.append(f"🥗 {r['dietary']}")
                        st.caption(" · ".join(details))
                    with col2:
                        st.caption(f"Match: {sim_pct}%")
                        st.progress(r["similarity"])

    # Save assistant message with metadata
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "filters": result["filters"],
        "recipes": result["recipes"],
    })