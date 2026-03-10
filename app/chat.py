"""
chat.py — Streamlit chat UI for RecipeRAG (with conversation memory).

Features:
- Chat-style message history
- Conversation memory (follow-up questions work)
- Auto-detected filter badges
- Expandable recipe citations
"""

import streamlit as st
import sys
import os

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
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
# INITIALIZE
# ═══════════════════════════════════════════════════

@st.cache_resource
def load_pipeline():
    return RAGPipeline()


if "messages" not in st.session_state:
    st.session_state.messages = []


# ═══════════════════════════════════════════════════
# HEADER & SIDEBAR
# ═══════════════════════════════════════════════════

st.title("🍳 RecipeRAG")
st.caption("Ask me anything about recipes — I'll search our database and find the best matches!")

with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    **Try queries like:**
    - "Quick chicken dinner under 30 minutes"
    - "Vegetarian Italian pasta"
    - "Low calorie dessert ideas"
    - "What can I make with mushrooms?"

    **Follow-up questions work!**
    - "What about a vegan version?"
    - "Anything faster?"
    - "Show me something with less calories"

    **Auto-detected filters:**
    - 🍽️ Cuisine · 🥗 Dietary · ⏱️ Cook time · 🔥 Calories
    """)

    st.divider()
    st.markdown("**About**")
    st.markdown(
        "RecipeRAG uses semantic search over 5,000 recipes "
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
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation history for the LLM
    # Only send the text content (not filters/recipes metadata)
    llm_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]  # Exclude the message we just added
    ]

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching recipes and generating answer..."):
            pipeline = load_pipeline()
            result = pipeline.ask(prompt, history=llm_history)

        st.markdown(result["answer"])

        filters = result["filters"]
        active = {k: v for k, v in filters.items()
                 if k != "search_query" and v is not None}
        if active:
            badges = " ".join(
                f'<span class="filter-badge">{k}: {v}</span>'
                for k, v in active.items()
            )
            st.markdown(f"**Filters applied:** {badges}", unsafe_allow_html=True)

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