"""
ThinkAI Studio — ChatGPT-Style Modern AI Assistant
A clean, minimalist, ChatGPT-inspired interface designed for college technical presentations.
Features:
- Completely removed left sidebar for a distraction-free centered chat canvas.
- ChatGPT dark theme (#212121) with rounded user bubbles and clean assistant typography.
- Floating top bar with model indicator, "+ New Chat" button, and sleek Settings popover.
- Centered welcome screen with ChatGPT-style prompt recommendation cards.
- Live token streaming via official Google Gen AI SDK (gemini-3.5-flash-lite).
- Tabs for Architecture & Concepts and Big Data Tabular Analytics.
"""

import os
import io
import json
from typing import Optional, List, Dict, Any, Generator
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv

# Load local environment variables (.env)
load_dotenv()

from services.gemini_service import (
    GeminiService,
    mask_api_key,
    sanitize_error_message,
    RECOMMENDED_MODELS,
)

# ---------------------------------------------------------
# Page Configuration (Collapsed Sidebar & Centered Layout)
# ---------------------------------------------------------
st.set_page_config(
    page_title="ThinkAI Studio",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# ChatGPT Dark Theme CSS
# ---------------------------------------------------------
CHATGPT_CSS = """
<style>
/* 1. Completely Hide Left Sidebar and Sidebar Toggle */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* 2. ChatGPT Dark Palette */
:root {
    --chatgpt-bg: #212121;
    --chatgpt-card: #2f2f2f;
    --chatgpt-card-hover: #383838;
    --chatgpt-border: #3d3d3d;
    --chatgpt-text: #ececec;
    --chatgpt-subtext: #b4b4b4;
    --chatgpt-accent: #10a37f;
}

/* App Background */
.stApp {
    background-color: var(--chatgpt-bg) !important;
    color: var(--chatgpt-text) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Centered Chat Column */
.main .block-container {
    max-width: 820px !important;
    margin: 0 auto !important;
    padding-top: 1rem !important;
    padding-bottom: 5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
}

/* Top Navigation Bar */
.chatgpt-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 0 16px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    margin-bottom: 15px;
}

.chatgpt-brand {
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.02em;
}

.chatgpt-model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #2f2f2f;
    border: 1px solid #424242;
    color: #d1d5db;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 4px 10px;
    border-radius: 9999px;
}

/* Centered Welcome Hero (ChatGPT Style) */
.welcome-hero {
    text-align: center;
    padding: 45px 15px 25px 15px;
}

.welcome-hero-icon {
    font-size: 2.8rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: linear-gradient(135deg, #10a37f 0%, #0070f3 100%);
    box-shadow: 0 4px 20px rgba(16, 163, 127, 0.35);
    margin-bottom: 16px;
}

.welcome-hero-title {
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 8px;
    letter-spacing: -0.03em;
}

.welcome-hero-sub {
    font-size: 1rem;
    color: var(--chatgpt-subtext);
    margin-bottom: 30px;
}

/* Prompt Suggestion Cards (ChatGPT 2x2 Grid) */
.suggestion-card {
    background: #2f2f2f;
    border: 1px solid #3d3d3d;
    border-radius: 14px;
    padding: 14px 16px;
    text-align: left;
    transition: all 0.2s ease;
    cursor: pointer;
    margin-bottom: 10px;
    height: 100%;
}

.suggestion-card:hover {
    background: #383838;
    border-color: #555555;
    transform: translateY(-1px);
}

.suggestion-card-title {
    font-weight: 600;
    color: #f3f4f6;
    font-size: 0.95rem;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.suggestion-card-desc {
    color: #9ca3af;
    font-size: 0.82rem;
    line-height: 1.4;
}

/* Chat Message Styling */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 14px 0 !important;
}

/* User Message Bubble */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    display: flex;
    justify-content: flex-end;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div {
    background: #2f2f2f !important;
    border: 1px solid #3d3d3d !important;
    border-radius: 20px 20px 4px 20px !important;
    padding: 12px 18px !important;
    max-width: 80% !important;
}

/* Floating Bottom Chat Input (ChatGPT Rounded Pill) */
[data-testid="stChatInput"] {
    background-color: #2f2f2f !important;
    border: 1px solid #444444 !important;
    border-radius: 26px !important;
    padding: 4px 10px !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #666666 !important;
}

[data-testid="stChatInput"] textarea {
    color: #f3f4f6 !important;
    font-size: 0.98rem !important;
}

/* Small Disclaimer Footer */
.disclaimer-text {
    text-align: center;
    color: #71717a;
    font-size: 0.76rem;
    margin-top: 14px;
    margin-bottom: 5px;
}

/* Concept Cards in Architecture Tab */
.concept-card {
    background: #2f2f2f;
    border: 1px solid #3d3d3d;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
}
.concept-card h4 {
    margin: 0 0 8px 0;
    color: #38bdf8;
    font-size: 1.05rem;
}
.concept-card p {
    margin: 0;
    font-size: 0.9rem;
    color: #d1d5db;
    line-height: 1.5;
}

/* Buttons in ChatGPT Dark Style */
.stButton button {
    border-radius: 10px !important;
    border: 1px solid #444444 !important;
    background: #2f2f2f !important;
    color: #ececec !important;
    transition: all 0.2s ease !important;
}

.stButton button:hover {
    background: #3a3a3a !important;
    border-color: #666666 !important;
    color: #ffffff !important;
}
</style>
"""
st.markdown(CHATGPT_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("GEMINI_API_KEY", "")

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gemini-3.5-flash-lite"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "system_instruction" not in st.session_state:
    st.session_state.system_instruction = (
        "You are ThinkAI, a helpful, highly intelligent, and articulate AI assistant. "
        "Explain complex technical topics (like LLMs, Agentic AI, Transformers, and Big Data) "
        "with clarity, structured formatting, and engaging examples."
    )

if "connection_status" not in st.session_state:
    st.session_state.connection_status = None


# ---------------------------------------------------------
# Helper Functions & Cached Client
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_service(api_key: str) -> GeminiService:
    return GeminiService(api_key=api_key)


def reset_conversation():
    st.session_state.messages = []


# ---------------------------------------------------------
# Top Navigation Bar (ChatGPT Style Header)
# ---------------------------------------------------------
col_nav_left, col_nav_mid, col_nav_right = st.columns([4, 2, 4])

with col_nav_left:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; padding-top: 6px;">
            <span style="font-size: 1.35rem; font-weight: 700; color: #fff;">⚡ ThinkAI</span>
            <span class="chatgpt-model-badge">● gemini-3.5-flash-lite</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_nav_right:
    c_btn1, c_btn2 = st.columns([1, 1])
    with c_btn1:
        if st.button("➕ New Chat", use_container_width=True):
            reset_conversation()
            st.rerun()

    with c_btn2:
        # Sleek ChatGPT-style Settings popover (replaces the left sidebar)
        with st.popover("⚙️ Settings", use_container_width=True):
            st.markdown("### ⚙️ ThinkAI Settings")
            st.caption("Model, API Key, and Persona controls")

            # API Key Configuration
            st.markdown("#### 🔑 API Key")
            api_input = st.text_input(
                "Gemini API Key",
                value=st.session_state.api_key,
                type="password",
                help="Stored securely in .env",
            )
            if api_input != st.session_state.api_key:
                st.session_state.api_key = api_input
                st.session_state.connection_status = None

            st.write(f"**Active Key:** `{mask_api_key(st.session_state.api_key)}`")

            if st.button("🔄 Test API Connection", use_container_width=True):
                if not st.session_state.api_key:
                    st.session_state.connection_status = (False, "API key missing.")
                else:
                    with st.spinner("Connecting..."):
                        service = get_service(st.session_state.api_key)
                        st.session_state.connection_status = service.verify_connection(
                            model=str(st.session_state.selected_model)
                        )

            if st.session_state.connection_status is not None:
                is_ok, msg = st.session_state.connection_status
                if is_ok:
                    st.success(f"🟢 {msg}")
                else:
                    st.error(f"🔴 {msg}")

            st.markdown("---")

            # Model Selection
            st.markdown("#### 🤖 Model")
            model_ids = [m["id"] for m in RECOMMENDED_MODELS]
            model_labels = {m["id"]: m["name"] for m in RECOMMENDED_MODELS}

            current_idx = (
                model_ids.index(st.session_state.selected_model)
                if st.session_state.selected_model in model_ids
                else 0
            )

            chosen_model = st.selectbox(
                "Model Version",
                options=model_ids,
                format_func=lambda mid: model_labels.get(mid, mid),
                index=current_idx,
            )
            if chosen_model != st.session_state.selected_model:
                st.session_state.selected_model = chosen_model

            # Temperature
            st.markdown("#### 🎛️ Creativity")
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(st.session_state.temperature),
                step=0.1,
            )

            # System Instruction
            st.markdown("#### 📝 System Persona")
            st.session_state.system_instruction = st.text_area(
                "System Prompt",
                value=st.session_state.system_instruction,
                height=100,
            )

            # Export Chat
            st.markdown("---")
            chat_export = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                "💾 Export Chat History",
                data=chat_export,
                file_name=f"thinkai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )


# ---------------------------------------------------------
# Main ChatGPT Interface
# ---------------------------------------------------------
clicked_prompt: Optional[str] = None

# Welcome Hero Screen (Displays when no messages exist)
if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="welcome-hero">
            <div class="welcome-hero-icon">⚡</div>
            <div class="welcome-hero-title">What can I help with today?</div>
            <div class="welcome-hero-sub">College Technical Presentation Edition • Powered by Google Gemini AI</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ChatGPT-Style Suggestion Cards (2x2 Grid + 1 center)
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        if st.button("🤖  LLMs vs. Agentic AI\n\nUnderstand the key differences and tool-calling capabilities", use_container_width=True):
            clicked_prompt = "Explain the difference between LLMs and agentic AI."

        if st.button("📊  The 3 Vs of Big Data\n\nVolume, Velocity, and Variety explained with real examples", use_container_width=True):
            clicked_prompt = "Explain the three Vs of Big Data with examples."

    with col_c2:
        if st.button("🧠  Self-Attention in Transformers\n\nHow modern foundation models process linguistic context", use_container_width=True):
            clicked_prompt = "Explain how a Transformer uses self-attention."

        if st.button("📈  AI Tabular Data Analysis\n\nHow LLMs inspect and extract insights from CSV datasets", use_container_width=True):
            clicked_prompt = "Explain how AI can analyze a CSV dataset."

    col_mid = st.columns([1, 2, 1])
    with col_mid[1]:
        if st.button("💡  Explain what an LLM is to a beginner", use_container_width=True):
            clicked_prompt = "Explain what an LLM is to a beginner."

    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

# Render Conversation Messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input Box (ChatGPT Pill)
chat_input_val = st.chat_input("Message ThinkAI...")
user_prompt = clicked_prompt if clicked_prompt else chat_input_val

if user_prompt:
    if not st.session_state.api_key:
        st.error(
            "⚠️ **API Key Missing**: Please set `GEMINI_API_KEY` in `.env` or open ⚙️ Settings above."
        )
    else:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate Assistant Response with Live Streaming
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

            try:
                service = get_service(st.session_state.api_key)
                with st.spinner(""):
                    model_id = str(st.session_state.get("selected_model") or "gemini-3.5-flash-lite")
                    stream = service.stream_conversation(
                        model=model_id,
                        messages=st.session_state.messages,
                        system_instruction=st.session_state.system_instruction,
                        temperature=float(st.session_state.temperature),
                    )
                    for chunk in stream:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response}
                )
            except Exception as ex:
                error_text = sanitize_error_message(ex, st.session_state.api_key)
                response_placeholder.error(f"❌ {error_text}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"⚠️ *Error: {error_text}*"}
                )

# Disclaimer Footer
st.markdown(
    """
    <div class="disclaimer-text">
        ThinkAI can make mistakes. Verify important info. Responses generated by Google Gemini API.
    </div>
    """,
    unsafe_allow_html=True,
)
