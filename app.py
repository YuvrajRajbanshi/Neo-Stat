"""
Smart AI Chatbot - Main Streamlit Application

A RAG-enabled chatbot with PDF document processing and web search fallback.
Supports both concise and detailed response modes.
"""

import streamlit as st

from config.config import validate_config, get_config_summary
from models.llm import get_response_with_context, get_response_without_context
from utils.rag import (
    process_pdf_for_rag,
    search_documents,
    format_context,
    has_relevant_context
)
from utils.web_search import search_web_and_format
from utils.helpers import (
    format_sources,
    sanitize_query,
    create_chat_message,
    calculate_context_relevance,
    generate_local_pdf_answer
)


# Page configuration
st.set_page_config(
    page_title="Smart AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


def inject_custom_styles():
    """Inject a handcrafted visual system for the Streamlit app UI."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

            :root {
                --bg: #f6f7f4;
                --paper: #ffffff;
                --ink: #18212f;
                --muted: #465364;
                --line: #d9dee7;
                --accent: #1f4b99;
                --accent-soft: #e9effa;
                --ok: #1f7a4d;
                --warn: #a16207;
                --chip: #eef2f8;
            }

            html, body, [class*="css"]  {
                font-family: "Manrope", sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at 20% 0%, #f0f4fb 0%, transparent 40%),
                    var(--bg);
                color: var(--ink);
            }

            .main .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
                max-width: 1120px;
            }

            section[data-testid="stSidebar"] {
                background: #f5f7fb;
                border-right: 1px solid var(--line);
            }

            section[data-testid="stSidebar"] .block-container {
                padding-top: 1.1rem;
            }

            section[data-testid="stSidebar"] * {
                color: #1f2a3a !important;
            }

            section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            section[data-testid="stSidebar"] label,
            section[data-testid="stSidebar"] span {
                color: #223044 !important;
            }

            section[data-testid="stSidebar"] .stRadio label {
                font-weight: 600;
            }

            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                background: #ffffff !important;
                border: 1px dashed #b7c4da !important;
                border-radius: 14px !important;
            }

            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
                color: #1b2a3d !important;
            }

            section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzoneInstructions"] {
                color: #33465f !important;
            }

            section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"] {
                background: #eef3fb !important;
                color: #16345f !important;
                border: 1px solid #b9c8e1 !important;
            }

            section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"]:hover {
                background: #e4ecf9 !important;
                color: #102c52 !important;
            }

            h1, h2, h3 {
                letter-spacing: -0.01em;
            }

            .hero-wrap {
                background: var(--paper);
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1.1rem 1.2rem;
                margin-bottom: 1rem;
                box-shadow: 0 10px 30px rgba(35, 45, 70, 0.06);
                animation: riseIn 420ms ease-out;
            }

            .hero-title {
                margin: 0;
                font-size: 1.55rem;
                font-weight: 800;
                color: var(--ink);
            }

            .hero-sub {
                margin: 0.35rem 0 0;
                color: var(--muted);
                font-size: 0.96rem;
                line-height: 1.45;
            }

            .chip-row {
                display: flex;
                gap: 0.5rem;
                margin-top: 0.9rem;
                flex-wrap: wrap;
            }

            .chip {
                background: var(--chip);
                border: 1px solid #dde3ee;
                color: #2f3e52;
                border-radius: 999px;
                font-size: 0.79rem;
                font-weight: 600;
                padding: 0.32rem 0.72rem;
            }

            .status-card {
                background: var(--paper);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 0.78rem 0.9rem;
                box-shadow: 0 6px 20px rgba(35, 45, 70, 0.04);
            }

            .status-label {
                color: var(--muted);
                font-size: 0.78rem;
                margin-bottom: 0.2rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                font-weight: 700;
            }

            .status-value {
                color: var(--ink);
                font-size: 1.02rem;
                font-weight: 700;
            }

            div[data-testid="stChatMessage"] {
                border-radius: 16px;
                border: 1px solid var(--line);
                background: var(--paper);
                padding: 0.2rem 0.1rem;
                color: #172433;
            }

            div[data-testid="stChatMessage"][data-testid*="user"] {
                background: #f0f4fb;
                border-color: #d8e2f3;
            }

            div[data-testid="stChatMessage"] p {
                line-height: 1.55;
                color: #172433;
            }

            div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
                color: #172433;
            }

            .stButton > button,
            .stDownloadButton > button {
                border-radius: 12px;
                border: 1px solid #cdd6e4;
                background: #f8faff;
                color: #1b3f7c;
                font-weight: 700;
                transition: all 160ms ease;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover {
                background: #eef4ff;
                border-color: #b8c8e3;
                transform: translateY(-1px);
            }

            .stTextInput > div > div > input,
            .stTextArea textarea {
                border-radius: 12px;
                border: 1px solid #ccd5e4;
            }

            .stChatInput textarea {
                border-radius: 14px !important;
                border: 1px solid #111111 !important;
                background: transparent !important;
                color: #ffffff !important;
            }

            .stChatInput textarea::placeholder {
                color: #d8dfeb !important;
                opacity: 1 !important;
            }

            [data-testid="stChatInput"] {
                background: transparent !important;
                border: 1px solid #111111 !important;
                border-radius: 14px !important;
                box-shadow: none !important;
            }

            [data-testid="stChatInput"] button {
                background: transparent !important;
                color: #111111 !important;
                border: 1px solid #111111 !important;
            }

            [data-testid="stChatInput"] button:hover {
                background: #f2f4f8 !important;
            }

            div[data-testid="stBottomBlockContainer"] {
                background: #f6f8fc !important;
                border-top: 1px solid #d8deea;
            }

            header[data-testid="stHeader"] {
                background: #f6f8fc;
            }

            @keyframes riseIn {
                from {
                    transform: translateY(8px);
                    opacity: 0;
                }
                to {
                    transform: translateY(0);
                    opacity: 1;
                }
            }

            @media (max-width: 768px) {
                .main .block-container {
                    padding-top: 0.9rem;
                }

                .hero-title {
                    font-size: 1.3rem;
                }

                .hero-sub {
                    font-size: 0.9rem;
                }
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def initialize_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0

    if "current_sources" not in st.session_state:
        st.session_state.current_sources = []

    if "source_type" not in st.session_state:
        st.session_state.source_type = None


def handle_uploaded_pdf(uploaded_file):
    """Process uploaded PDF and update session state safely."""
    if uploaded_file is None:
        return

    if st.session_state.pdf_processed and st.session_state.get("current_file") == uploaded_file.name:
        return

    with st.spinner("Processing PDF..."):
        try:
            vector_store, chunk_count = process_pdf_for_rag(uploaded_file)
            st.session_state.vector_store = vector_store
            st.session_state.pdf_processed = True
            st.session_state.chunk_count = chunk_count
            st.session_state.current_file = uploaded_file.name
            st.success(f"PDF processed! ({chunk_count} chunks created)")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.session_state.pdf_processed = False


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # Response mode toggle
        st.subheader("Response Mode")
        response_mode = st.radio(
            "Choose response style:",
            options=["Concise", "Detailed"],
            index=0,
            help="Concise: Short, direct answers. Detailed: In-depth explanations with examples."
        )

        st.divider()

        # PDF Upload section
        st.subheader("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help="Upload a PDF to enable RAG-based responses"
        )

        # Process PDF if uploaded
        handle_uploaded_pdf(uploaded_file)

        # Show PDF status
        if st.session_state.pdf_processed:
            st.info(f"📚 Active: {st.session_state.get('current_file', 'Unknown')}")
            st.caption(f"Chunks: {st.session_state.chunk_count}")

            if st.button("Clear Document"):
                st.session_state.vector_store = None
                st.session_state.pdf_processed = False
                st.session_state.chunk_count = 0
                st.session_state.current_file = None
                st.rerun()

        st.divider()

        # Configuration status
        st.subheader("🔧 Configuration")
        config_errors = validate_config()

        if config_errors:
            for error in config_errors:
                st.warning(error)
        else:
            st.success("API configured correctly")

        # Show config summary
        with st.expander("View Config"):
            config = get_config_summary()
            st.json(config)

        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.current_sources = []
            st.session_state.source_type = None
            st.rerun()

        return response_mode.lower()


def render_chat_history():
    """Render the chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message.get("sources") and message["role"] == "assistant":
                with st.expander("📚 View Sources"):
                    source_type = message.get("source_type", "document")
                    st.markdown(format_sources(message["sources"], source_type))


def render_insights_panel(response_mode: str):
    """Render right-side insights and app state panel."""
    with st.container(border=True):
        st.markdown("### PDF Upload")
        uploaded_file_main = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            key="main_pdf_uploader",
            help="Upload here if the sidebar is collapsed"
        )
        handle_uploaded_pdf(uploaded_file_main)

        if st.session_state.pdf_processed:
            st.caption(f"Active file: {st.session_state.get('current_file', 'Unknown')}")
            st.caption(f"Chunks indexed: {st.session_state.chunk_count}")

            if st.button("Clear Document", key="main_clear_document"):
                st.session_state.vector_store = None
                st.session_state.pdf_processed = False
                st.session_state.chunk_count = 0
                st.session_state.current_file = None
                st.rerun()

    with st.container(border=True):
        st.markdown("### Workspace")
        st.markdown(f"**Mode:** {response_mode.capitalize()}")
        st.markdown(f"**PDF:** {'Loaded' if st.session_state.pdf_processed else 'Not loaded'}")
        st.markdown(f"**Messages:** {len(st.session_state.messages)}")

        if st.session_state.pdf_processed:
            st.caption(f"Active file: {st.session_state.get('current_file', 'Unknown')}")
            st.caption(f"Chunks indexed: {st.session_state.chunk_count}")

    with st.container(border=True):
        st.markdown("### Latest Sources")
        latest_sources = st.session_state.messages[-1].get("sources", []) if st.session_state.messages else []

        if latest_sources:
            source_type = st.session_state.messages[-1].get("source_type", "document")
            st.markdown(format_sources(latest_sources[:3], source_type))
        else:
            st.caption("Sources from your latest assistant reply will appear here.")

    with st.container(border=True):
        st.markdown("### Tips")
        st.markdown("- Upload a PDF first for grounded answers.")
        st.markdown("- Ask specific questions for better retrieval.")
        st.markdown("- Switch to Detailed mode for deeper explanations.")


def process_query(query: str, response_mode: str) -> tuple[str, list, str]:
    """
    Process a user query through RAG pipeline with web search fallback.

    Args:
        query: User's question
        response_mode: "concise" or "detailed"

    Returns:
        Tuple of (response, sources, source_type)
    """
    sources = []
    source_type = None
    context = ""

    # Step 1: Try RAG if PDF is loaded
    if st.session_state.vector_store is not None:
        try:
            rag_results = search_documents(st.session_state.vector_store, query)

            if has_relevant_context(rag_results):
                context = format_context(rag_results)
                sources = rag_results
                source_type = "document"
        except Exception as e:
            st.warning(f"RAG search failed: {str(e)}")

    # Step 2: Fallback to web search if no relevant context
    if not context:
        try:
            web_context, web_results = search_web_and_format(query)

            if web_context:
                context = web_context
                sources = web_results
                source_type = "web"
        except Exception as e:
            st.warning(f"Web search failed: {str(e)}")

    # Step 3: Generate response
    if context:
        response = get_response_with_context(query, context, response_mode)

        # Graceful fallback when the LLM is unavailable.
        if response.startswith("Error:") and source_type == "document" and sources:
            response = generate_local_pdf_answer(query, sources, response_mode)
    else:
        response = get_response_without_context(query, response_mode)

    return response, sources, source_type


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session_state()

    # Inject visual styling
    inject_custom_styles()

    # Render sidebar and get settings
    response_mode = render_sidebar()

    # Main content area
    pdf_badge = "PDF Ready" if st.session_state.pdf_processed else "No PDF"
    st.markdown(
        f"""
        <div class="hero-wrap">
            <h1 class="hero-title">NeoStats Assistant</h1>
            <p class="hero-sub">
                Retrieval-first workspace for PDF intelligence with web fallback.
            </p>
            <div class="chip-row">
                <span class="chip">{pdf_badge}</span>
                <span class="chip">Mode: {response_mode.capitalize()}</span>
                <span class="chip">Messages: {len(st.session_state.messages)}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    chat_col, side_col = st.columns([2.2, 1], gap="large")

    with chat_col:
        st.markdown("### Conversation")
        render_chat_history()

    with side_col:
        render_insights_panel(response_mode)

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Sanitize the query
        query = sanitize_query(prompt)

        if not query:
            st.warning("Please enter a valid question.")
            return

        # Display user message
        with chat_col:
            with st.chat_message("user"):
                st.markdown(query)

        # Add user message to history
        st.session_state.messages.append(
            create_chat_message("user", query)
        )

        # Generate response
        with chat_col:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response, sources, source_type = process_query(query, response_mode)

                        # Display the response
                        st.markdown(response)

                        # Show sources if available
                        if sources:
                            relevance = calculate_context_relevance(sources) if source_type == "document" else ""
                            source_label = f"📚 Sources ({source_type})"
                            if relevance:
                                source_label += f" - {relevance}"

                            with st.expander(source_label):
                                st.markdown(format_sources(sources, source_type))

                        # Add assistant message to history
                        assistant_message = create_chat_message("assistant", response, sources)
                        assistant_message["source_type"] = source_type
                        st.session_state.messages.append(assistant_message)

                    except Exception as e:
                        error_msg = f"An error occurred: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            create_chat_message("assistant", error_msg)
                        )


if __name__ == "__main__":
    main()
