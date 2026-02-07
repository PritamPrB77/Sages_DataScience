"""
Causal RAG System - Streamlit Application
A beautiful, research-grade interface for causal analysis of conversational data.
"""

import streamlit as st
import json
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="Causal RAG Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #4F46E5;
        --secondary-color: #7C3AED;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --error-color: #EF4444;
        --bg-dark: #1F2937;
        --bg-light: #F9FAFB;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(79, 70, 229, 0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    /* Card styling */
    .evidence-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.75rem 0;
        border-left: 4px solid;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .evidence-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    .evidence-card.agent {
        border-left-color: #3B82F6;
        background: linear-gradient(to right, #EFF6FF, white);
    }
    
    .evidence-card.customer {
        border-left-color: #10B981;
        background: linear-gradient(to right, #ECFDF5, white);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-agent {
        background: #DBEAFE;
        color: #1E40AF;
    }
    
    .badge-customer {
        background: #D1FAE5;
        color: #065F46;
    }
    
    .badge-transcript {
        background: #F3E8FF;
        color: #6B21A8;
    }
    
    .badge-score {
        background: #FEF3C7;
        color: #92400E;
    }
    
    /* Causal factor card */
    .factor-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid #E5E7EB;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
    }
    
    .factor-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    
    .factor-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1F2937;
    }
    
    .confidence-bar {
        height: 8px;
        background: #E5E7EB;
        border-radius: 4px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease-out;
    }
    
    .confidence-high { background: linear-gradient(90deg, #10B981, #34D399); }
    .confidence-medium { background: linear-gradient(90deg, #F59E0B, #FBBF24); }
    .confidence-low { background: linear-gradient(90deg, #EF4444, #F87171); }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin: 0.75rem 0;
        max-width: 85%;
    }
    
    .chat-user {
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 4px;
    }
    
    .chat-assistant {
        background: #F3F4F6;
        color: #1F2937;
        border-bottom-left-radius: 4px;
    }
    
    /* Stats card */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid #E5E7EB;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4F46E5;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: #6B7280;
        margin-top: 0.25rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1F2937 0%, #111827 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #E5E7EB;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.4);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #E5E7EB;
        transition: border-color 0.2s;
    }
    
    .stTextArea textarea:focus {
        border-color: #4F46E5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #1F2937;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Loading animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "current_analysis" not in st.session_state:
        st.session_state.current_analysis = None
    if "selected_outcome" not in st.session_state:
        st.session_state.selected_outcome = None
    if "selected_domain" not in st.session_state:
        st.session_state.selected_domain = None
    if "show_evidence" not in st.session_state:
        st.session_state.show_evidence = True
    if "available_outcomes" not in st.session_state:
        st.session_state.available_outcomes = []
    if "available_domains" not in st.session_state:
        st.session_state.available_domains = []
    if "stats" not in st.session_state:
        st.session_state.stats = None


def load_data(file_path: str, force_rebuild: bool = False):
    """Load data and initialize the orchestrator."""
    try:
        from langgraph_orchestrator import CausalRAGOrchestrator
        
        with st.spinner("🔄 Initializing Causal RAG System..."):
            progress = st.progress(0)
            status = st.empty()
            
            status.text("📂 Loading conversation data...")
            progress.progress(20)
            time.sleep(0.5)
            
            status.text("🧠 Creating embeddings (this may take a while)...")
            progress.progress(40)
            
            # Initialize orchestrator
            orchestrator = CausalRAGOrchestrator(
                data_path=file_path,
                force_rebuild=force_rebuild
            )
            
            progress.progress(80)
            status.text("✅ Finalizing setup...")
            
            # Store in session state
            st.session_state.orchestrator = orchestrator
            st.session_state.data_loaded = True
            st.session_state.available_outcomes = orchestrator.get_available_outcomes()
            st.session_state.available_domains = orchestrator.get_available_domains()
            st.session_state.stats = orchestrator.get_statistics()
            
            progress.progress(100)
            status.text("🎉 Ready!")
            time.sleep(0.5)
            
            progress.empty()
            status.empty()
            
        return True
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return False


def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🔍 CauseConverser</h1>
        <p>Evidence-grounded causal analysis of customer service conversations</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Data loading section
        st.markdown("### 📁 Data Source")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload JSON Dataset",
            type=["json"],
            help="Upload a conversational transcript dataset in JSON format"
        )
        
        # Or use default file
        use_default = st.checkbox(
            "Use default dataset",
            value=True,
            help="Use the Conversational_Transcript_Dataset.json file"
        )
        
        force_rebuild = st.checkbox(
            "Force rebuild index",
            value=False,
            help="Rebuild the vector index from scratch"
        )
        
        if st.button("🚀 Load Data", type="primary", use_container_width=True):
            if uploaded_file is not None:
                # Save uploaded file temporarily
                temp_path = "temp_upload.json"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                load_data(temp_path, force_rebuild)
            elif use_default:
                default_path = "Conversational_Transcript_Dataset.json"
                if os.path.exists(default_path):
                    load_data(default_path, force_rebuild)
                else:
                    st.error("Default dataset not found!")
            else:
                st.warning("Please upload a dataset or use the default one.")
        
        st.markdown("---")
        
        # Outcome and domain selection (only if data is loaded)
        if st.session_state.data_loaded:
            st.markdown("### 🎯 Analysis Settings")
            
            # Outcome selector
            outcomes = ["All Outcomes"] + st.session_state.available_outcomes
            selected = st.selectbox(
                "Select Outcome",
                outcomes,
                help="Select the outcome category to analyze (e.g., Escalation, Resolution)"
            )
            st.session_state.selected_outcome = None if selected == "All Outcomes" else selected
            
            # Domain filter
            domains = ["All Domains"] + st.session_state.available_domains
            selected_domain = st.selectbox(
                "Filter by Domain",
                domains,
                help="Optionally filter by business domain"
            )
            st.session_state.selected_domain = None if selected_domain == "All Domains" else selected_domain
            
            st.markdown("---")
            
            # Display options
            st.markdown("### 👁️ Display Options")
            st.session_state.show_evidence = st.toggle(
                "Show Retrieved Evidence",
                value=True,
                help="Toggle the evidence panel visibility"
            )
            
            st.markdown("---")
            
            # Reset button
            if st.button("🔄 Reset Context", use_container_width=True):
                if st.session_state.orchestrator:
                    st.session_state.orchestrator.reset_context()
                st.session_state.chat_history = []
                st.session_state.current_analysis = None
                st.success("Context reset!")
            
            # Export button
            if st.button("📥 Export Analysis", use_container_width=True):
                if st.session_state.orchestrator and st.session_state.current_analysis:
                    export_path = f"analysis_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    st.session_state.orchestrator.export_analysis(export_path)
                    st.success(f"Exported to {export_path}")
        
        st.markdown("---")
        
        # Stats display
        if st.session_state.stats:
            st.markdown("### 📊 Dataset Statistics")
            stats = st.session_state.stats["data"]
            st.metric("Transcripts", stats.get("total_transcripts", 0))
            st.metric("Total Turns", stats.get("total_turns", 0))
            st.metric("Avg Turns/Conv", stats.get("average_turns_per_conversation", 0))


def render_evidence_card(evidence: Dict[str, Any]):
    """Render a single evidence card."""
    speaker = evidence.get("speaker", "Unknown")
    is_agent = speaker == "Agent"
    
    speaker_badge = "badge-agent" if is_agent else "badge-customer"
    card_class = "agent" if is_agent else "customer"
    icon = "👤" if is_agent else "💬"
    
    relevance = evidence.get("relevance", evidence.get("score", 0))
    if isinstance(relevance, float):
        relevance = f"{relevance:.2f}"
    
    st.markdown(f"""
    <div class="evidence-card {card_class}">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div>
                <span class="badge {speaker_badge}">{icon} {speaker}</span>
                <span class="badge badge-transcript">📋 {evidence.get('transcript_id', 'N/A')[:15]}...</span>
            </div>
            <span class="badge badge-score">⭐ {relevance}</span>
        </div>
        <p style="margin: 0; color: #374151; line-height: 1.6;">{evidence.get('text', '')}</p>
    </div>
    """, unsafe_allow_html=True)


def render_causal_factor(factor: Dict[str, Any], index: int):
    """Render a causal factor card."""
    confidence = factor.get("confidence", 0.5)
    confidence_pct = int(confidence * 100)
    
    # Determine confidence level color
    if confidence >= 0.7:
        conf_class = "confidence-high"
        conf_label = "High"
    elif confidence >= 0.4:
        conf_class = "confidence-medium"
        conf_label = "Medium"
    else:
        conf_class = "confidence-low"
        conf_label = "Low"
    
    with st.container():
        st.markdown(f"""
        <div class="factor-card">
            <div class="factor-header">
                <span class="factor-name">🔹 {index}. {factor.get('name', 'Unknown Factor')}</span>
                <span style="color: #6B7280; font-size: 0.875rem;">{conf_label} Confidence</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill {conf_class}" style="width: {confidence_pct}%;"></div>
            </div>
            <p style="color: #4B5563; margin: 0.75rem 0 0 0; line-height: 1.6;">
                {factor.get('causal_explanation', factor.get('description', 'No explanation available.'))}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show evidence snippets in expander
        evidence_snippets = factor.get("evidence_snippets", [])
        if evidence_snippets:
            with st.expander(f"📌 Evidence ({len(evidence_snippets)} snippets)"):
                for snippet in evidence_snippets[:5]:
                    st.markdown(f"> *\"{snippet[:200]}...\"*" if len(snippet) > 200 else f"> *\"{snippet}\"*")


def render_chat_message(message: Dict[str, Any]):
    """Render a chat message."""
    role = message.get("role", "user")
    content = message.get("content", "")
    
    if role == "user":
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)


def process_query(query: str):
    """Process a user query."""
    if not st.session_state.orchestrator:
        st.error("Please load data first!")
        return None
    
    if not st.session_state.selected_outcome:
        st.warning("Please select an outcome to analyze in the sidebar.")
        return None
    
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": query
    })
    
    # Process query
    with st.spinner("🔍 Analyzing..."):
        result = st.session_state.orchestrator.process_query(
            query=query,
            outcome=st.session_state.selected_outcome,
            domain=st.session_state.selected_domain
        )
    
    # Add assistant response to history
    response = result.get("response", "I couldn't generate a response.")
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response
    })
    
    # Store current analysis
    st.session_state.current_analysis = result
    
    return result


def render_main_interface():
    """Render the main chat interface."""
    
    # Check if data is loaded
    if not st.session_state.data_loaded:
        st.info("👈 Please load a dataset using the sidebar to get started.")
        
        # Show example queries
        st.markdown("### 💡 Example Queries")
        st.markdown("""
        Once you load data and select an outcome, you can ask questions like:
        
        - **Task 1 (Causal Explanation):** "Why do calls escalate?"
        - **Task 2 (Follow-up):** "Could this have been prevented?"
        - **Agent Analysis:** "What agent behaviors contributed to this?"
        - **Comparison:** "Compare escalated vs resolved calls"
        - **Evidence:** "Show me the evidence for the first factor"
        """)
        return
    
    # Two-column layout
    if st.session_state.show_evidence:
        col1, col2 = st.columns([3, 2])
    else:
        col1 = st.container()
        col2 = None
    
    # Main chat interface
    with col1:
        st.markdown("### 💬 Analysis Chat")
        
        # Display current context
        if st.session_state.selected_outcome:
            st.markdown(f"**Analyzing:** `{st.session_state.selected_outcome}`" + 
                       (f" in `{st.session_state.selected_domain}`" if st.session_state.selected_domain else ""))
        
        # Chat history container
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                render_chat_message(msg)
        
        # Query input
        st.markdown("---")
        
        # Quick query buttons
        st.markdown("**Quick Queries:**")
        quick_cols = st.columns(4)
        
        quick_queries = [
            ("Why?", f"Why do calls result in {st.session_state.selected_outcome or 'this outcome'}?"),
            ("Prevent?", "Could this have been prevented?"),
            ("Agent?", "What agent behaviors contributed to this?"),
            ("Evidence?", "Show me the key evidence")
        ]
        
        for i, (label, query) in enumerate(quick_queries):
            with quick_cols[i]:
                if st.button(label, use_container_width=True, key=f"quick_{i}"):
                    process_query(query)
                    st.rerun()
        
        # Main input
        user_query = st.chat_input(
            "Ask a causal question...",
            key="main_input"
        )
        
        if user_query:
            process_query(user_query)
            st.rerun()
        
        # Display causal factors if available
        if st.session_state.current_analysis:
            factors = st.session_state.current_analysis.get("causal_factors", [])
            if factors:
                st.markdown("### 🔗 Identified Causal Factors")
                for i, factor in enumerate(factors, 1):
                    render_causal_factor(factor, i)
    
    # Evidence panel
    if col2 and st.session_state.show_evidence:
        with col2:
            st.markdown("### 📋 Retrieved Evidence")
            
            if st.session_state.current_analysis:
                evidence = st.session_state.current_analysis.get("evidence_display", [])
                
                if evidence:
                    st.markdown(f"*Showing {len(evidence)} most relevant turns*")
                    
                    for ev in evidence:
                        render_evidence_card(ev)
                else:
                    st.info("No evidence retrieved yet. Ask a question to see relevant dialogue turns.")
            else:
                st.info("Evidence will appear here after you ask a question.")
            
            # Toggle for JSON view
            if st.session_state.current_analysis:
                with st.expander("🔧 Raw JSON Output"):
                    # Remove large fields for display
                    display_analysis = {
                        k: v for k, v in st.session_state.current_analysis.items()
                        if k not in ["retrieved_results"]
                    }
                    st.json(display_analysis)


def render_statistics_dashboard():
    """Render a statistics dashboard."""
    if not st.session_state.stats:
        return
    
    st.markdown("### 📊 Data Overview")
    
    stats = st.session_state.stats
    data_stats = stats.get("data", {})
    
    # Stats cards
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{data_stats.get('total_transcripts', 0):,}</div>
            <div class="stat-label">Total Transcripts</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{data_stats.get('total_turns', 0):,}</div>
            <div class="stat-label">Total Turns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{data_stats.get('average_turns_per_conversation', 0)}</div>
            <div class="stat-label">Avg Turns/Conv</div>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{data_stats.get('unique_intents', 0)}</div>
            <div class="stat-label">Unique Intents</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Outcome distribution
    st.markdown("### 📈 Outcome Distribution")
    outcomes = data_stats.get("outcomes", {})
    if outcomes:
        import pandas as pd
        df = pd.DataFrame([
            {"Outcome": k, "Count": v}
            for k, v in sorted(outcomes.items(), key=lambda x: -x[1])
        ])
        st.bar_chart(df.set_index("Outcome"))


def main():
    """Main application entry point."""
    # Initialize session state
    init_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["💬 Analysis", "📊 Statistics", "ℹ️ About"])
    
    with tab1:
        render_main_interface()
    
    with tab2:
        if st.session_state.data_loaded:
            render_statistics_dashboard()
        else:
            st.info("Load data to view statistics.")
    
    with tab3:
        st.markdown("""
        ## About Causal RAG Analyzer
        
        This system is a **Causal, Evidence-Constrained, Context-Preserving RAG** implementation 
        using **LangGraph** for analyzing customer service conversations.
        
        ### Key Features
        
        - 🔍 **Causal Analysis**: Identifies WHY specific outcomes (escalations, complaints, etc.) occurred
        - 📋 **Evidence-Grounded**: All claims are traceable to specific dialogue turns
        - 🔄 **Multi-Turn Context**: Maintains context across follow-up questions
        - 🛡️ **No Hallucinations**: Strict evidence validation blocks unsupported claims
        
        ### Architecture
        
        ```
        JSON Data → Parser → Turn-Level Chunks → Embeddings → FAISS
                                                      ↓
        Query → LangGraph Pipeline → Causal Reasoning → Validated Output
                    ↓
             [Query Understanding] → [RAG Retrieval] → [Causal Analysis]
                                           ↓
                               [Evidence Validation] → [Context Memory]
        ```
        
        ### How to Use
        
        1. **Load Data**: Upload a JSON dataset or use the default one
        2. **Select Outcome**: Choose the outcome category to analyze (e.g., Escalation)
        3. **Ask Questions**: Use natural language to ask WHY the outcome occurred
        4. **Review Evidence**: Examine the supporting dialogue turns
        5. **Follow Up**: Ask additional questions while maintaining context
        
        ### Task Types
        
        - **Task 1 - Causal Explanation**: "Why do calls escalate?"
        - **Task 2 - Follow-up Reasoning**: "Could this have been prevented?"
        
        ### Technical Stack
        
        - **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
        - **Vector Store**: FAISS
        - **Orchestration**: LangGraph
        - **LLM**: Google Gemini 1.5 Flash
        - **UI**: Streamlit
        """)


if __name__ == "__main__":
    main()
