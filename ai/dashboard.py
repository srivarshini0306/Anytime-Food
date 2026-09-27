import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import snowflake.connector
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# ============================================================
# PAGE CONFIGURATION & METADATA
# ============================================================
st.set_page_config(
    page_title="Zomato AI Analytics Hub",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PATHS & ENVIRONMENT SETUP
# ============================================================
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

# Robust .env search
env_path = CURRENT_DIR / ".env"
if not env_path.exists():
    env_path = ROOT_DIR / ".env"
load_dotenv(dotenv_path=env_path)

CACHE_FILE = CURRENT_DIR / "review_embeddings.parquet"

# ============================================================
# CUSTOM CSS / THEME INJECTION (Zomato Crimson & Sleek Dark Glass)
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --zomato-red: #E23744;
        --zomato-crimson: #CB202D;
        --zomato-coral: #FF5A60;
        --bg-surface: rgba(22, 27, 34, 0.75);
        --bg-card: rgba(30, 36, 46, 0.65);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-hover: rgba(226, 55, 68, 0.4);
        --text-primary: var(--text-color, #1e293b);
        --text-secondary: #64748B;
        --accent-glow: rgba(226, 55, 68, 0.25);
    }

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }

    /* Hero Header Card */
    .hero-banner {
        background: linear-gradient(135deg, rgba(226, 55, 68, 0.15) 0%, rgba(20, 24, 33, 0.85) 60%, rgba(255, 90, 96, 0.08) 100%);
        border: 1px solid rgba(226, 55, 68, 0.3);
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px var(--accent-glow);
        position: relative;
        overflow: hidden;
    }

    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 250px;
        height: 250px;
        background: radial-gradient(circle, rgba(226, 55, 68, 0.2) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-title {
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-subtitle {
        font-size: 14.5px;
        color: var(--text-secondary);
        font-weight: 400;
        margin-bottom: 14px;
    }

    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid var(--border-subtle);
        background: rgba(0, 0, 0, 0.3);
        color: #E2E8F0;
    }

    .status-dot-green {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }

    .status-dot-red {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #E23744;
        box-shadow: 0 0 8px #E23744;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(128, 128, 128, 0.08);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.18);
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        border-radius: 8px;
        color: var(--text-color, #475569);
        font-size: 14px;
        font-weight: 600;
        padding: 0 18px;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #E23744;
        background-color: rgba(226, 55, 68, 0.06);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #E23744 0%, #CB202D 100%) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(226, 55, 68, 0.35) !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* Section & Cards */
    .content-card {
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.05));
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 18px;
    }

    .metric-badge-chip {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .chip-red {
        background: rgba(226, 55, 68, 0.15);
        color: #FF5A60;
        border: 1px solid rgba(226, 55, 68, 0.3);
    }

    .chip-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .chip-blue {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Review Evidence Card */
    .review-item-card {
        background: var(--secondary-background-color, rgba(128, 128, 128, 0.05));
        border-left: 4px solid #E23744;
        border-top: 1px solid rgba(128, 128, 128, 0.2);
        border-right: 1px solid rgba(128, 128, 128, 0.2);
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 0 12px 12px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }

    .review-item-card:hover {
        transform: translateY(-2px);
        border-color: #E23744;
        box-shadow: 0 4px 12px rgba(226, 55, 68, 0.15);
    }

    .review-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .review-city-badge {
        font-size: 12px;
        font-weight: 600;
        color: var(--text-color, #1e293b);
        background: rgba(128, 128, 128, 0.15);
        padding: 2px 8px;
        border-radius: 6px;
    }

    .review-stars {
        color: #F59E0B;
        font-weight: 700;
        font-size: 13px;
        letter-spacing: 1px;
    }

    .review-score {
        font-size: 12px;
        color: #10B981;
        font-weight: 700;
    }

    .review-text {
        color: var(--text-color, #1e293b);
        font-size: 13.5px;
        line-height: 1.55;
    }

    /* Code Container */
    pre, code {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        border-color: #E23744;
        box-shadow: 0 4px 12px rgba(226, 55, 68, 0.2);
    }

    /* Prompt Suggestion Buttons */
    div[data-testid="column"] .stButton button {
        text-align: left;
        width: 100%;
        font-size: 13px;
        border-radius: 10px;
        background: rgba(30, 36, 48, 0.5);
        border: 1px solid var(--border-subtle);
        padding: 8px 12px;
    }

    div[data-testid="column"] .stButton button:hover {
        background: rgba(226, 55, 68, 0.12);
        border-color: rgba(226, 55, 68, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_CHAT_MODEL = "openai/gpt-oss-120b"
DEFAULT_SQL_MODEL = "openai/gpt-oss-120b"
DEFAULT_SAMPLE_SIZE = 500
DEFAULT_TOP_K = 5

FORBIDDEN_WORDS = [
    "drop", "delete", "truncate", "alter",
    "update", "insert", "create", "replace",
    "grant", "revoke",
]

EXAMPLE_SQL_QUESTIONS = [
    "Top 10 cities by GMV",
    "Which cuisine has the most orders?",
    "Average delivery time by city, worst first",
    "Cancel rate by payment method",
    "Top 10 restaurants by total revenue",
    "Delivery SLA: P50 delivery minutes & late rate by hour",
]

EXAMPLE_RAG_QUESTIONS = [
    "What are the most common complaints about delivery delays and cold food?",
    "Which cuisines or dishes do customers rave about the most?",
    "What are frequent customer complaints regarding packaging and leakage?",
    "How do customers perceive pricing and portion sizes?",
    "What do 5-star reviews specifically praise about customer service?",
]

SNOWFLAKE_SCHEMA_DOC = """
Tables available (Snowflake). Use bare table names, no database or schema prefix.

FCT_ORDERS(
    order_id,
    order_date,
    customer_id,
    restaurant_id,
    city,
    cuisine,
    payment_method,
    order_status,
    is_delivered,
    sales_amount,
    discount,
    delivery_fee,
    gst,
    customer_rating,
    delivery_time_min
)

DIM_RESTAURANTS(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    rating,
    cost_for_two
)

DIM_CUSTOMER(
    customer_id,
    customer_name,
    age,
    age_segment,
    gender,
    city
)

MART_DAILY_CITY_REVENUNE(
    order_date,
    city,
    orders,
    cancel_rate,
    gmv,
    aov
)

MART_RESTUARANT_PERFORMANCE(
    restaurant_id,
    restaurant_name,
    city,
    cuisine,
    orders,
    revenue,
    avg_customer_rating,
    cancel_rate
)

MART_DELIVERY_SLA(
    city,
    order_hour,
    delivered_orders,
    p50_delivery_min,
    late_rate
)

Note:
gmv means delivered revenue.
Prefer the MART_ tables when they fit the question.
"""

SQL_SYSTEM_PROMPT = f"""
You are a Snowflake SQL expert.

Write ONE SELECT query that answers the question.

Rules:
- SELECT queries only, never modify data.
- Use bare table names.
- Add a LIMIT of 100 or less, unless the question asks for a single total.
- Reply as JSON in this exact format:
  {{"sql": "your query here"}}

{SNOWFLAKE_SCHEMA_DOC}
"""

# ============================================================
# INITIALIZE STATE
# ============================================================
if "rag_question" not in st.session_state:
    st.session_state.rag_question = ""
if "sql_question" not in st.session_state:
    st.session_state.sql_question = ""
if "last_rag_answer" not in st.session_state:
    st.session_state.last_rag_answer = None
if "last_rag_reviews" not in st.session_state:
    st.session_state.last_rag_reviews = None
if "last_sql_query" not in st.session_state:
    st.session_state.last_sql_query = None
if "last_sql_df" not in st.session_state:
    st.session_state.last_sql_df = None
if "last_sql_exec_time" not in st.session_state:
    st.session_state.last_sql_exec_time = None

# ============================================================
# CLIENT & CREDENTIAL INITIALIZATION
# ============================================================
groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

def get_snowflake_env():
    return {
        "account": (os.getenv("SNOWFLAKE_ACCOUNT") or "").strip(),
        "user": (os.getenv("SNOWFLAKE_USER") or "").strip(),
        "password": (os.getenv("SNOWFLAKE_PASSWORD") or "").strip(),
        "warehouse": (os.getenv("SNOWFLAKE_WAREHOUSE") or "").strip(),
        "database": (os.getenv("SNOWFLAKE_DATABASE") or "").strip(),
        "schema": (os.getenv("SNOWFLAKE_SCHEMA") or "AI").strip(),
    }

# ============================================================
# CACHED RESOURCE: EMBEDDING MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL):
    return SentenceTransformer(model_name)

# ============================================================
# SNOWFLAKE CONNECTIONS
# ============================================================
def get_snowflake_conn(schema: str = "MARTS", role: str = "DBT_ROLE"):
    env = get_snowflake_env()
    conn_params = {
        "account": env["account"],
        "user": env["user"],
        "password": env["password"],
        "warehouse": env["warehouse"],
        "database": env["database"],
        "schema": schema,
    }
    if role:
        conn_params["role"] = role
    return snowflake.connector.connect(**conn_params)

# ============================================================
# RAG ENGINE FUNCTIONS
# ============================================================
def fetch_reviews_from_snowflake(sample_size: int = DEFAULT_SAMPLE_SIZE):
    env = get_snowflake_env()
    conn = snowflake.connector.connect(
        account=env["account"],
        user=env["user"],
        password=env["password"],
        warehouse=env["warehouse"],
        database=env["database"],
        schema=env["schema"],
    )
    query = f"""
        SELECT
            REVIEW_ID,
            CITY,
            RATING,
            COMMENT
        FROM ZOMATO.STAGING.STG_REVIEWS
        SAMPLE ({sample_size} ROWS)
    """
    df = conn.cursor().execute(query).fetch_pandas_all()
    conn.close()
    df.columns = [col.lower() for col in df.columns]
    return df

def generate_embeddings(texts, model):
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()

@st.cache_data(show_spinner=False)
def load_rag_reviews(sample_size: int = DEFAULT_SAMPLE_SIZE, force_reload: bool = False):
    cache_path = str(CACHE_FILE)
    if not force_reload and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            if "embedding" in df.columns and "comment" in df.columns:
                return df, "Loaded from local parquet cache"
        except Exception:
            pass  # Fall through to fetching

    # Read from Snowflake
    try:
        df = fetch_reviews_from_snowflake(sample_size)
    except Exception as e:
        # Fallback sample dataset in case Snowflake is unreachable
        st.warning(f"Note: Could not query Snowflake directly ({e}). Using simulated reviews.")
        df = pd.DataFrame([
            {"review_id": "REV_001", "city": "Bangalore", "rating": 1, "comment": "Order took 95 minutes to arrive! The biryani was ice cold and curry spilled everywhere inside the bag."},
            {"review_id": "REV_002", "city": "Mumbai", "rating": 5, "comment": "Crispy masala dosa arrived piping hot within 25 minutes! Extremely satisfied with the fast rider delivery."},
            {"review_id": "REV_003", "city": "Delhi", "rating": 2, "comment": "Delivery partner was polite, but pizza was completely crushed and cheese stuck to the top carton box. Terrible packaging."},
            {"review_id": "REV_004", "city": "Bangalore", "rating": 5, "comment": "Exceptional customer support! When an item was missed, support refunded it in 2 minutes on chat."},
            {"review_id": "REV_005", "city": "Hyderabad", "rating": 4, "comment": "Authentic Haleem and biryani. Portion sizes are generous, though packaging containers could be sturdier."},
            {"review_id": "REV_006", "city": "Kolkata", "rating": 1, "comment": "Horrible delay. The delivery status showed 10 mins away for 45 minutes straight. Driver refused to answer call."},
            {"review_id": "REV_007", "city": "Pune", "rating": 5, "comment": "Awesome burgers and crunchy fries! Packaging kept everything warm and fresh. 10/10 service."},
            {"review_id": "REV_008", "city": "Mumbai", "rating": 3, "comment": "Food was delicious, but Rs 60 delivery fee + taxes made an Rs 250 roll cost over Rs 400. Prices on app are inflated."},
        ])

    model = load_embedding_model()
    df["embedding"] = generate_embeddings(df["comment"].fillna("").tolist(), model)
    try:
        df.to_parquet(cache_path)
        status_msg = f"Generated embeddings for {len(df)} reviews and saved to cache"
    except Exception as err:
        status_msg = f"Generated embeddings for {len(df)} reviews (cache write failed: {err})"

    return df, status_msg

def cosine_similarity(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

def find_similar_reviews(question: str, df: pd.DataFrame, top_k: int = DEFAULT_TOP_K):
    model = load_embedding_model()
    question_vector = generate_embeddings([question], model)[0]
    scores = [cosine_similarity(question_vector, review_vec) for review_vec in df["embedding"]]
    df_copy = df.copy()
    df_copy["similarity_score"] = scores
    return df_copy.nlargest(top_k, "similarity_score")

def ask_rag_llm(question: str, top_reviews: pd.DataFrame, model_name: str, temperature: float = 0.2):
    if not groq_client:
        return "⚠️ Error: GROQ_API_KEY is not configured in `.env`. Please provide a valid Groq API key."

    context = ""
    for _, row in top_reviews.iterrows():
        context += (
            f"City: {row.get('city', 'Unknown')}\n"
            f"Rating: {row.get('rating', 'N/A')} stars\n"
            f"Review: {row.get('comment', '')}\n\n"
        )

    system_prompt = """
You are a senior customer intelligence analyst for Zomato food delivery.
Answer the user's question clearly, concisely, and accurately based ONLY on the customer reviews provided in the context.

Formatting Guidelines:
- Highlight key trends or patterns in bold bullet points.
- Cite specific cities or rating patterns when relevant.
- Do not hallucinate or extrapolate beyond the provided customer quotes.
- If the reviews don't contain enough evidence, explicitly state that the available reviews do not provide sufficient information.
"""

    user_prompt = f"""Question:
{question}

Customer Reviews Context:
{context}
"""

    response = groq_client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content

# ============================================================
# TEXT-TO-SQL FUNCTIONS
# ============================================================
def generate_sql(question: str, model_name: str = DEFAULT_SQL_MODEL):
    if not groq_client:
        raise ValueError("GROQ_API_KEY is missing in your .env configuration.")

    response = groq_client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SQL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )

    answer = response.choices[0].message.content
    sql = json.loads(answer)["sql"]
    sql = sql.replace("ZOMATO.MARTS.", "").replace("ZOMATO.", "")
    return sql.strip().rstrip(";")

def is_sql_safe(sql: str) -> tuple[bool, str]:
    lowered = sql.lower().strip()
    if not lowered.startswith("select") and not lowered.startswith("with"):
        return False, "Query must strictly begin with 'SELECT' or 'WITH' (CTE)."
    for word in FORBIDDEN_WORDS:
        # Check whole word / substring boundary
        if f" {word} " in f" {lowered} " or f"\n{word} " in f"\n{lowered} ":
            return False, f"Forbidden keyword detected: '{word.upper()}'. DDL and DML operations are blocked."
    return True, "Query passed read-only safety guardrails."

def execute_snowflake_query(sql: str):
    start_time = time.time()
    conn = get_snowflake_conn(schema="MARTS", role="DBT_ROLE")
    cursor = conn.cursor()
    cursor.execute(sql)
    df = cursor.fetch_pandas_all()
    cursor.close()
    conn.close()
    elapsed = time.time() - start_time
    return df, elapsed

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            <div style="background:#E23744; width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:22px; box-shadow:0 4px 12px rgba(226,55,68,0.4);">
                🍕
            </div>
            <div>
                <div style="font-size:18px; font-weight:800; color:var(--text-color, #1e293b); letter-spacing:-0.3px;">ZOMATO AI</div>
                <div style="font-size:11px; font-weight:600; color:#64748B; text-transform:uppercase; letter-spacing:0.8px;">Intelligence Studio</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Engine Parameters")

    selected_llm = st.selectbox(
        "Groq Reasoning Model",
        options=["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
        index=0,
        help="Select the LLM model to power both Text-to-SQL translation and RAG synthesis.",
    )

    rag_top_k = st.slider(
        "RAG Evidence Depth (Top-K)",
        min_value=1,
        max_value=12,
        value=DEFAULT_TOP_K,
        help="Number of most semantically relevant reviews to retrieve for grounding.",
    )

    sample_size = st.select_slider(
        "Snowflake Reviews Sample Size",
        options=[100, 250, 500, 1000],
        value=DEFAULT_SAMPLE_SIZE,
        help="Number of reviews sampled from Snowflake staging table.",
    )

    llm_temp = st.slider(
        "RAG Creativity (Temperature)",
        min_value=0.0,
        max_value=0.8,
        value=0.2,
        step=0.05,
        help="Lower values yield more factual and grounded summaries.",
    )

    st.markdown("---")
    st.markdown("#### 🔌 Connection Status")

    env_creds = get_snowflake_env()
    has_sf = bool(env_creds["account"] and env_creds["user"])
    has_groq = bool(groq_api_key)

    sf_color = "#10B981" if has_sf else "#E23744"
    groq_color = "#10B981" if has_groq else "#E23744"
    sf_text = f"Connected ({env_creds['database'] or 'ZOMATO'})" if has_sf else "Missing Credentials"
    groq_text = "API Key Active" if has_groq else "Missing GROQ_API_KEY"

    st.markdown(
        f"""
        <div style="background:var(--secondary-background-color, rgba(128,128,128,0.08)); border:1px solid rgba(128,128,128,0.2); border-radius:10px; padding:12px; margin-bottom:14px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:12px; color:var(--text-color, #64748B);">❄️ Snowflake Warehouse</span>
                <span style="font-size:11px; font-weight:700; color:{sf_color};">{sf_text}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:12px; color:var(--text-color, #64748B);">⚡ Groq Inference</span>
                <span style="font-size:11px; font-weight:700; color:{groq_color};">{groq_text}</span>
            </div>
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:12px; color:var(--text-color, #64748B);">🧬 Vector Embeddings</span>
                <span style="font-size:11px; font-weight:700; color:#10B981;">{DEFAULT_EMBEDDING_MODEL}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🔄 Rebuild Embeddings Cache", use_container_width=True):
        st.cache_data.clear()
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        st.success("Cache cleared! New reviews will be fetched on next run.")
        st.rerun()

# ============================================================
# HERO HEADER BANNER
# ============================================================
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">
            <span>🍔 Zomato AI Analytics Suite</span>
        </div>
        <div class="hero-subtitle">
            Enterprise Generative Intelligence Hub: Natural Language Data Warehousing & Semantic Review Intelligence
        </div>
        <div class="badge-container">
            <span class="status-badge"><span class="status-dot-green"></span> Snowflake MARTS Live</span>
            <span class="status-badge">⚡ Model: openai/gpt-oss-120b</span>
            <span class="status-badge">🔍 Semantic Search: all-MiniLM-L6-v2</span>
            <span class="status-badge">🛡️ Read-Only Guardrails Active</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DASHBOARD TABS
# ============================================================
tab_rag, tab_sql, tab_schema = st.tabs([
    "💬 Customer Review Intelligence (RAG)",
    "📊 Text-to-SQL Analytics",
    "📚 Snowflake Data Catalog & Schema",
])

# ============================================================
# TAB 1: RAG CHAT (REVIEW INTELLIGENCE)
# ============================================================
with tab_rag:
    col_left, col_right = st.columns([7, 3])

    with col_left:
        st.markdown("### 💬 Semantic Review Explorer")
        st.markdown(
            "Search through thousands of unstructured customer reviews using vector similarity, "
            "synthesized into clear actionable insights by Groq."
        )

    with col_right:
        cache_status = "⚡ In-Memory / Parquet" if CACHE_FILE.exists() else "⏳ Pending First Load"
        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); padding:10px 14px; border-radius:10px; text-align:right;">
                <div style="font-size:11px; color:#94A3B8; text-transform:uppercase;">Vector Cache Status</div>
                <div style="font-size:13px; font-weight:700; color:#38BDF8;">{cache_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Prompt Suggestion Chips
    st.markdown("##### 💡 Suggested Questions")
    chip_cols = st.columns(len(EXAMPLE_RAG_QUESTIONS))
    for idx, prompt_text in enumerate(EXAMPLE_RAG_QUESTIONS):
        short_label = [
            "🚚 Delivery & Cold Food",
            "🍕 Top Dishes & Praise",
            "📦 Packaging & Leakage",
            "💰 Prices & Portions",
            "⭐ 5-Star Service Praise"
        ][idx]
        with chip_cols[idx]:
            if st.button(short_label, key=f"rag_chip_{idx}", use_container_width=True):
                st.session_state.rag_question = prompt_text
                st.rerun()

    # Question Input
    rag_input = st.text_input(
        "Ask anything about customer experiences, food quality, or delivery:",
        value=st.session_state.rag_question,
        placeholder="e.g. What are the most common complaints regarding delivery time and driver attitude?",
        key="rag_input_box",
    )

    col_btn_run, col_btn_clear, _ = st.columns([1.5, 1, 6])
    with col_btn_run:
        run_rag = st.button("🔍 Analyze Reviews", type="primary", use_container_width=True)
    with col_btn_clear:
        if st.button("Clear", key="clear_rag", use_container_width=True):
            st.session_state.rag_question = ""
            st.session_state.last_rag_answer = None
            st.session_state.last_rag_reviews = None
            st.rerun()

    if (run_rag or (rag_input and rag_input != st.session_state.get("prev_rag_q", ""))) and rag_input.strip():
        st.session_state.prev_rag_q = rag_input.strip()
        with st.spinner("🔍 Retrieving semantically similar reviews & generating synthesis..."):
            try:
                # 1. Load reviews
                reviews_df, status_note = load_rag_reviews(sample_size=sample_size)
                
                # 2. Similarity search
                top_matches = find_similar_reviews(rag_input.strip(), reviews_df, top_k=rag_top_k)
                
                # 3. Groq LLM synthesis
                answer = ask_rag_llm(rag_input.strip(), top_matches, model_name=selected_llm, temperature=llm_temp)
                
                # Store in session state
                st.session_state.last_rag_answer = answer
                st.session_state.last_rag_reviews = top_matches
                st.session_state.rag_status_note = status_note
            except Exception as e:
                st.error(f"❌ Failed to run RAG pipeline: {str(e)}")

    # Render Results if available
    if st.session_state.last_rag_answer and st.session_state.last_rag_reviews is not None:
        st.markdown("---")
        
        # Synthesis Card
        st.markdown(
            """
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span class="metric-badge-chip chip-red">AI SYNTHESIS</span>
                <span style="font-size:16px; font-weight:700; color:var(--text-color, #1e293b);">Executive Customer Intelligence Summary</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        with st.container(border=True):
            st.markdown(st.session_state.last_rag_answer)

        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

        # Grounding Evidence Section
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="metric-badge-chip chip-blue">GROUNDING EVIDENCE</span>
                    <span style="font-size:15px; font-weight:700; color:var(--text-color, #1e293b);">Top {len(st.session_state.last_rag_reviews)} Retrieved Customer Reviews</span>
                </div>
                <span style="font-size:12px; color:var(--text-color, #64748B);">Sorted by Cosine Similarity</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Render Evidence Cards
        for _, r in st.session_state.last_rag_reviews.iterrows():
            stars = "⭐" * int(r.get("rating", 3)) if pd.notnull(r.get("rating")) else "⭐"
            sim_pct = int(r.get("similarity_score", 0) * 100)
            city_val = r.get("city", "India")
            comment_text = r.get("comment", "")

            st.markdown(
                f"""
                <div class="review-item-card">
                    <div class="review-header">
                        <div>
                            <span class="review-city-badge">📍 {city_val}</span>
                            <span class="review-stars" style="margin-left:8px;">{stars} ({r.get('rating', 'N/A')}/5)</span>
                        </div>
                        <span class="review-score">🎯 {sim_pct}% Relevance</span>
                    </div>
                    <div class="review-text">"{comment_text}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("🔎 View Full Retrieved Dataframe & Export CSV"):
            clean_display = st.session_state.last_rag_reviews[["review_id", "city", "rating", "similarity_score", "comment"]].copy()
            st.dataframe(clean_display, hide_index=True, use_container_width=True)
            csv_data = clean_display.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Reviews CSV", csv_data, "zomato_rag_retrieved_reviews.csv", "text/csv")


# ============================================================
# TAB 2: TEXT-TO-SQL ANALYTICS
# ============================================================
with tab_sql:
    st.markdown("### 📊 Natural Language to Snowflake SQL")
    st.markdown(
        "Ask complex business queries in plain English. Groq's high-parameter reasoning models write "
        "production-grade Snowflake SQL, check safety guardrails, and execute live against your data warehouse."
    )

    # Prompt Suggestions
    st.markdown("##### ⚡ Quick Prompt Templates")
    sql_cols = st.columns(3)
    for idx, prompt_text in enumerate(EXAMPLE_SQL_QUESTIONS):
        col_target = sql_cols[idx % 3]
        with col_target:
            if st.button(prompt_text, key=f"sql_chip_{idx}", use_container_width=True):
                st.session_state.sql_question = prompt_text
                st.rerun()

    # Question Input
    sql_input = st.text_input(
        "Enter your analytical question for Snowflake:",
        value=st.session_state.sql_question,
        placeholder="e.g. Top 10 restaurants in Bangalore by total delivered revenue",
        key="sql_input_box",
    )

    col_s1, col_s2, _ = st.columns([1.5, 1, 6])
    with col_s1:
        run_sql_btn = st.button("⚡ Generate & Run SQL", type="primary", use_container_width=True)
    with col_s2:
        if st.button("Clear", key="clear_sql", use_container_width=True):
            st.session_state.sql_question = ""
            st.session_state.last_sql_query = None
            st.session_state.last_sql_df = None
            st.rerun()

    if (run_sql_btn or (sql_input and sql_input != st.session_state.get("prev_sql_q", ""))) and sql_input.strip():
        st.session_state.prev_sql_q = sql_input.strip()
        with st.spinner("🤖 Translating English into optimized Snowflake SQL..."):
            try:
                generated_query = generate_sql(sql_input.strip(), model_name=selected_llm)
                st.session_state.last_sql_query = generated_query

                # Safety Check
                safe, reason = is_sql_safe(generated_query)
                st.session_state.sql_safe = safe
                st.session_state.sql_safe_reason = reason

                if safe:
                    with st.spinner("❄️ Running query on Snowflake DBT_ROLE / MARTS..."):
                        df_res, exec_time = execute_snowflake_query(generated_query)
                        st.session_state.last_sql_df = df_res
                        st.session_state.last_sql_exec_time = exec_time
                else:
                    st.session_state.last_sql_df = None
            except Exception as e:
                st.error(f"❌ Error during SQL generation or execution: {str(e)}")
                st.session_state.last_sql_df = None

    # Render SQL Results
    if st.session_state.last_sql_query:
        st.markdown("---")
        
        # SQL Card Header
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                <div style="display:flex; align-items:center; gap:8px;">
                    <span class="metric-badge-chip chip-red">GENERATED SQL</span>
                    <span style="font-size:15px; font-weight:700; color:var(--text-color, #1e293b);">Snowflake Query</span>
                </div>
                <span class="status-badge" style="border-color:{'#10B981' if st.session_state.get('sql_safe') else '#E23744'};">
                    <span class="status-dot-{'green' if st.session_state.get('sql_safe') else 'red'}"></span>
                    {st.session_state.get('sql_safe_reason', 'Verified')}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.code(st.session_state.last_sql_query, language="sql")

        if not st.session_state.get("sql_safe"):
            st.error(f"🛡️ Security Block: {st.session_state.get('sql_safe_reason')}")
        elif st.session_state.last_sql_df is not None:
            df = st.session_state.last_sql_df
            exec_time = st.session_state.last_sql_exec_time or 0.0

            # Metrics Row
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows Returned", f"{len(df):,}")
            m2.metric("Columns", f"{len(df.columns)}")
            m3.metric("Snowflake Latency", f"{exec_time:.2f} s")

            # Data Table
            st.markdown("##### 📋 Result Set")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Auto-Visualizations
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            if len(df) > 0 and len(num_cols) > 0 and len(cat_cols) > 0:
                st.markdown("##### 📈 Interactive Data Visualization")
                
                v1, v2 = st.columns([1, 1])
                with v1:
                    x_axis = st.selectbox("Category / X-Axis", options=cat_cols, index=0)
                with v2:
                    y_axis = st.selectbox("Metric / Y-Axis", options=num_cols, index=0)

                chart_type = st.radio(
                    "Chart Type",
                    options=["Bar Chart", "Line Chart", "Area Chart"],
                    horizontal=True,
                )

                if chart_type == "Bar Chart":
                    st.bar_chart(df, x=x_axis, y=y_axis)
                elif chart_type == "Line Chart":
                    st.line_chart(df, x=x_axis, y=y_axis)
                elif chart_type == "Area Chart":
                    st.area_chart(df, x=x_axis, y=y_axis)

            # Export button
            csv_sql = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Query Results to CSV", csv_sql, "snowflake_query_results.csv", "text/csv")


# ============================================================
# TAB 3: DATA CATALOG & SCHEMA
# ============================================================
with tab_schema:
    st.markdown("### 📚 Snowflake Data Catalog & Marts Dictionary")
    st.markdown(
        "Reference documentation for all available tables and schemas accessible by the Text-to-SQL engine."
    )

    t1, t2 = st.columns(2)

    with t1:
        st.markdown(
            """
            <div class="content-card">
                <div style="font-size:16px; font-weight:700; color:#FF5A60; margin-bottom:8px;">📊 MART_DAILY_CITY_REVENUNE</div>
                <div style="font-size:13px; color:#94A3B8; margin-bottom:12px;">Aggregated daily financial and order performance per city.</div>
                <table style="width:100%; font-size:12.5px; color:var(--text-color, #1e293b);">
                    <tr><td><b>order_date</b></td><td>Date of orders</td></tr>
                    <tr><td><b>city</b></td><td>Delivery city name</td></tr>
                    <tr><td><b>orders</b></td><td>Total order count</td></tr>
                    <tr><td><b>cancel_rate</b></td><td>Ratio of canceled orders</td></tr>
                    <tr><td><b>gmv</b></td><td>Gross Merchandise Value (Delivered Revenue)</td></tr>
                    <tr><td><b>aov</b></td><td>Average Order Value</td></tr>
                </table>
            </div>
            
            <div class="content-card">
                <div style="font-size:16px; font-weight:700; color:#FF5A60; margin-bottom:8px;">🏪 MART_RESTUARANT_PERFORMANCE</div>
                <div style="font-size:13px; color:#94A3B8; margin-bottom:12px;">Restaurant level volume, revenue, and customer satisfaction metrics.</div>
                <table style="width:100%; font-size:12.5px; color:var(--text-color, #1e293b);">
                    <tr><td><b>restaurant_name</b></td><td>Name of the restaurant</td></tr>
                    <tr><td><b>city</b></td><td>City location</td></tr>
                    <tr><td><b>cuisine</b></td><td>Primary cuisine type</td></tr>
                    <tr><td><b>orders</b></td><td>Delivered order count</td></tr>
                    <tr><td><b>revenue</b></td><td>Total restaurant sales</td></tr>
                    <tr><td><b>avg_customer_rating</b></td><td>Average rating (1.0 to 5.0)</td></tr>
                    <tr><td><b>cancel_rate</b></td><td>Cancellation frequency</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with t2:
        st.markdown(
            """
            <div class="content-card">
                <div style="font-size:16px; font-weight:700; color:#FF5A60; margin-bottom:8px;">⏱️ MART_DELIVERY_SLA</div>
                <div style="font-size:13px; color:#94A3B8; margin-bottom:12px;">Hourly delivery turnaround metrics and SLA breach rates.</div>
                <table style="width:100%; font-size:12.5px; color:var(--text-color, #1e293b);">
                    <tr><td><b>city</b></td><td>City location</td></tr>
                    <tr><td><b>order_hour</b></td><td>Hour of the day (0 - 23)</td></tr>
                    <tr><td><b>delivered_orders</b></td><td>Delivered count in this window</td></tr>
                    <tr><td><b>p50_delivery_min</b></td><td>Median delivery time in minutes</td></tr>
                    <tr><td><b>late_rate</b></td><td>Percentage of deliveries breaching SLA</td></tr>
                </table>
            </div>

            <div class="content-card">
                <div style="font-size:16px; font-weight:700; color:#FF5A60; margin-bottom:8px;">📦 Core Fact & Dimension Tables</div>
                <div style="font-size:13px; color:#94A3B8; margin-bottom:12px;">Underlying transaction logs and entity profiles.</div>
                <table style="width:100%; font-size:12.5px; color:var(--text-color, #1e293b);">
                    <tr><td><b>FCT_ORDERS</b></td><td>Granular order-level records (sales, fee, discounts, rating)</td></tr>
                    <tr><td><b>DIM_RESTAURANTS</b></td><td>Restaurant catalog, cost for two, rating, cuisine</td></tr>
                    <tr><td><b>DIM_CUSTOMER</b></td><td>Customer demographics, age segment, gender</td></tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="display:flex; justify-content:space-between; align-items:center; font-size:12px; color:#64748B;">
        <span>Zomato Analytics Pipeline • Powered by Snowflake & Groq LPU</span>
        <span>Version 2.0 • Hybrid Intelligence Studio</span>
    </div>
    """,
    unsafe_allow_html=True,
)
