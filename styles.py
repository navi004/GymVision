CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: #0a0a0f; }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0a0a0f 0%, #12121e 50%, #0d1117 100%);
        border: 1px solid #1e2030;
        border-radius: 20px;
        padding: 48px 40px;
        text-align: center;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle at 50% 50%, rgba(99,102,241,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #6366f1, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 8px 0;
        letter-spacing: -1px;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #64748b;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin: 0 0 24px 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a78bfa;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 4px 14px;
        border-radius: 20px;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0 4px;
    }

    /* Mode cards */
    .mode-card {
        background: #0f1117;
        border: 1px solid #1e2030;
        border-radius: 16px;
        padding: 32px 28px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        height: 100%;
    }
    .mode-card:hover { border-color: #6366f1; transform: translateY(-2px); }
    .mode-card.active { border-color: #6366f1; background: rgba(99,102,241,0.08); }
    .mode-icon { font-size: 2.8rem; margin-bottom: 12px; }
    .mode-title { font-size: 1.3rem; font-weight: 700; color: #e2e8f0; margin-bottom: 8px; }
    .mode-desc { font-size: 0.9rem; color: #64748b; line-height: 1.5; }

    /* Metric tiles */
    .metric-tile {
        background: #0f1117;
        border: 1px solid #1e2030;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #e2e8f0;
        line-height: 1;
        margin-bottom: 4px;
    }
    .metric-label { font-size: 0.8rem; color: #64748b; text-transform: uppercase; letter-spacing: 1px; }
    .metric-good { color: #22c55e; }
    .metric-warn { color: #f59e0b; }
    .metric-bad  { color: #ef4444; }

    /* Alert banners */
    .alert-good { background: rgba(34,197,94,0.12); border: 1px solid rgba(34,197,94,0.3);
                  color: #4ade80; border-radius: 8px; padding: 10px 16px; font-weight: 600; margin: 6px 0; }
    .alert-warn { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.3);
                  color: #fbbf24; border-radius: 8px; padding: 10px 16px; font-weight: 600; margin: 6px 0; }
    .alert-bad  { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.3);
                  color: #f87171; border-radius: 8px; padding: 10px 16px; font-weight: 600; margin: 6px 0; }

    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 28px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        padding: 12px 32px;
        font-size: 1rem;
        width: 100%;
        transition: all 0.2s;
    }
    .stButton > button:hover { opacity: 0.9; transform: translateY(-1px); }
    .stSelectbox > div > div { background: #0f1117; border-color: #1e2030; color: #e2e8f0; }
    div[data-testid="stFileUploader"] { background: #0f1117; border: 2px dashed #1e2030; border-radius: 12px; }
    .stProgress > div > div { background: linear-gradient(90deg, #6366f1, #a78bfa); }
    h1,h2,h3 { color: #e2e8f0 !important; }
    p, li { color: #94a3b8; }
    .stMarkdown p { color: #94a3b8; }
    [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
    [data-testid="stMetricLabel"] { color: #64748b !important; }
    .stTabs [data-baseweb="tab"] { color: #64748b; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { color: #a78bfa; border-bottom-color: #a78bfa; }
    .stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1e2030; }
    .stInfo { background: rgba(99,102,241,0.1); border-color: rgba(99,102,241,0.3); color: #a78bfa; }
    .stSuccess { background: rgba(34,197,94,0.1); border-color: rgba(34,197,94,0.3); }
    .stWarning { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.3); }
    footer { display: none; }
    #MainMenu { visibility: hidden; }
</style>
"""
