"""
GlucoSight AI Dashboard - Diabetes Risk Assessment
====================================================
Modern dark-themed clinical decision support dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import logging
import hashlib
import json
import re
import textwrap
import base64
import sys
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    roc_curve,
)

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from utils.benchmarking import benchmark_models
from utils.explainability import explain_prediction
from utils.monitoring import session_metrics
from utils.recommendations import (
    DISCLAIMER,
    generate_clinical_summary,
    generate_recommendations,
)
from utils.reporting import (
    build_html_report as build_rich_html_report,
    build_pdf_report as build_rich_pdf_report,
    build_text_report as build_rich_text_report,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve project resources from this file so launch commands work from any cwd.
APP_DIR = SCRIPT_DIR
ASSET_DIR = APP_DIR / "assets"
MODEL_DIR = str(APP_DIR / "models")
DATA_DIR = str(APP_DIR.parent / "data")
MODEL_FILE = os.path.join(MODEL_DIR, "diabetes_model.pkl")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_FILE = os.path.join(MODEL_DIR, "feature_names.pkl")
MODEL_META_FILE = os.path.join(MODEL_DIR, "model_metadata.json")

# Page configuration
st.set_page_config(
    page_title="GlucoSight AI",
    page_icon=str(ASSET_DIR / "glucosight_logo.png"),
    layout="wide",
    initial_sidebar_state="expanded"
)

def render_html(markup, **_kwargs):
    """Render HTML without Streamlit treating indented blocks as code."""
    cleaned = textwrap.dedent(markup).strip()
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)

# Custom CSS for dark GlucoSight AI Dashboard
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles - GlucoSight AI Brand Colors */
    :root {
        --bg-primary: #0e151f;
        --bg-secondary: #101925;
        --bg-card: #172436;
        --bg-card-deep: #121c2a;
        --bg-glass: rgba(23, 36, 54, 0.90);
        --text-primary: #ffffff;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;
        --accent-blue: #4b80e0;
        --accent-blue-light: #6296ed;
        --accent-blue-dark: #3267c5;
        --accent-green: #10b981;
        --accent-green-light: #34d399;
        --accent-red: #ef4444;
        --accent-red-light: #f87171;
        --accent-orange: #f59e0b;
        --border-color: rgba(75, 128, 224, 0.20);
        --shadow-glass: 0 18px 55px rgba(5, 12, 23, 0.28);
        --font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    html, body, .stApp {
        background: var(--bg-primary) !important;
    }

    .main {
        font-family: var(--font-family);
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .block-container {
        max-width: 1320px !important;
        padding-top: 0 !important;
        padding-bottom: 2.2rem !important;
        padding-left: 2.1rem !important;
        padding-right: 2.1rem !important;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stHeader"] {background-color: transparent !important;}
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-color) !important;
        width: 230px !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: transparent !important;
        padding: 0 !important;
    }
    
    /* Custom Sidebar Brand */
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1.5rem 1.25rem 1.65rem 1.25rem;
        margin-bottom: 1.15rem;
    }
    
    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-blue-light));
        color: #ffffff;
        box-shadow: 0 0 26px rgba(37, 99, 235, 0.28);
    }
    
    .brand-text {
        display: flex;
        flex-direction: column;
    }
    
    .brand-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.025em;
    }
    
    .brand-sub {
        font-size: 0.65rem;
        font-weight: 700;
        color: var(--text-muted);
        letter-spacing: 0.15em;
        text-transform: uppercase;
        margin-top: -0.1rem;
    }
    
    /* Sidebar menu */
    .sidebar-menu {
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
        padding: 0 1rem;
    }
    
    .menu-item {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 0.78rem 0.92rem;
        border-radius: 12px;
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease-in-out;
        text-decoration: none !important;
    }
    
    .menu-item:hover {
        background: rgba(255, 255, 255, 0.03);
        color: var(--text-primary);
    }
    
    .menu-item.active {
        background: rgba(37, 99, 235, 0.20);
        color: #d9e7ff;
        font-weight: 600;
        position: relative;
    }
    
    .menu-item.active::after {
        content: '';
        position: absolute;
        right: 1rem;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--accent-blue-light);
    }
    
    .menu-icon {
        flex-shrink: 0;
    }
    
    .menu-chevron {
        margin-left: auto;
        opacity: 0.6;
    }
    
    /* Project card */
    .project-card, .ask-input-container { display: none; }
    
    .project-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }
    
    .project-buttons {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    
    .project-btn {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        padding: 0.45rem 0;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .project-btn:hover {
        background: var(--accent-blue);
        color: white;
        border-color: var(--accent-blue);
    }
    
    .project-description {
        font-size: 0.75rem;
        color: var(--text-muted);
        line-height: 1.4;
    }
    
    /* Credits section */
    .credits-section {
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.1) 0%, rgba(37, 99, 235, 0.03) 100%);
        border: 1px solid rgba(37, 99, 235, 0.15);
        border-radius: 12px;
        padding: 1rem;
        margin: 0 0.75rem 1.5rem 0.75rem;
    }
    
    .credits-text {
        font-size: 0.8rem;
        color: var(--accent-blue-light);
        font-weight: 500;
        margin-bottom: 0.75rem;
        text-align: center;
    }
    
    .upgrade-btn {
        background: var(--accent-blue);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
        width: 100%;
        transition: all 0.2s;
    }
    
    .upgrade-btn:hover {
        background: var(--accent-blue-light);
    }
    
    /* Ask input */
    .ask-input-container {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin: 0 0.75rem;
        display: flex;
        align-items: center;
    }

    .ask-input-container {
        display: none !important;
    }
    
    .ask-input {
        background: transparent;
        border: none;
        color: var(--text-primary);
        font-size: 0.85rem;
        width: 100%;
        outline: none;
    }
    
    /* Sidebar footer */
    .sidebar-footer {
        position: fixed;
        left: 18px;
        bottom: 14px;
        width: 196px;
        padding: 0.82rem 0.95rem;
        border: 1px solid var(--border-color);
        border-radius: 14px;
        background: rgba(18, 28, 42, 0.96);
        font-size: 0.75rem;
        color: var(--text-muted);
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
        justify-content: space-between;
    }
    
    /* Top navigation bar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-left: -2.1rem;
        margin-right: -2.1rem;
        margin-bottom: 1.7rem;
        padding: 1rem 2.1rem;
        border-bottom: 1px solid var(--border-color);
        background: rgba(14, 21, 31, 0.88);
        backdrop-filter: blur(16px);
        position: sticky;
        top: 0;
        z-index: 20;
    }
    
    .breadcrumb {
        font-size: 0.875rem;
        font-weight: 500;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .breadcrumb span {
        color: var(--text-muted);
        font-weight: 400;
    }
    
    .nav-actions {
        display: flex;
        gap: 0.65rem;
    }
    
    .nav-btn {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        padding: 0.55rem 0.9rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    
    .nav-btn:hover {
        background: rgba(255, 255, 255, 0.08);
        color: var(--text-primary);
    }
    
    .nav-btn.primary {
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-blue-dark));
        color: white;
        border-color: rgba(59, 130, 246, 0.45);
    }
    
    .nav-btn.primary:hover {
        background: var(--accent-blue-light);
    }

    [data-testid="stSidebar"] .stElementContainer:has(button[data-testid="stBaseButton-secondary"]) {
        width: 100% !important;
    }

    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
        background: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid transparent !important;
        color: var(--text-secondary) !important;
        box-shadow: none !important;
        justify-content: flex-start !important;
        min-height: 2.75rem !important;
        padding: 0.75rem 0.95rem !important;
        border-radius: 12px !important;
        font-size: 0.9rem !important;
        font-weight: 650 !important;
        text-align: left !important;
        width: 100% !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"]:hover {
        background: rgba(37, 99, 235, 0.18) !important;
        color: #d9e7ff !important;
        border-color: rgba(59, 130, 246, 0.20) !important;
        transform: none !important;
    }

    .st-key-top_actions [data-testid="stHorizontalBlock"] {
        align-items: center;
        margin-bottom: 1.15rem;
    }

    .st-key-top_actions .stElementContainer:has(button) {
        width: 100% !important;
    }

    .st-key-top_actions .stButton > button,
    .st-key-top_actions .stDownloadButton > button,
    .st-key-top_actions button[data-testid="stBaseButton-secondary"] {
        min-height: 2.42rem !important;
        height: 2.42rem !important;
        width: 100% !important;
        padding: 0 0.9rem !important;
        border-radius: 12px !important;
        font-size: 0.78rem !important;
        font-weight: 700 !important;
        line-height: 1 !important;
        white-space: nowrap !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 0.4rem !important;
        box-shadow: none !important;
    }

    .st-key-top_actions .stButton > button,
    .st-key-top_actions button[data-testid="stBaseButton-secondary"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-secondary) !important;
    }

    .st-key-top_actions .stDownloadButton > button {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-secondary) !important;
    }

    .st-key-top_actions .st-key-download_pdf_report button,
    .st-key-download_pdf_report button,
    .st-key-download_pdf_report button[data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-blue-dark) 100%) !important;
        color: white !important;
        border-color: rgba(59, 130, 246, 0.45) !important;
    }

    .st-key-top_actions .stButton > button:hover,
    .st-key-top_actions .stDownloadButton > button:hover,
    .st-key-top_actions button[data-testid="stBaseButton-secondary"]:hover {
        transform: translateY(-1px);
        background: rgba(255, 255, 255, 0.08) !important;
        color: var(--text-primary) !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.18) !important;
    }

    .st-key-workspace_tabs {
        width: 100%;
        max-width: 100%;
        padding: 0.22rem;
        margin: 1.25rem 0 1.35rem;
        border: 1px solid var(--border-color);
        border-radius: 15px;
        background: rgba(23, 33, 47, 0.72);
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.03);
        overflow-x: auto;
    }

    .st-key-workspace_tabs [data-testid="stHorizontalBlock"] {
        gap: 0.25rem !important;
        align-items: stretch;
    }

    .st-key-workspace_tabs .stElementContainer:has(button[data-testid="stBaseButton-secondary"]) {
        width: 100% !important;
    }

    .st-key-workspace_tabs .stButton > button,
    .st-key-workspace_tabs button[data-testid="stBaseButton-secondary"] {
        min-height: 2.35rem !important;
        height: auto !important;
        padding: 0.45rem 0.6rem !important;
        border-radius: 12px !important;
        background: transparent !important;
        border: 1px solid transparent !important;
        color: var(--text-secondary) !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        line-height: 1.08 !important;
        white-space: normal !important;
        text-align: center !important;
        box-shadow: none !important;
        width: 100% !important;
    }

    .st-key-workspace_tabs button[data-testid="stBaseButton-secondary"] p {
        margin: 0 !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }

    .st-key-workspace_tabs .stButton > button:hover,
    .st-key-workspace_tabs .stButton > button:focus,
    .st-key-workspace_tabs .stButton > button:focus-visible,
    .st-key-workspace_tabs button[data-testid="stBaseButton-secondary"]:hover,
    .st-key-workspace_tabs button[data-testid="stBaseButton-secondary"]:focus,
    .st-key-workspace_tabs button[data-testid="stBaseButton-secondary"]:focus-visible {
        transform: none !important;
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(255, 255, 255, 0.04) !important;
        color: var(--text-primary) !important;
        box-shadow: none !important;
    }
    
    /* Title area */
    .section-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.03em;
        margin-bottom: 0.25rem;
    }
    
    .section-subtitle {
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-bottom: 2rem;
    }
    
    /* Glass card / panels */
    .assessment-panel {
        background: linear-gradient(180deg, var(--bg-card), var(--bg-card-deep));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.35rem;
        box-shadow: var(--shadow-glass);
        min-height: 438px;
        display: flex;
        flex-direction: column;
    }

    div[data-testid="stForm"] {
        background: linear-gradient(180deg, var(--bg-card), var(--bg-card-deep));
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.35rem;
        box-shadow: var(--shadow-glass);
        min-height: 438px;
    }

    div[data-testid="stForm"] [data-testid="stVerticalBlock"] {
        gap: 0.2rem;
    }
    
    /* Scrollable compact inputs wrapper */
    .scrollable-inputs {
        max-height: 295px;
        overflow-y: auto;
        padding-right: 0.25rem;
    }
    
    .scrollable-inputs::-webkit-scrollbar {
        width: 4px;
    }
    
    .scrollable-inputs::-webkit-scrollbar-track {
        background: transparent;
    }
    
    .scrollable-inputs::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 2px;
    }
    
    .card-header-flex {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }
    
    .card-header-left {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .card-header-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    .card-header-badge {
        font-size: 0.65rem;
        font-weight: 700;
        background: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.25);
        color: var(--accent-blue-light);
        padding: 0.15rem 0.5rem;
        border-radius: 4px;
        letter-spacing: 0.05em;
    }
    
    .card-header-subtitle {
        font-size: 0.8rem;
        color: var(--text-secondary);
        margin-bottom: 1.5rem;
    }
    
    /* Custom Input fields inside grids */
    .custom-input-wrap {
        background: rgba(14, 21, 31, 0.94);
        border: 1px solid var(--border-color);
        border-radius: 11px;
        padding: 0.5rem 0.75rem;
        position: relative;
        display: flex;
        flex-direction: column;
        justify-content: center;
        height: 50px;
        margin-bottom: 0.65rem;
        transition: border-color 0.2s;
    }
    
    .custom-input-wrap:focus-within {
        border-color: var(--accent-blue);
    }
    
    .custom-input-label {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.1rem;
    }
    
    .custom-input-suffix {
        position: absolute;
        right: 0.75rem;
        top: -0.8rem;
        font-size: 0.75rem;
        color: var(--text-muted);
        pointer-events: none;
        font-weight: 500;
        z-index: 10;
    }
    
    /* Streamlit numeric inputs overrides */
    .custom-input-wrap div[data-testid="stNumberInput"] {
        background: transparent !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .custom-input-wrap div[data-testid="stNumberInput"] > div {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }

    .custom-input-wrap div[data-testid="stNumberInput"] div {
        background-color: transparent !important;
    }
    
    .custom-input-wrap input {
        background: transparent !important;
        border: none !important;
        color: var(--text-primary) !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        padding: 0 !important;
        padding-right: 3rem !important;
        height: auto !important;
        width: 100% !important;
        outline: none !important;
        box-shadow: none !important;
    }
    
    .custom-input-wrap div[data-testid="stNumberInput"] button,
    .custom-input-wrap button[data-testid="stNumberInputStepUp"],
    .custom-input-wrap button[data-testid="stNumberInputStepDown"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        position: absolute !important;
        right: -100px !important;
    }
    
    /* Custom generate button style */
    .generate-btn-container {
        margin-top: 1rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-blue-dark) 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1.5rem !important;
        width: 100% !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
    }

    .stButton > button:active,
    .stButton > button:focus,
    .stButton > button:focus-visible {
        background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-blue-dark) 100%) !important;
        color: white !important;
        border: none !important;
        outline: none !important;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3) !important;
    }
    
    .hipaa-flex {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        margin-top: 1rem;
        color: var(--text-muted);
        font-size: 0.7rem;
    }
    
    /* Hero Risk Card */
    .hero-risk-card {
        background:
            linear-gradient(135deg, #1c4375 0%, #182b45 48%, #19375f 100%);
        border: 1px solid var(--border-color);
        border-radius: 18px;
        padding: 1.45rem 1.65rem;
        margin-bottom: 1.35rem;
        min-height: 218px;
        overflow: hidden;
        position: relative;
    }

    .hero-risk-layout {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 2rem;
    }

    .hero-risk-left {
        max-width: 620px;
    }

    .hero-pills {
        display: flex;
        gap: 0.45rem;
        margin-bottom: 1rem;
        align-items: center;
    }

    .hero-pill {
        border: 1px solid rgba(96, 165, 250, 0.24);
        background: rgba(37, 99, 235, 0.18);
        color: #4fa3ff;
        border-radius: 999px;
        padding: 0.18rem 0.62rem;
        font-size: 0.68rem;
        font-weight: 700;
    }

    .hero-kicker {
        color: #90a0b5;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .hero-risk-title {
        color: #fff;
        font-size: 2.55rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: 0;
        margin-bottom: 0.7rem;
    }

    .hero-risk-copy {
        color: #a8b1c1;
        font-size: 0.95rem;
        line-height: 1.45;
        max-width: 560px;
        margin-bottom: 1.15rem;
    }

    .hero-risk-stats {
        display: grid;
        grid-template-columns: 140px 140px;
        gap: 1.6rem;
    }

    .hero-risk-label {
        font-size: 0.72rem;
        font-weight: 700;
        color: #8997aa;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.22rem;
    }

    .hero-risk-value {
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.45rem;
    }
    
    .hero-risk-value.red-text {
        color: var(--accent-red-light);
    }
    
    .hero-risk-value.green-text {
        color: var(--accent-green-light);
    }

    .hero-risk-value.yellow-text {
        color: var(--accent-orange);
    }
    
    .hero-risk-bar-track {
        height: 5px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 3px;
        overflow: hidden;
    }
    
    .hero-risk-bar {
        height: 100%;
        border-radius: 3px;
    }
    
    .hero-risk-bar.red-bg {
        background: linear-gradient(90deg, #ff4b55, #f43f5e);
    }
    
    .hero-risk-bar.green-bg {
        background: linear-gradient(90deg, var(--accent-green-light), var(--accent-green));
    }

    .hero-risk-bar.yellow-bg {
        background: linear-gradient(90deg, #fbbf24, var(--accent-orange));
    }

    .risk-ring {
        width: 162px;
        height: 162px;
        position: relative;
        flex: 0 0 auto;
        display: grid;
        place-items: center;
    }

    .risk-ring svg {
        position: absolute;
        inset: 0;
        transform: rotate(-82deg);
    }

    .risk-ring-value {
        color: #fff;
        font-size: 1.95rem;
        font-weight: 800;
        line-height: 1;
    }

    .risk-ring-label {
        color: #8f9aae;
        font-size: 0.62rem;
        font-weight: 800;
        letter-spacing: 0.2em;
        margin-top: 0.35rem;
        text-transform: uppercase;
        text-align: center;
    }
    
    /* Metrics grid 2x2 */
    .metrics-grid-2x2 {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
    }
    
    .metric-card-custom {
        background: linear-gradient(180deg, var(--bg-card), var(--bg-card-deep));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1rem 1.05rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 120px;
    }
    
    .metric-card-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-icon-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        border-radius: 50%;
    }
    
    .metric-icon-wrap.blue-bg-light { background: rgba(59, 130, 246, 0.12); color: var(--accent-blue-light); }
    .metric-icon-wrap.yellow-bg-light { background: rgba(245, 158, 11, 0.12); color: var(--accent-orange); }
    .metric-icon-wrap.red-bg-light { background: rgba(239, 68, 68, 0.12); color: var(--accent-red-light); }
    
    .metric-card-label {
        font-size: 0.65rem;
        font-weight: 600;
        color: var(--text-secondary);
        letter-spacing: 0.05em;
    }
    
    .metric-card-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0.25rem 0;
    }
    
    .metric-card-unit {
        font-size: 0.85rem;
        font-weight: 400;
        color: var(--text-secondary);
    }
    
    .metric-card-sub {
        font-size: 0.7rem;
        font-weight: 600;
    }
    
    .metric-card-sub.blue-text { color: var(--accent-blue-light); }
    .metric-card-sub.yellow-text { color: var(--accent-orange); }
    .metric-card-sub.red-text { color: var(--accent-red-light); }
    .metric-card-sub.green-text { color: var(--accent-green-light); }
    
    /* Tabs Overrides */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(23, 36, 54, 0.82) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 18px !important;
        gap: 0.1rem !important;
        padding: 0.25rem !important;
        margin-top: 1.75rem !important;
        margin-bottom: 1.4rem !important;
        width: fit-content !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: 14px !important;
        padding: 0.48rem 1rem !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        border: 1px solid transparent !important;
        transition: all 0.2s !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.08) !important;
        color: var(--text-primary) !important;
        border-color: var(--border-color) !important;
        font-weight: 600 !important;
    }
    
    /* Content cards */
    .content-card {
        background: linear-gradient(180deg, var(--bg-card), var(--bg-card-deep));
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-color);
        border-radius: 14px;
        padding: 1.45rem;
        box-shadow: var(--shadow-glass);
        margin-bottom: 1rem;
    }
    
    .tag {
        background: rgba(59, 130, 246, 0.1);
        color: var(--accent-blue-light);
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }
    
    /* Centered SHAP force-plot layout */
    .shap-plot-container {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        margin-top: 1.5rem;
    }
    
    .shap-row {
        display: flex;
        align-items: center;
        width: 100%;
        height: 36px;
    }
    
    .shap-feature-label {
        width: 25%;
        font-size: 0.85rem;
        color: var(--text-secondary);
        font-weight: 500;
        text-align: left;
    }
    
    .shap-bar-wrapper {
        width: 63%;
        height: 100%;
        position: relative;
        display: flex;
        align-items: center;
    }
    
    .shap-center-line {
        position: absolute;
        left: 50%;
        top: 0;
        bottom: 0;
        width: 1px;
        background: rgba(255, 255, 255, 0.15);
        z-index: 2;
    }
    
    .shap-bar {
        position: absolute;
        height: 12px;
        border-radius: 6px;
        z-index: 1;
    }
    
    .shap-bar.positive {
        left: 50%;
        background: linear-gradient(90deg, var(--accent-red-light), var(--accent-red));
    }
    
    .shap-bar.negative {
        right: 50%;
        background: linear-gradient(270deg, var(--accent-green-light), var(--accent-green));
    }
    
    .shap-value {
        width: 12%;
        text-align: right;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .shap-value.positive {
        color: var(--accent-red-light);
    }
    
    .shap-value.negative {
        color: var(--accent-green-light);
    }
    
    .legend {
        display: flex;
        gap: 1.5rem;
        margin-top: 1.5rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-color);
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.4rem;
        font-size: 0.75rem;
        color: var(--text-secondary);
    }
    
    .legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    
    .legend-dot.red { background-color: var(--accent-red); }
    .legend-dot.green { background-color: var(--accent-green); }
    
    /* Recommendations 2x2 grid */
    .recommendations-grid-2x2 {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .recommendation-card-custom {
        background: rgba(6, 8, 20, 0.4);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 1.25rem;
        display: flex;
        gap: 0.85rem;
        transition: border-color 0.25s, transform 0.25s;
    }
    
    .recommendation-card-custom:hover {
        border-color: var(--accent-blue);
        transform: translateY(-2px);
    }
    
    .recom-icon-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    
    .recom-icon-wrap.red-bg-light { background: rgba(239, 68, 68, 0.1); color: var(--accent-red-light); }
    .recom-icon-wrap.yellow-bg-light { background: rgba(245, 158, 11, 0.1); color: var(--accent-orange); }
    .recom-icon-wrap.green-bg-light { background: rgba(16, 185, 129, 0.1); color: var(--accent-green-light); }
    
    .recom-content {
        display: flex;
        flex-direction: column;
    }
    
    .recom-tag {
        font-size: 0.65rem;
        font-weight: 700;
        color: var(--text-muted);
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    
    .recom-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.4rem;
    }
    
    .recom-desc {
        font-size: 0.8rem;
        color: var(--text-secondary);
        line-height: 1.45;
    }
    
    /* Analytics Gauges and Bars */
    .analytics-grid-2x2 {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .analytics-gauge-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1rem 0;
        position: relative;
    }
    
    .analytics-gauge-svg {
        transform: rotate(-90deg);
    }
    
    .analytics-gauge-text-wrap {
        position: absolute;
        top: 55%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        display: flex;
        flex-direction: column;
    }
    
    .analytics-gauge-val {
        font-size: 2rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1;
    }
    
    .analytics-gauge-label {
        font-size: 0.65rem;
        font-weight: 700;
        color: var(--accent-red-light);
        margin-top: 0.35rem;
        letter-spacing: 0.05em;
    }
    
    .analytics-gauge-legend {
        display: flex;
        gap: 1rem;
        justify-content: center;
        margin-top: 1rem;
    }
    
    .analytics-gauge-legend-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.3rem;
    }
    
    .analytics-gauge-legend-dot.green { background-color: var(--accent-green); }
    .analytics-gauge-legend-dot.yellow { background-color: var(--accent-orange); }
    .analytics-gauge-legend-dot.red { background-color: var(--accent-red); }
    
    .analytics-gauge-legend-text {
        font-size: 0.7rem;
        color: var(--text-muted);
    }
    
    /* Horizontal Bar Chart */
    .bar-chart-container {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        padding-top: 0.5rem;
    }
    
    .bar-chart-row {
        display: flex;
        align-items: center;
    }
    
    .bar-chart-label {
        width: 20%;
        font-size: 0.8rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .bar-chart-track {
        width: 80%;
        height: 12px;
        background: rgba(255, 255, 255, 0.04);
        border-radius: 3px;
        overflow: hidden;
        position: relative;
    }
    
    .bar-chart-fill {
        height: 100%;
        border-radius: 3px;
    }
    
    .bar-chart-fill.blue-1 { background-color: #3b82f6; }
    .bar-chart-fill.blue-2 { background-color: #60a5fa; }
    .bar-chart-fill.blue-3 { background-color: #93c5fd; }
    .bar-chart-fill.green-1 { background-color: #10b981; }
    .bar-chart-fill.yellow-1 { background-color: #f59e0b; }
    
    .bar-chart-axis {
        display: flex;
        margin-top: 0.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        padding-top: 0.35rem;
    }
    
    .bar-chart-axis-label {
        width: 20%;
    }
    
    .bar-chart-axis-ticks {
        width: 80%;
        display: flex;
        justify-content: space-between;
        font-size: 0.65rem;
        color: var(--text-muted);
    }
    
    /* Dataset boxes and Model cards */
    .dataset-box-grid {
        display: flex;
        gap: 0.75rem;
        margin-top: 1rem;
    }
    
    .dataset-box {
        flex: 1;
        background: rgba(6, 8, 20, 0.5);
        border: 1px solid var(--border-color);
        padding: 0.75rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .dataset-box-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text-primary);
        margin-bottom: 0.15rem;
    }
    
    .dataset-box-lbl {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 500;
    }
    
    .model-metrics-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.65rem;
        margin-top: 1rem;
    }
    
    .model-metric-item {
        background: rgba(6, 8, 20, 0.5);
        border: 1px solid var(--border-color);
        padding: 0.6rem 0.8rem;
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .model-metric-lbl {
        font-size: 0.7rem;
        color: var(--text-secondary);
        font-weight: 500;
    }
    
    .model-metric-val {
        font-size: 0.85rem;
        font-weight: 700;
        color: var(--text-primary);
    }
    
    /* Footer logo & badges */
    .app-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-top: 2rem;
        margin-top: 3rem;
        border-top: 1px solid var(--border-color);
    }
    
    .footer-text {
        font-size: 0.75rem;
        color: var(--text-muted);
    }
    
    .footer-badges {
        display: flex;
        gap: 0.5rem;
    }
    
    .footer-badge {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--border-color);
        color: var(--text-secondary);
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 500;
    }

    /* Custom loading screen with logo */
    [data-testid="stStatusWidget"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }

    /* Content card header, title, subtitle */
    .content-card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 1.25rem;
    }
    
    .content-card-title {
        font-size: 1.28rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.2rem 0;
        padding: 0;
    }
    
    .content-card-subtitle {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin: 0;
        padding: 0;
    }

    div[data-testid="stTextInput"] input {
        background: rgba(14, 21, 31, 0.96) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        min-height: 38px !important;
    }

    .history-table {
        background: rgba(18, 28, 42, 0.72);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        overflow: hidden;
        margin-top: 1rem;
    }

    .history-row {
        display: grid;
        grid-template-columns: 1.1fr 1.7fr 1.1fr 1.25fr 1.25fr 0.9fr;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .history-row:last-child {
        border-bottom: none;
    }

    .history-head {
        background: rgba(255, 255, 255, 0.02);
        color: #64748b;
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .history-head > div,
    .history-cell {
        padding: 0.9rem 1rem;
    }

    .history-cell {
        color: #94a3b8;
        font-size: 0.84rem;
    }

    .history-id,
    .history-probability {
        color: #d8dee9;
        font-weight: 700;
    }

    .history-status {
        background: rgba(15, 23, 42, 0.55);
        color: #cbd5e1;
        border: 1px solid rgba(148, 163, 184, 0.14);
        padding: 0.2rem 0.62rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .history-action {
        text-align: right;
    }

    @media (max-width: 980px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .top-nav {
            margin-left: -1rem;
            margin-right: -1rem;
            padding: 0.9rem 1rem;
            align-items: flex-start;
            gap: 0.8rem;
            flex-direction: column;
        }
        .nav-actions {
            width: 100%;
            overflow-x: auto;
            padding-bottom: 0.2rem;
        }
        .hero-risk-layout {
            align-items: flex-start;
            flex-direction: column;
        }
        .risk-ring {
            width: 134px;
            height: 134px;
        }
        .metrics-grid-2x2,
        .recommendations-grid-2x2,
        .analytics-grid-2x2 {
            grid-template-columns: 1fr;
        }
        .hero-risk-stats {
            grid-template-columns: 1fr 1fr;
            width: 100%;
        }
        .hero-risk-title {
            font-size: 2.1rem;
        }
        .app-footer {
            align-items: flex-start;
            flex-direction: column;
            gap: 0.8rem;
        }
        .history-table {
            overflow-x: auto;
        }
        .history-row {
            min-width: 760px;
        }
    }
    </style>
""", unsafe_allow_html=True)


def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    logger.info(f"Ensured directory exists: {MODEL_DIR}")


def get_dataset_path():
    """Return the dataset path used by the app."""
    cleaned_path = os.path.join(DATA_DIR, 'diabetes_cleaned.csv')
    raw_path = os.path.join(DATA_DIR, 'diabetes.csv')
    if os.path.exists(cleaned_path):
        return cleaned_path
    if os.path.exists(raw_path):
        return raw_path
    return raw_path


def dataset_fingerprint():
    """Hash the active dataset so stale model artifacts can be detected."""
    path = get_dataset_path()
    if not os.path.exists(path):
        return {"path": path, "sha256": None, "size_bytes": 0}

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": path,
        "sha256": digest.hexdigest(),
        "size_bytes": os.path.getsize(path),
        "modified_at": os.path.getmtime(path),
    }


def read_model_metadata():
    """Load saved model metadata if available."""
    try:
        if os.path.exists(MODEL_META_FILE):
            with open(MODEL_META_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Could not read model metadata: {e}")
    return {}


def model_artifacts_are_current():
    """Validate that saved artifacts match the current dataset."""
    required = [MODEL_FILE, SCALER_FILE, FEATURE_NAMES_FILE, MODEL_META_FILE]
    if not all(os.path.exists(f) for f in required):
        return False

    metadata = read_model_metadata()
    current = dataset_fingerprint()
    return (
        metadata.get("dataset_sha256") == current.get("sha256")
        and metadata.get("dataset_size_bytes") == current.get("size_bytes")
    )


@st.cache_data
def load_data():
    """Load and preprocess the dataset with proper error handling."""
    try:
        # Try to load cleaned data first
        cleaned_path = os.path.join(DATA_DIR, 'diabetes_cleaned.csv')
        raw_path = os.path.join(DATA_DIR, 'diabetes.csv')
        
        if os.path.exists(cleaned_path):
            logger.info(f"Loading cleaned dataset from {cleaned_path}")
            df = pd.read_csv(cleaned_path)
        elif os.path.exists(raw_path):
            logger.info(f"Loading raw dataset from {raw_path}")
            df = pd.read_csv(raw_path)
            # Preprocess: replace invalid zeros with NaN and impute with median
            invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in invalid_zero_cols:
                df[col] = df[col].replace(0, np.nan)
                df[col] = df[col].fillna(df[col].median())
            logger.info("Applied preprocessing to raw data")
        else:
            raise FileNotFoundError(f"Dataset not found in {DATA_DIR}")
        
        # Validate dataset
        required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Dataset loaded successfully with {len(df)} records")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        st.error(f"❌ **Error loading dataset**: {e}")
        st.error("Please ensure diabetes.csv or diabetes_cleaned.csv is in the data directory.")
        st.stop()


def train_and_save_model():
    """Train model and save it to disk with proper error handling."""
    try:
        logger.info("Starting model training...")
        
        # Load data
        df = load_data()
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info(f"Data split: {len(X_train)} training, {len(X_test)} test samples")
        
        # Median imputation handles physiological zero values converted during load_data().
        preprocessor = SimpleImputer(strategy='median')
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Train model. Random Forest performed best on hold-out F1/Brier among
        # tested classical ML models for the full 768-row PIMA dataset.
        model = RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight='balanced',
            min_samples_leaf=3,
            n_jobs=-1
        )
        model.fit(X_train_processed, y_train)
        
        # Calculate model performance on the hold-out split.
        X_test_processed = preprocessor.transform(X_test)
        y_pred = model.predict(X_test_processed)
        y_prob = model.predict_proba(X_test_processed)[:, 1]
        train_score = model.score(X_train_processed, y_train)
        test_score = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else 0.0
        brier = brier_score_loss(y_test, y_prob)
        logger.info(f"Model trained - Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}, AUC: {auc:.3f}")
        
        # Save model components
        setup_directories()
        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {MODEL_FILE}")
        
        with open(SCALER_FILE, 'wb') as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Preprocessor saved to {SCALER_FILE}")
        
        with open(FEATURE_NAMES_FILE, 'wb') as f:
            pickle.dump(X.columns.tolist(), f)
        logger.info(f"Feature names saved to {FEATURE_NAMES_FILE}")

        fingerprint = dataset_fingerprint()
        metadata = {
            "model_type": re.sub(r"(?<!^)(?=[A-Z])", " ", model.__class__.__name__),
            "dataset_path": os.path.relpath(fingerprint.get("path"), APP_DIR),
            "dataset_sha256": fingerprint.get("sha256"),
            "dataset_size_bytes": fingerprint.get("size_bytes"),
            "row_count": int(len(df)),
            "feature_count": int(X.shape[1]),
            "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
            "train_size": int(len(X_train)),
            "test_size": int(len(X_test)),
            "split_random_state": 42,
            "test_fraction": 0.2,
            "metrics": {
                "train_accuracy": float(train_score),
                "accuracy": float(test_score),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "brier": float(brier),
            },
            "feature_medians": {col: float(X[col].median()) for col in X.columns},
            "feature_iqr": {
                col: float(max(X[col].quantile(0.75) - X[col].quantile(0.25), 1e-6))
                for col in X.columns
            },
        }
        with open(MODEL_META_FILE, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Model metadata saved to {MODEL_META_FILE}")
        
        return model, preprocessor, X.columns.tolist()
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        st.error(f"❌ **Error training model**: {e}")
        st.stop()


def load_model():
    """Load trained model from disk or train new one if not exists."""
    try:
        if model_artifacts_are_current():
            logger.info("Loading existing trained model with matching dataset metadata...")
            
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            
            with open(SCALER_FILE, 'rb') as f:
                scaler = pickle.load(f)
            
            with open(FEATURE_NAMES_FILE, 'rb') as f:
                feature_names = pickle.load(f)

            # Pickled sklearn estimators are not guaranteed to work across versions.
            # Probe the complete inference path before accepting saved artifacts.
            probe = load_data()[feature_names].head(1)
            probe_processed = scaler.transform(probe)
            model.predict_proba(probe_processed)
            
            logger.info("Model loaded successfully")
            return model, scaler, feature_names
        else:
            logger.info("Model artifacts missing or stale, training new model...")
            return train_and_save_model()
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        logger.info("Attempting to train new model...")
        return train_and_save_model()


def evaluate_loaded_model(model, preprocessor, feature_names, saved_metadata):
    """Evaluate the loaded artifacts on their deterministic stratified hold-out split."""
    df = load_data()
    X = df[feature_names]
    y = df['Outcome']
    split_random_state = int(saved_metadata.get("split_random_state", 42))
    test_fraction = float(saved_metadata.get("test_fraction", 0.2))
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=split_random_state,
        stratify=y,
    )

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    estimator_name = re.sub(r"(?<!^)(?=[A-Z])", " ", model.__class__.__name__)
    metadata = dict(saved_metadata)
    metadata.update({
        "model_type": estimator_name,
        "row_count": int(len(df)),
        "feature_count": int(len(feature_names)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "split_random_state": split_random_state,
        "test_fraction": test_fraction,
        "metrics": {
            "train_accuracy": float(model.score(X_train_processed, y_train)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, y_prob)),
            "brier": float(brier_score_loss(y_test, y_prob)),
        },
        "roc_curve": [[float(x), float(y)] for x, y in zip(fpr, tpr)],
    })
    metadata.pop("model_version", None)
    return metadata


@st.cache_data(show_spinner=False)
def get_benchmark_results(dataset_hash):
    """Cache deterministic model comparisons for the active dataset version."""
    del dataset_hash  # The hash is the cache invalidation key.
    df = load_data()
    return benchmark_models(df.drop(columns="Outcome"), df["Outcome"], random_state=42)


def prepare_patient_frame(user_input, feature_names, reference_data):
    """Apply the same zero-value normalization used by the inference path."""
    frame = pd.DataFrame([user_input])[feature_names].copy()
    for column in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        if frame[column].iloc[0] == 0:
            frame.loc[:, column] = reference_data[column].median()
    return frame


def predict_patient_probability(model, preprocessor, feature_names, user_input, reference_data):
    frame = prepare_patient_frame(user_input, feature_names, reference_data)
    processed = preprocessor.transform(frame)
    return float(model.predict_proba(processed)[0][1])


def validate_input(user_input):
    """Validate user input data."""
    try:
        # Check for negative values where not appropriate
        if user_input['Glucose'] < 0 or user_input['BloodPressure'] < 0 or user_input['BMI'] < 0:
            raise ValueError("Glucose, Blood Pressure, and BMI cannot be negative")
        
        # Check for reasonable ranges
        if user_input['Glucose'] > 600:
            st.warning("⚠️ Glucose value seems unusually high. Please verify.")
        if user_input['BMI'] > 100:
            st.warning("⚠️ BMI value seems unusually high. Please verify.")
        if user_input['Age'] > 150:
            st.warning("⚠️ Age value seems unrealistic. Please verify.")
        
        return True
        
    except Exception as e:
        logger.error(f"Input validation error: {e}")
        st.error(f"❌ **Input validation error**: {e}")
        return False


def get_risk_level(probability):
    """Determine risk level based on probability."""
    if probability >= 0.7:
        return "High Risk", "high-risk"
    elif probability >= 0.4:
        return "Moderate Risk", "moderate-risk"
    else:
        return "Low Risk", "low-risk"


def build_clinical_summary(prediction_results, model_metadata):
    """Build a downloadable plain-text clinical summary."""
    return build_rich_text_report(prediction_results, model_metadata)


def build_report_html(prediction_results, model_metadata):
    """Build a downloadable HTML report."""
    return build_rich_html_report(prediction_results, model_metadata)


def build_report_pdf(prediction_results, model_metadata):
    """Build a compact downloadable PDF report."""
    return build_rich_pdf_report(prediction_results, model_metadata)


def render_custom_number_input(label, key, default_value, min_value, max_value, step, suffix, help_text=""):
    st.markdown(f'<div class="custom-input-wrap"><div class="custom-input-label">{label}</div>', unsafe_allow_html=True)
    val = st.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        step=step,
        key=key,
        label_visibility="collapsed",
        help=help_text
    )
    if suffix:
        st.markdown(f'<div class="custom-input-suffix">{suffix}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    return val


def render_patient_assessment_card(model, scaler, feature_names, model_metadata):
    # Horizontal logo above header
    st.markdown(f"""
    <div style="display:flex; justify-content:flex-start; margin-bottom:1rem;">
        <img src="data:image/png;base64,{base64.b64encode((ASSET_DIR / 'glucosight_horizontal_logo.png').read_bytes()).decode()}" alt="GlucoSight AI" style="width: 350px; height: auto; object-fit: contain;"/>
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    encounter_id = (
        st.session_state.prediction_results.get('assessment_id')
        if st.session_state.prediction_results else None
    ) or f"ASM-{len(st.session_state.assessment_history) + 1:05d}"
    st.markdown(f"""
    <div class="card-header-flex">
        <div class="card-header-left">
            <svg viewBox="0 0 24 24" width="18" height="18" stroke="#3b82f6" stroke-width="2.5" fill="none" class="menu-icon"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            <span class="card-header-title">Patient Assessment</span>
        </div>
        <div class="card-header-badge">ENCOUNTER - #{encounter_id}</div>
    </div>
    <div class="card-header-subtitle">Enter clinical inputs to compute diabetes risk.</div>
    """, unsafe_allow_html=True)

    with st.form("patient_form", clear_on_submit=False):
        if st.session_state.prediction_made:
            st.markdown('<div class="scrollable-inputs">', unsafe_allow_html=True)

        # Row 1
        col_r1a, col_r1b = st.columns(2)
        with col_r1a:
            pregnancies = render_custom_number_input("PREGNANCIES", "pregnancies_input", 2, 0, 20, 1, "")
        with col_r1b:
            glucose = render_custom_number_input("GLUCOSE", "glucose_input", 168.0, 0.0, 600.0, 1.0, "mg/dL")

        # Row 2
        col_r2a, col_r2b = st.columns(2)
        with col_r2a:
            blood_pressure = render_custom_number_input("BLOOD PRESSURE", "blood_pressure_input", 88.0, 0.0, 200.0, 1.0, "mmHg")
        with col_r2b:
            skin_thickness = render_custom_number_input("SKIN THICKNESS", "skin_thickness_input", 35.0, 0.0, 100.0, 1.0, "mm")

        # Row 3
        col_r3a, col_r3b = st.columns(2)
        with col_r3a:
            insulin = render_custom_number_input("INSULIN", "insulin_input", 180.0, 0.0, 1000.0, 1.0, "μU/mL")
        with col_r3b:
            bmi = render_custom_number_input("BMI", "bmi_input", 33.6, 0.0, 100.0, 0.1, "kg/m²")

        # Row 4
        col_r4a, col_r4b = st.columns(2)
        with col_r4a:
            diabetes_pedigree = render_custom_number_input("DIABETES PEDIGREE", "diabetes_pedigree_input", 0.627, 0.0, 3.0, 0.001, "")
        with col_r4b:
            age = render_custom_number_input("AGE", "age_input", 50, 0, 150, 1, "yrs")

        if st.session_state.prediction_made:
            st.markdown('</div>', unsafe_allow_html=True)

        submit_button = st.form_submit_button("✨ Generate AI Assessment", type="primary")
        
        st.markdown(f"""
        <div class="hipaa-flex">
            <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
            <span>HIPAA-aligned - Inference runs on validated PIMA-trained {model_metadata.get('model_type', 'model')}</span>
        </div>
        """, unsafe_allow_html=True)
        
    return submit_button, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age


def render_results_panel():
    if st.session_state.prediction_results is None:
        return
        
    results = st.session_state.prediction_results
    prob = results['probability']
    risk_percent = prob * 100
    
    # Inputs values
    user_input = results['user_input']
    age = user_input['Age']
    bmi = user_input['BMI']
    bp = user_input['BloodPressure']
    glucose = user_input['Glucose']
    
    # Dynamic metric calculations
    age_diff = age - 48
    age_status = f"+{age_diff} vs avg" if age_diff >= 0 else f"{age_diff} vs avg"
    
    if bmi >= 30:
        bmi_status = "Obese class I"
        bmi_class = "red-text"
    elif bmi >= 25:
        bmi_status = "Overweight"
        bmi_class = "yellow-text"
    else:
        bmi_status = "Normal"
        bmi_class = "green-text"
        
    if bp >= 90:
        bp_status = "Stage 2"
        bp_class = "red-text"
    elif bp >= 80:
        bp_status = "Stage 1"
        bp_class = "yellow-text"
    else:
        bp_status = "Normal"
        bp_class = "green-text"
        
    if glucose >= 126:
        glucose_status = "Hyperglycemic"
        glucose_class = "red-text"
    elif glucose >= 100:
        glucose_status = "Prediabetic"
        glucose_class = "yellow-text"
    else:
        glucose_status = "Normal"
        glucose_class = "green-text"
        
    # Determine risk level label and color for badge
    risk_level = results['risk_level']
    if risk_percent >= 70:
        risk_badge_color = '#ef4444'
        risk_badge_bg = 'rgba(239,68,68,0.12)'
        risk_color_class = 'red-text'
        risk_bar_class = 'red-bg'
        ring_color_start = '#f87171'
        ring_color_end = '#dc2626'
    elif risk_percent >= 40:
        risk_badge_color = '#f59e0b'
        risk_badge_bg = 'rgba(245,158,11,0.12)'
        risk_color_class = 'yellow-text'
        risk_bar_class = 'yellow-bg'
        ring_color_start = '#fbbf24'
        ring_color_end = '#d97706'
    else:
        risk_badge_color = '#10b981'
        risk_badge_bg = 'rgba(16,185,129,0.12)'
        risk_color_class = 'green-text'
        risk_bar_class = 'green-bg'
        ring_color_start = '#34d399'
        ring_color_end = '#059669'

    risk_title = risk_level.upper()
    confidence_percent = max(prob, 1 - prob) * 100
    if prob >= 0.7:
        risk_copy = "The model indicates a high probability of Type 2 diabetes.<br>Prompt clinical review and diagnostic confirmation are recommended."
    elif prob >= 0.4:
        risk_copy = "The model indicates a moderate probability of Type 2 diabetes.<br>Preventive follow-up and appropriate screening are recommended."
    else:
        risk_copy = "The model indicates a low probability of Type 2 diabetes.<br>Continue routine screening and healthy habits."
    ring_circumference = 2 * np.pi * 62
    ring_offset = ring_circumference * (1 - min(max(risk_percent, 0), 100) / 100)

    render_html(f"""
    <div class="hero-risk-card">
        <div class="hero-risk-layout">
            <div class="hero-risk-left">
                <div class="hero-pills">
                    <span class="hero-pill">
                        <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none" style="display:inline; vertical-align:-2px; margin-right:4px;"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                        AI Assessment Generated
                    </span>
                    <span class="hero-pill" style="background:rgba(15,23,42,0.28); color:#9aa8bd;">1.2s</span>
                </div>
                <div class="hero-kicker">Risk Category</div>
                <div class="hero-risk-title">{risk_title}</div>
                <div class="hero-risk-copy">{risk_copy}</div>
                <div class="hero-risk-stats">
                    <div>
                        <div class="hero-risk-label">Probability</div>
                        <div class="hero-risk-value {risk_color_class}">{risk_percent:.1f}%</div>
                        <div class="hero-risk-bar-track">
                            <div class="hero-risk-bar {risk_bar_class}" style="width: {risk_percent:.1f}%"></div>
                        </div>
                    </div>
                    <div>
                        <div class="hero-risk-label">Confidence</div>
                        <div class="hero-risk-value green-text">{confidence_percent:.1f}%</div>
                        <div class="hero-risk-bar-track">
                            <div class="hero-risk-bar green-bg" style="width: {confidence_percent:.1f}%"></div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="risk-ring" aria-label="Risk score {risk_percent:.1f}%">
                <svg viewBox="0 0 150 150" width="150" height="150">
                    <circle cx="75" cy="75" r="62" fill="none" stroke="{risk_badge_bg}" stroke-width="12" />
                    <circle cx="75" cy="75" r="62" fill="none" stroke="url(#riskRingGrad)" stroke-width="12" stroke-linecap="round" stroke-dasharray="{ring_circumference:.1f}" stroke-dashoffset="{ring_offset:.1f}" />
                    <defs>
                        <linearGradient id="riskRingGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="{ring_color_start}" />
                            <stop offset="100%" stop-color="{ring_color_end}" />
                        </linearGradient>
                    </defs>
                </svg>
                <div style="position:relative; text-align:center;">
                    <div class="risk-ring-value" style="color:{risk_badge_color};">{risk_percent:.1f}<span style="font-size:1rem;">%</span></div>
                    <div class="risk-ring-label">Risk Score</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="metrics-grid-2x2">
        <!-- Card 1: Age -->
        <div class="metric-card-custom">
            <div class="metric-card-header">
                <div class="metric-icon-wrap blue-bg-light">
                    <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                </div>
                <span class="metric-card-label">AGE</span>
            </div>
            <div class="metric-card-value">{age} <span class="metric-card-unit">yrs</span></div>
            <div class="metric-card-sub blue-text">
                <svg viewBox="0 0 24 24" width="10" height="10" stroke="currentColor" stroke-width="2.5" fill="none" style="display:inline; vertical-align:middle; margin-right:2px;"><polyline points="18 15 12 9 6 15"></polyline></svg>
                {age_status}
            </div>
        </div>
        <!-- Card 2: BMI -->
        <div class="metric-card-custom">
            <div class="metric-card-header">
                <div class="metric-icon-wrap yellow-bg-light">
                    <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><line x1="7" y1="7" x2="7.01" y2="7"></line></svg>
                </div>
                <span class="metric-card-label">BMI</span>
            </div>
            <div class="metric-card-value">{bmi:.1f} <span class="metric-card-unit">kg/m²</span></div>
            <div class="metric-card-sub {bmi_class}">{bmi_status}</div>
        </div>
        <!-- Card 3: Blood Pressure -->
        <div class="metric-card-custom">
            <div class="metric-card-header">
                <div class="metric-icon-wrap yellow-bg-light">
                    <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"></path></svg>
                </div>
                <span class="metric-card-label">BLOOD PRESSURE</span>
            </div>
            <div class="metric-card-value">{int(bp)} <span class="metric-card-unit">mmHg</span></div>
            <div class="metric-card-sub {bp_class}">{bp_status}</div>
        </div>
        <!-- Card 4: Glucose -->
        <div class="metric-card-custom">
            <div class="metric-card-header">
                <div class="metric-icon-wrap red-bg-light">
                    <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>
                </div>
                <span class="metric-card-label">GLUCOSE</span>
            </div>
            <div class="metric-card-value">{int(glucose)} <span class="metric-card-unit">mg/dL</span></div>
            <div class="metric-card-sub {glucose_class}">{glucose_status}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_roc_curve(roc_points, auc_value=None):
    """Render an SVG ROC curve chart as a styled content card."""
    roc_points = roc_points or [(0, 0), (1, 1)]
    # Scale to SVG coordinates (width=260, height=200, with some padding)
    W, H, pad = 260, 180, 20
    def sx(x): return pad + x * (W - 2*pad)
    def sy(y): return H - pad - y * (H - 2*pad)
    
    path_d = " ".join([f"{'M' if i==0 else 'L'} {sx(p[0]):.1f},{sy(p[1]):.1f}" for i, p in enumerate(roc_points)])
    fill_d = path_d + f" L {sx(1.0):.1f},{sy(0):.1f} L {sx(0):.1f},{sy(0):.1f} Z"
    diag_d = f"M {sx(0):.1f},{sy(0):.1f} L {sx(1.0):.1f},{sy(1.0):.1f}"
    
    # Axis ticks
    ticks_x = [0, 0.25, 0.5, 0.75, 1.0]
    ticks_y = [0, 0.25, 0.5, 0.75, 1.0]
    
    tick_lines_x = "".join([f'<line x1="{sx(t):.1f}" y1="{sy(0)+2:.1f}" x2="{sx(t):.1f}" y2="{sy(0)-2:.1f}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/><text x="{sx(t):.1f}" y="{sy(0)+12:.1f}" fill="#64748b" font-size="7" text-anchor="middle">{t:.2f}</text>' for t in ticks_x])
    tick_lines_y = "".join([f'<line x1="{sx(0)-2:.1f}" y1="{sy(t):.1f}" x2="{sx(0)+2:.1f}" y2="{sy(t):.1f}" stroke="rgba(255,255,255,0.15)" stroke-width="1"/><text x="{sx(0)-5:.1f}" y="{sy(t)+3:.1f}" fill="#64748b" font-size="7" text-anchor="end">{t:.2f}</text>' for t in ticks_y])
    
    # Grid lines
    grid = "".join([f'<line x1="{sx(0):.1f}" y1="{sy(t):.1f}" x2="{sx(1.0):.1f}" y2="{sy(t):.1f}" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>' for t in ticks_y[1:]])
    grid += "".join([f'<line x1="{sx(t):.1f}" y1="{sy(0):.1f}" x2="{sx(t):.1f}" y2="{sy(1.0):.1f}" stroke="rgba(255,255,255,0.04)" stroke-width="1"/>' for t in ticks_x[1:]])
    
    svg = f"""
    <svg viewBox="0 0 {W} {H+15}" width="100%" style="overflow:visible;">
        <defs>
            <linearGradient id="rocFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.25"/>
                <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.02"/>
            </linearGradient>
        </defs>
        {grid}
        <!-- Diagonal reference line -->
        <path d="{diag_d}" fill="none" stroke="rgba(255,255,255,0.18)" stroke-width="1" stroke-dasharray="4 3"/>
        <!-- ROC curve fill area -->
        <path d="{fill_d}" fill="url(#rocFill)"/>
        <!-- ROC curve line -->
        <path d="{path_d}" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>
        <!-- Axis lines -->
        <line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(1.0):.1f}" y2="{sy(0):.1f}" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
        <line x1="{sx(0):.1f}" y1="{sy(0):.1f}" x2="{sx(0):.1f}" y2="{sy(1.0):.1f}" stroke="rgba(255,255,255,0.2)" stroke-width="1"/>
        {tick_lines_x}
        {tick_lines_y}
        <!-- AUC label -->
        <text x="{sx(0.6):.1f}" y="{sy(0.3):.1f}" fill="#60a5fa" font-size="9" font-weight="700">AUC = {(auc_value if auc_value is not None else 0):.3f}</text>
        <!-- Axis labels -->
        <text x="{sx(0.5):.1f}" y="{H+13:.1f}" fill="#64748b" font-size="8" text-anchor="middle">False Positive Rate</text>
        <text x="8" y="{sy(0.5):.1f}" fill="#64748b" font-size="8" text-anchor="middle" transform="rotate(-90, 8, {sy(0.5):.1f})">True Positive Rate</text>
    </svg>
    """
    st.markdown(f'<div style="background:rgba(6,8,20,0.4); border:1px solid rgba(255,255,255,0.08); border-radius:12px; padding:1rem; margin-top:0;">{svg}</div>', unsafe_allow_html=True)


def render_history_tab():
    """Render the assessment history tab with search."""
    history_data = list(reversed(st.session_state.assessment_history))
    df_history = pd.DataFrame(history_data)

    # Render as a styled div table. Streamlit's HTML sanitizer can expose raw
    # table rows in some versions, so this avoids native table tags.
    rows_html = ""
    for row_number, (_, row) in enumerate(df_history.iterrows()):
        level = row['Risk']
        if level == 'High':
            badge_color = '#ef4444'
            badge_bg = 'rgba(239,68,68,0.1)'
        elif level == 'Moderate':
            badge_color = '#f59e0b'
            badge_bg = 'rgba(245,158,11,0.1)'
        else:
            badge_color = '#10b981'
            badge_bg = 'rgba(16,185,129,0.1)'
        rows_html += f"""
        <div class="history-row">
            <div class="history-cell history-id">{row['ID']}</div>
            <div class="history-cell">{row['Date']}</div>
            <div class="history-cell">
                <span style="width:6px; height:6px; display:inline-block; border-radius:50%; background:{badge_color}; margin-right:0.45rem;"></span>{level}
            </div>
            <div class="history-cell history-probability">{row['Probability']}</div>
            <div class="history-cell"><span class="history-status">{row['Status']}</span></div>
            <div class="history-cell history-action" style="color:var(--accent-blue-light); font-weight:600;">{'Current' if row_number == 0 else 'Stored'}</div>
        </div>
        """
    
    render_html(f"""
    <div class="content-card">
        <div class="content-card-header">
            <div>
                <h3 class="content-card-title">Assessment History</h3>
                <p class="content-card-subtitle">Recent clinical inferences</p>
            </div>
        </div>
        <div class="history-table">
            <div class="history-row history-head">
                <div>ID</div>
                <div>Date</div>
                <div>Risk</div>
                <div>Probability</div>
                <div>Status</div>
                <div style="text-align:right;">Action</div>
            </div>
            {rows_html or '<div style="padding:2rem; color:#64748b; text-align:center;">No assessments yet. Generate an AI assessment to add the first record.</div>'}
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:1rem; color:#94a3b8; font-size:0.78rem;">
            <span>Showing {len(history_data)} of {len(history_data)}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application logic."""
    logger.info("Starting GlucoSight AI Dashboard")
    
    # Initialize session state for prediction
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False
    if 'prediction_results' not in st.session_state:
        st.session_state.prediction_results = None
    if 'assessment_history' not in st.session_state:
        st.session_state.assessment_history = []
    if 'workspace_tab' not in st.session_state:
        st.session_state.workspace_tab = "Insights & Explainability"
    if 'active_sidebar_nav' not in st.session_state:
        st.session_state.active_sidebar_nav = "Dashboard"

    # Load and evaluate the actual artifacts before rendering any model claims.
    try:
        with st.spinner("🔄 Loading GlucoSight AI model..."):
            model, scaler, feature_names = load_model()
            model_metadata = evaluate_loaded_model(
                model, scaler, feature_names, read_model_metadata()
            )
    except Exception as e:
        logger.error(f"Fatal error loading model: {e}")
        st.error("❌ **Fatal Error**: Unable to load or evaluate the model. Please check the logs.")
        st.stop()

    model_type = model_metadata["model_type"]
        
    # Sidebar - GlucoSight AI Dashboard
    with st.sidebar:
        # Custom Logo with actual image
        st.markdown("""
        <div class="sidebar-brand">
            <div class="brand-icon">
                <img src="data:image/png;base64,{}" alt="GlucoSight AI" style="width: 36px; height: 36px; border-radius: 50%; object-fit: contain;"/>
            </div>
            <div class="brand-text">
                <div class="brand-name">GlucoSight</div>
                <div class="brand-sub">AI &bull; CLINICAL</div>
            </div>
        </div>
        """.format(base64.b64encode((ASSET_DIR / 'glucosight_logo.png').read_bytes()).decode()), unsafe_allow_html=True)
        
        # Navigation Menu
        nav_items = [
            ("Dashboard", "Insights & Explainability"),
            ("Patient Assessment", "Insights & Explainability"),
            ("Analytics", "Analytics"),
            ("Reports", "History"),
            ("Documentation", "Documentation"),
        ]
        for label, target_tab in nav_items:
            active = st.session_state.active_sidebar_nav == label
            button_label = f"{'● ' if active else ''}{label}"
            if st.button(button_label, key=f"sidebar_nav_{label.lower().replace(' ', '_')}"):
                st.session_state.workspace_tab = target_tab
                st.session_state.active_sidebar_nav = label
                st.rerun()
        
        # Project card and upgrade info
        st.markdown("""
        <div class="project-card">
            <div class="project-title">Built GlucoSight AI dashboard</div>
            <div class="project-buttons">
                <button class="project-btn">Details</button>
                <button class="project-btn">Preview</button>
            </div>
            <p class="project-description">
                Built GlucoSight AI as an enterprise-grade dark dashboard: minimal sidebar with brand + model status, sticky topbar with PDF/share actions, an 8-field Patient Assessment panel, a hero risk card with animated radial gauge and count-up probability/confidence, four clinical metric cards, a tabbed workspace covering SHAP-style explainability, Recharts analytics, clinical recommendations, model insights with ROC curve, and a searchable assessment history table.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="ask-input-container">
            <input type="text" class="ask-input" placeholder="Ask Lovable..." />
        </div>
        """, unsafe_allow_html=True)
        
        # Model status footer at the bottom of the sidebar
        st.markdown(f"""
        <div class="sidebar-footer">
            <span style="display:flex; align-items:center; gap:0.3rem;">
                <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
                {model_type}
            </span>
            <span style="display:flex; align-items:center; gap:0.3rem;">
                <span style="width:6px; height:6px; border-radius:50%; background-color:#10b981; display:inline-block; box-shadow:0 0 8px #10b981;"></span>
                Operational
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    # Top navigation bar
    st.markdown(
        """
        <div class="breadcrumb" style="margin-bottom:0.85rem;">
            Clinical Workspace <span style="font-size:0.75rem; margin:0 0.25rem; color:#64748b;">&gt;</span> Diabetes Risk Assessment
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="top_actions"):
        top_left, share_col, summary_col, report_col = st.columns([1, 0.3, 0.3, 0.3], gap="small")
        with share_col:
            if st.button("Share", key="top_share_button"):
                st.toast("Share link ready: http://localhost:8501/")
        with summary_col:
            st.download_button(
                "Clinical Summary",
                data=build_clinical_summary(st.session_state.prediction_results, model_metadata),
                file_name="glucosight_clinical_summary.txt",
                mime="text/plain",
                key="download_clinical_summary",
            )
        with report_col:
            pdf_report = build_report_pdf(st.session_state.prediction_results, model_metadata)
            st.download_button(
                "Export PDF Report",
                data=pdf_report or build_report_html(st.session_state.prediction_results, model_metadata),
                file_name="glucosight_clinical_report.pdf" if pdf_report else "glucosight_clinical_report.html",
                mime="application/pdf" if pdf_report else "text/html",
                key="download_pdf_report",
            )

    render_html(f"""
    <div class="content-card" style="padding:1.35rem 1.5rem; margin-bottom:1.35rem; background:linear-gradient(115deg, rgba(35,67,112,.94), rgba(18,28,42,.96));">
        <div style="display:flex; justify-content:space-between; gap:2rem; align-items:center; flex-wrap:wrap;">
            <div style="max-width:760px;">
                <div class="recom-tag" style="margin-bottom:.55rem;">EXPLAINABLE CLINICAL AI</div>
                <div style="font-size:1.5rem; font-weight:750; color:#fff; letter-spacing:-.025em;">Diabetes risk intelligence, not just a prediction</div>
                <div style="color:#b8c7da; margin-top:.45rem; line-height:1.55;">Combine {model_type} inference with patient-level explanations, what-if simulation, clinical narrative, model benchmarking, and session monitoring.</div>
            </div>
            <div style="display:flex; gap:.55rem; flex-wrap:wrap;">
                <span class="hero-pill">Local inference</span><span class="hero-pill">Explainable</span><span class="hero-pill">No API key</span>
            </div>
        </div>
    </div>
    """)
    
    # Main content starts directly below the clinical workspace topbar.
    
    # Initialize form variables
    submit_button = False
    pregnancies = 2
    glucose = 168.0
    blood_pressure = 88.0
    skin_thickness = 35.0
    insulin = 180.0
    bmi = 33.6
    diabetes_pedigree = 0.627
    age = 50

    # Render inputs card and optional results card based on prediction state
    if not st.session_state.prediction_made:
        col_main, col_empty = st.columns([1.2, 0.8], gap="large")
        with col_main:
            submit_button, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age = (
                render_patient_assessment_card(model, scaler, feature_names, model_metadata)
            )
    else:
        col_inputs, col_results = st.columns([0.78, 2.02], gap="large")
        with col_inputs:
            submit_button, pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age = (
                render_patient_assessment_card(model, scaler, feature_names, model_metadata)
            )
        with col_results:
            render_results_panel()
            
        # Optional Collapsed Status Widget at bottom left (matches screenshot 3)
        st.markdown(f"""
        <div class="collapsed-status-widget" style="position: fixed; bottom: 1.5rem; left: 1.5rem; z-index: 99; display: flex; align-items: center; gap: 1rem; background: rgba(13, 18, 51, 0.85); border: 1px solid rgba(255,255,255,0.08); padding: 0.5rem 1rem; border-radius: 8px; backdrop-filter: blur(10px); box-shadow: 0 4px 12px rgba(0,0,0,0.5);">
            <span style="font-size: 0.75rem; color: #a0aec0; display: flex; align-items: center; gap: 0.35rem;">
                <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="9" y1="3" x2="9" y2="21"></line></svg>
                {model_type}
            </span>
            <span style="width: 1px; height: 12px; background: rgba(255,255,255,0.15);"></span>
            <span style="font-size: 0.75rem; color: #a0aec0; display: flex; align-items: center; gap: 0.35rem;">
                <span style="width:6px; height:6px; border-radius:50%; background-color:#10b981; display:inline-block; box-shadow: 0 0 8px #10b981;"></span>
                System Operational
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Handle prediction when form is submitted
    if submit_button:
        user_input = {
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': blood_pressure,
            'SkinThickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'DiabetesPedigreeFunction': diabetes_pedigree,
            'Age': age
        }
        
        # Validate input
        if not validate_input(user_input):
            st.stop()
        
        # Prepare input for prediction
        df = load_data()
        input_df = prepare_patient_frame(user_input, feature_names, df)
        
        # Apply the saved preprocessing object.
        input_processed = scaler.transform(input_df)
        
        # Make prediction
        try:
            probability = model.predict_proba(input_processed)[0][1]
            prediction = model.predict(input_processed)[0]
            logger.info(f"Prediction made - Probability: {probability:.3f}, Prediction: {prediction}")
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            st.error(f"❌ **Error making prediction**: {e}")
            st.stop()
        
        # Determine risk level
        risk_level, risk_class = get_risk_level(probability)
        
        # Calculate feature contributions. For tree models, combine global
        # feature importance with this patient's distance from the training median.
        if hasattr(model, "coef_"):
            coefficients = model.coef_[0]
            contributions = coefficients * input_processed[0]
        elif hasattr(model, "feature_importances_"):
            metadata = read_model_metadata()
            medians = metadata.get("feature_medians", {})
            iqrs = metadata.get("feature_iqr", {})
            risk_directions = {
                'Pregnancies': 1, 'Glucose': 1, 'BloodPressure': 1,
                'SkinThickness': 1, 'Insulin': 1, 'BMI': 1,
                'DiabetesPedigreeFunction': 1, 'Age': 1
            }
            contributions = []
            for feature, value, importance in zip(feature_names, input_df.iloc[0].values, model.feature_importances_):
                median = medians.get(feature, float(df[feature].median()))
                iqr = max(iqrs.get(feature, float(df[feature].quantile(0.75) - df[feature].quantile(0.25))), 1e-6)
                z_score = np.clip((float(value) - median) / iqr, -3, 3)
                contributions.append(float(importance) * z_score * risk_directions.get(feature, 1))
            contributions = np.array(contributions)
            coefficients = model.feature_importances_
        else:
            contributions = np.zeros(len(feature_names))
            coefficients = np.zeros(len(feature_names))

        feature_contributions = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Value': input_df.iloc[0].values,
            'Contribution': contributions
        }).sort_values('Contribution', key=abs, ascending=False)

        background_processed = scaler.transform(df[feature_names])
        feature_contributions, global_importance, explanation_method = explain_prediction(
            model=model,
            processed_patient=input_processed,
            processed_background=background_processed,
            raw_patient=input_df.iloc[0].to_dict(),
            feature_names=feature_names,
            fallback=feature_contributions.set_index('Feature').reindex(feature_names)['Contribution'].values,
        )
        top_features = [
            {
                'feature': row['Feature'],
                'contribution': float(row['Contribution']),
                'direction': 'increased' if row['Contribution'] >= 0 else 'reduced',
            }
            for _, row in feature_contributions.head(5).iterrows()
        ]
        recommendations = generate_recommendations(probability, user_input)
        clinical_summary = generate_clinical_summary(probability, user_input, top_features)
        
        # Store prediction results in session state
        assessment_id = f"ASM-{len(st.session_state.assessment_history) + 1:05d}"
        assessed_at = datetime.now().astimezone()
        st.session_state.prediction_results = {
            'probability': probability,
            'prediction': prediction,
            'risk_level': risk_level,
            'risk_class': risk_class,
            'user_input': user_input,
            'feature_contributions': feature_contributions,
            'global_importance': global_importance,
            'explanation_method': explanation_method,
            'recommendations': recommendations,
            'clinical_summary': clinical_summary,
            'assessment_id': assessment_id,
            'assessed_at': assessed_at.isoformat(),
        }
        st.session_state.assessment_history.append({
            'ID': assessment_id,
            'Date': assessed_at.strftime('%Y-%m-%d %H:%M'),
            'Risk': risk_level.replace(' Risk', ''),
            'Probability': f"{probability * 100:.1f}%",
            'Status': 'Escalated' if probability >= 0.7 else ('Follow-up' if probability >= 0.4 else 'Routine'),
            'probability': float(probability),
            'confidence': float(max(probability, 1 - probability)),
            'user_input': dict(user_input),
        })
        st.session_state.pop('simulation_results', None)
        st.session_state.prediction_made = True
        st.rerun()

    # Retrieve prediction results for tabs
    prediction_results = st.session_state.prediction_results
    metrics = model_metadata.get("metrics", {})
    row_count = int(model_metadata.get("row_count", 0) or 0)
    train_size = int(model_metadata.get("train_size", 0) or 0)
    test_size = int(model_metadata.get("test_size", 0) or 0)
    val_size = 0
    model_type = model_metadata.get("model_type", model.__class__.__name__)
    accuracy_pct = metrics.get("accuracy", 0) * 100
    precision_pct = metrics.get("precision", 0) * 100
    recall_pct = metrics.get("recall", 0) * 100
    auc_val = metrics.get("auc", 0)
    f1_val = metrics.get("f1", 0)
    brier_val = metrics.get("brier", 0)
    roc_points = model_metadata.get("roc_curve", [])

    # Workspace navigation. Native buttons keep the sections functional from both
    # the sidebar and the in-page tab strip.
    tab_labels = ["Insights & Explainability", "Analytics", "Recommendations", "Model Insights", "History", "Documentation"]
    if st.session_state.workspace_tab not in tab_labels:
        st.session_state.workspace_tab = "Insights & Explainability"

    with st.container(key="workspace_tabs"):
        tab_cols = st.columns([2.05, 1.0, 1.45, 1.35, 0.85, 1.45], gap="small")
        for tab_label, tab_col in zip(tab_labels, tab_cols):
            with tab_col:
                active_prefix = "● " if st.session_state.workspace_tab == tab_label else ""
                if st.button(f"{active_prefix}{tab_label}", key=f"workspace_tab_{tab_label.lower().replace(' ', '_').replace('&', 'and')}"):
                    st.session_state.workspace_tab = tab_label
                    st.session_state.active_sidebar_nav = {
                        "Analytics": "Analytics",
                        "History": "Reports",
                        "Documentation": "Documentation",
                    }.get(tab_label, "Dashboard")
                    st.rerun()

    selected_tab = st.session_state.workspace_tab
    
    if selected_tab == "Insights & Explainability":
        # Generate feature bars HTML
        if prediction_results is not None:
            feature_contributions = prediction_results['feature_contributions']
            
            # Find maximum absolute contribution for scaling
            max_contrib = feature_contributions['Contribution'].abs().max()
            if max_contrib == 0:
                max_contrib = 1.0
                
            feature_bars_html = ""
            for idx, row in feature_contributions.head(6).iterrows():
                feature_name = row['Feature']
                contribution = row['Contribution']
                value = row['Value']
                
                display_name = feature_name
                if feature_name == 'DiabetesPedigreeFunction':
                    display_name = 'Pedigree'
                
                is_positive = contribution > 0
                contribution_class = "positive" if is_positive else "negative"
                bar_width = (abs(contribution) / max_contrib) * 45
                
                if feature_name == 'Glucose':
                    value_str = f"{value:.0f} mg/dL"
                elif feature_name == 'BMI':
                    value_str = f"{value:.1f}"
                elif feature_name == 'BloodPressure':
                    value_str = f"{value:.0f} mmHg"
                elif feature_name == 'Insulin':
                    value_str = f"{value:.0f} µU/mL"
                elif feature_name == 'Age':
                    value_str = f"{value:.0f} yrs"
                elif feature_name == 'DiabetesPedigreeFunction':
                    value_str = f"{value:.2f}"
                else:
                    value_str = f"{value:.2f}"
                
                feature_bars_html += (
                    f'<div class="shap-row">'
                    f'<div class="shap-feature-label">{display_name} = {value_str}</div>'
                    f'<div class="shap-bar-wrapper">'
                    f'<div class="shap-center-line"></div>'
                    f'<div class="shap-bar {contribution_class}" style="width: {bar_width}%"></div>'
                    f'</div>'
                    f'<div class="shap-value {contribution_class}">{"+" if is_positive else ""}{contribution:.2f}</div>'
                    f'</div>'
                )
        else:
            # Empty state — no prediction made yet
            feature_bars_html = ""
            # Will be replaced by the empty-state card below
            feature_bars_html = None

        if feature_bars_html is None:
            # No prediction yet — show clean empty state
            st.markdown("""
            <div class="content-card">
                <div class="content-card-header">
                    <div>
                        <h3 class="content-card-title">Why the Model Predicted This</h3>
                        <p class="content-card-subtitle">SHAP-based feature attributions in plain clinical English</p>
                    </div>
                    <span class="tag">SHAP + SAFE FALLBACK</span>
                </div>
                <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding:3rem 1rem; gap:1rem;">
                    <svg viewBox="0 0 24 24" width="40" height="40" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" fill="none">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                    </svg>
                    <div style="text-align:center;">
                        <div style="font-size:0.95rem; font-weight:600; color:#94a3b8; margin-bottom:0.35rem;">No Assessment Yet</div>
                        <div style="font-size:0.8rem; color:#64748b;">Submit patient data to generate feature attributions</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            shap_html = f"""<div class="content-card">
<div class="content-card-header">
<div>
<h3 class="content-card-title">Why the Model Predicted This</h3>
<p class="content-card-subtitle">SHAP-based feature attributions in plain clinical English</p>
</div>
<span class="tag">{prediction_results.get('explanation_method', 'Model explanation')}</span>
</div>
<div class="shap-plot-container">{feature_bars_html}</div>
<div class="legend">
<div class="legend-item"><div class="legend-dot red"></div><span>Pushes prediction toward diabetic</span></div>
<div class="legend-item"><div class="legend-dot green"></div><span>Pushes prediction toward non-diabetic</span></div>
</div>
</div>"""
            render_html(shap_html, unsafe_allow_html=True)

            global_rows = ""
            global_frame = prediction_results.get('global_importance')
            if global_frame is not None and not global_frame.empty:
                maximum = max(float(global_frame['Importance'].max()), 1e-9)
                for _, row in global_frame.head(8).iterrows():
                    global_rows += f"""
                    <div style="display:grid; grid-template-columns:155px 1fr 55px; gap:.8rem; align-items:center; margin:.7rem 0;">
                        <span style="color:#cbd5e1; font-size:.8rem;">{row['Feature']}</span>
                        <div class="bar-chart-track"><div class="bar-chart-fill blue-1" style="width:{float(row['Importance']) / maximum * 100:.1f}%"></div></div>
                        <span style="color:#94a3b8; font-size:.75rem; text-align:right;">{float(row['Importance']):.3f}</span>
                    </div>"""
            render_html(f"""
            <div class="content-card" style="margin-top:1rem;">
                <div class="content-card-header"><div><h3 class="content-card-title">Global Feature Importance</h3><p class="content-card-subtitle">Average impact across the active dataset</p></div><span class="tag">{prediction_results.get('explanation_method', 'Model')}</span></div>
                {global_rows}
            </div>
            """)
    
    elif selected_tab == "Analytics":
        col_a, col_b = st.columns(2, gap="medium")
        
        if prediction_results is None:
            # ── Empty state: no prediction yet ──
            with col_a:
                render_html("""
                <div class="content-card" style="height:320px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:1rem;">
                    <svg viewBox="0 0 24 24" width="36" height="36" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" fill="none">
                        <circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/>
                    </svg>
                    <div style="text-align:center;">
                        <div style="font-size:0.9rem; font-weight:600; color:#94a3b8; margin-bottom:0.3rem;">Risk Probability</div>
                        <div style="font-size:0.78rem; color:#64748b;">Submit an assessment to view the risk gauge</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                render_html("""
                <div class="content-card" style="height:320px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:1rem;">
                    <svg viewBox="0 0 24 24" width="36" height="36" stroke="rgba(255,255,255,0.15)" stroke-width="1.5" fill="none">
                        <path d="M3 3v18h18"/><path d="M18.7 8l-5.1 5.2-2.8-2.7L7 14.3"/>
                    </svg>
                    <div style="text-align:center;">
                        <div style="font-size:0.9rem; font-weight:600; color:#94a3b8; margin-bottom:0.3rem;">Feature Contribution</div>
                        <div style="font-size:0.78rem; color:#64748b;">Submit an assessment to view feature importance</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            prob_val = prediction_results['probability'] * 100
            feat_contribs = prediction_results['feature_contributions']

            with col_a:
                # Semi-circular Risk Probability Gauge
                stroke_dashoffset = 188.5 - (188.5 * (prob_val / 100.0))
                gauge_label = 'HIGH RISK BAND' if prob_val >= 70 else ('MODERATE RISK' if prob_val >= 40 else 'LOW RISK')
                render_html(f"""
                <div class="content-card" style="height: 320px;">
                    <div class="content-card-header">
                        <h3 class="content-card-title">Risk Probability</h3>
                        <p class="content-card-subtitle">Clinical risk gauge with calibrated bands</p>
                    </div>
                    
                    <div class="analytics-gauge-container">
                        <svg viewBox="0 0 160 100" width="180" height="110" class="analytics-gauge-svg">
                            <path d="M 20,80 A 60,60 0 0,1 140,80" fill="none" stroke="rgba(255,255,255,0.06)" stroke-width="12" stroke-linecap="round"/>
                            <path d="M 20,80 A 60,60 0 0,1 140,80" fill="none" stroke="url(#gaugeGrad2)" stroke-width="12" stroke-linecap="round"
                                  stroke-dasharray="188.5" stroke-dashoffset="{stroke_dashoffset:.1f}"/>
                            <defs>
                                <linearGradient id="gaugeGrad2" x1="0%" y1="0%" x2="100%" y2="0%">
                                    <stop offset="0%" stop-color="#3b82f6" />
                                    <stop offset="60%" stop-color="#f59e0b" />
                                    <stop offset="100%" stop-color="#ef4444" />
                                </linearGradient>
                            </defs>
                        </svg>
                        <div class="analytics-gauge-text-wrap">
                            <span class="analytics-gauge-val">{prob_val:.1f}%</span>
                            <span class="analytics-gauge-label">{gauge_label}</span>
                        </div>
                    </div>
                    
                    <div class="analytics-gauge-legend">
                        <span class="analytics-gauge-legend-text"><span class="analytics-gauge-legend-dot green"></span>Low &lt;30%</span>
                        <span class="analytics-gauge-legend-text"><span class="analytics-gauge-legend-dot yellow"></span>Moderate</span>
                        <span class="analytics-gauge-legend-text"><span class="analytics-gauge-legend-dot red"></span>High &gt;70%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                # Feature Contribution chart — built from real model data
                max_abs = feat_contribs['Contribution'].abs().max()
                if max_abs == 0:
                    max_abs = 1.0
                bar_colors = ['blue-1', 'blue-2', 'blue-3', 'green-1', 'yellow-1']
                bar_rows_html = ""
                for idx, (_, row) in enumerate(feat_contribs.head(5).iterrows()):
                    feat_name = row['Feature']
                    if feat_name == 'DiabetesPedigreeFunction':
                        feat_name = 'Pedigree'
                    elif feat_name == 'BloodPressure':
                        feat_name = 'Blood Press'
                    elif feat_name == 'SkinThickness':
                        feat_name = 'Skin Thick'
                    bar_w = min((abs(row['Contribution']) / max_abs) * 100, 100)
                    color_cls = bar_colors[idx % len(bar_colors)]
                    bar_rows_html += f"""
                    <div class="bar-chart-row">
                        <span class="bar-chart-label">{feat_name}</span>
                        <div class="bar-chart-track"><div class="bar-chart-fill {color_cls}" style="width:{bar_w:.1f}%;"></div></div>
                    </div>"""
                render_html(f"""
                <div class="content-card" style="height: 320px;">
                    <div class="content-card-header">
                        <h3 class="content-card-title">Feature Contribution</h3>
                        <p class="content-card-subtitle">Ranked importance for this prediction</p>
                    </div>
                    <div class="bar-chart-container">
                        {bar_rows_html}
                        <div class="bar-chart-axis">
                            <div class="bar-chart-axis-label"></div>
                            <div class="bar-chart-axis-ticks">
                                <span>0</span><span>25</span><span>50</span><span>75</span><span>100</span>
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
        col_c, col_d = st.columns(2, gap="medium")
        with col_c:
            # Prediction Confidence Distribution (Beautiful SVG Curve)
            patient_x = 10 + (prediction_results['probability'] * 180) if prediction_results else 10
            render_html(f"""
            <div class="content-card" style="height: 240px;">
                <div class="content-card-header">
                    <h3 class="content-card-title">Prediction Confidence Distribution</h3>
                    <p class="content-card-subtitle">Posterior probability across calibrated bins</p>
                </div>
                <div style="display:flex; justify-content:center; align-items:center; height:120px;">
                    <svg viewBox="0 0 200 80" width="100%" height="80" style="overflow:visible;">
                        <!-- Distribution Bell Curve -->
                        <path d="M 0,70 Q 50,70 80,45 T 130,15 T 160,65 T 200,70" fill="rgba(59, 130, 246, 0.08)" stroke="#3b82f6" stroke-width="2"/>
                        <!-- Threshold Line -->
                        <line x1="130" y1="0" x2="130" y2="70" stroke="rgba(239, 68, 68, 0.4)" stroke-dasharray="3"/>
                        <!-- Current Patient Line -->
                        <line x1="{patient_x:.1f}" y1="0" x2="{patient_x:.1f}" y2="70" stroke="#f59e0b" stroke-width="2"/>
                        <circle cx="{patient_x:.1f}" cy="30" r="4" fill="#f59e0b"/>
                        <text x="{min(patient_x + 6, 150):.1f}" y="25" fill="#f59e0b" font-size="7" font-weight="600">Current Patient {prediction_results['probability'] * 100:.1f}%</text>
                    </svg>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_d:
            # Model Performance (Hold-out validation)
            render_html(f"""
            <div class="content-card" style="height: 240px;">
                <div class="content-card-header">
                    <h3 class="content-card-title">Model Performance</h3>
                    <p class="content-card-subtitle">Hold-out validation on the current dataset</p>
                </div>
                <div style="display:flex; flex-direction:column; gap:0.5rem; justify-content:center; height:110px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:0.35rem;">
                        <span style="color:var(--text-secondary);">Accuracy</span>
                        <span style="color:var(--text-primary); font-weight:700;">{accuracy_pct:.1f}%</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; border-bottom:1px solid rgba(255,255,255,0.05); padding-bottom:0.35rem;">
                        <span style="color:var(--text-secondary);">Precision</span>
                        <span style="color:var(--text-primary); font-weight:700;">{precision_pct:.1f}%</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem;">
                        <span style="color:var(--text-secondary);">Recall</span>
                        <span style="color:var(--text-primary); font-weight:700;">{recall_pct:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        monitoring = session_metrics(st.session_state.assessment_history)
        render_html("""
        <div class="content-card" style="margin-top:1rem;">
            <div class="content-card-header"><div><h3 class="content-card-title">Session Model Monitoring</h3><p class="content-card-subtitle">Live operational analytics for this browser session</p></div><span class="tag">SESSION ONLY</span></div>
        </div>
        """)
        monitor_cols = st.columns(4)
        monitor_cols[0].metric("Assessments", monitoring['count'])
        monitor_cols[1].metric("Average risk", f"{monitoring['average_risk'] * 100:.1f}%")
        monitor_cols[2].metric("Average confidence", f"{monitoring['average_confidence'] * 100:.1f}%")
        monitor_cols[3].metric("High-risk results", monitoring['distribution']['High'])
        trend_col, distribution_col = st.columns(2)
        with trend_col:
            st.caption("Recent risk trend")
            if monitoring['trend']:
                st.line_chart(pd.DataFrame({'Risk probability': monitoring['trend']}), height=190)
            else:
                st.info("Generate assessments to populate the risk trend.")
        with distribution_col:
            st.caption("Risk category distribution")
            st.bar_chart(pd.DataFrame.from_dict(monitoring['distribution'], orient='index', columns=['Assessments']), height=190)
        if monitoring['confidences']:
            confidence_bins = pd.cut(
                pd.Series(monitoring['confidences']),
                bins=[0.49, 0.6, 0.7, 0.8, 0.9, 1.0],
                labels=['50-60%', '60-70%', '70-80%', '80-90%', '90-100%'],
            ).value_counts(sort=False)
            st.caption("Classification confidence distribution")
            st.bar_chart(confidence_bins.rename('Assessments'), height=170)

        render_html("""
        <div class="content-card" style="margin-top:1rem; margin-bottom:.6rem;">
            <div class="content-card-header"><div><h3 class="content-card-title">What-if Risk Simulator</h3><p class="content-card-subtitle">Explore how adjusted clinical inputs change the model estimate; this does not predict treatment response.</p></div><span class="tag">INTERACTIVE</span></div>
        </div>
        """)
        if prediction_results is None:
            st.info("Generate an assessment before using the simulator.")
        else:
            current = prediction_results['user_input']
            with st.form("risk_simulator"):
                sim_cols = st.columns(5)
                sim_glucose = sim_cols[0].number_input("Glucose", 0.0, 600.0, float(current['Glucose']), 1.0)
                sim_bmi = sim_cols[1].number_input("BMI", 0.0, 100.0, float(current['BMI']), 0.1)
                sim_bp = sim_cols[2].number_input("Blood Pressure", 0.0, 200.0, float(current['BloodPressure']), 1.0)
                sim_insulin = sim_cols[3].number_input("Insulin", 0.0, 1000.0, float(current['Insulin']), 1.0)
                sim_age = sim_cols[4].number_input("Age", 0, 150, int(current['Age']), 1)
                simulate = st.form_submit_button("Run What-if Simulation")
            if simulate:
                simulated_input = dict(current)
                simulated_input.update({'Glucose': sim_glucose, 'BMI': sim_bmi, 'BloodPressure': sim_bp, 'Insulin': sim_insulin, 'Age': sim_age})
                simulated_probability = predict_patient_probability(model, scaler, feature_names, simulated_input, load_data())
                st.session_state.simulation_results = {'probability': simulated_probability, 'inputs': simulated_input}
            if st.session_state.get('simulation_results'):
                simulated_probability = st.session_state.simulation_results['probability']
                difference = simulated_probability - prediction_results['probability']
                direction = "Improved" if difference < -0.01 else ("Worsened" if difference > 0.01 else "Stayed similar")
                sim_result_cols = st.columns(3)
                sim_result_cols[0].metric("Current risk", f"{prediction_results['probability'] * 100:.1f}%")
                sim_result_cols[1].metric("Simulated risk", f"{simulated_probability * 100:.1f}%", f"{difference * 100:+.1f} pp", delta_color="inverse")
                sim_result_cols[2].metric("Risk direction", direction)
            
    elif selected_tab == "Recommendations":
        if prediction_results is None:
            render_html("""
            <div class="content-card" style="text-align:center; padding:3rem; color:#64748b;">
                Generate an AI assessment to receive patient-specific clinical recommendations.
            </div>
            """)
        else:
            render_html(f"""
            <div class="content-card" style="margin-bottom:1rem; border-left:3px solid var(--accent-blue);">
                <div class="content-card-header"><div><h3 class="content-card-title">AI Clinical Summary</h3><p class="content-card-subtitle">Locally generated clinical narrative; no external API or patient-data transmission</p></div><span class="tag">LOCAL AI</span></div>
                <p style="color:#cbd5e1; line-height:1.7; margin:0;">{prediction_results['clinical_summary']}</p>
            </div>
            """)
            recommendation_cards = prediction_results.get('recommendations') or generate_recommendations(
                prediction_results['probability'], prediction_results['user_input'])
            cards_html = ""
            for recommendation in recommendation_cards:
                tag = recommendation['category']
                title = recommendation['title']
                description = recommendation['detail']
                color_class = f"{recommendation['tone']}-bg-light"
                cards_html += f"""
                <div class="recommendation-card-custom">
                    <div class="recom-icon-wrap {color_class}">
                        <svg viewBox="0 0 24 24" width="18" height="18" stroke="currentColor" stroke-width="2" fill="none"><circle cx="12" cy="12" r="10"></circle><path d="M12 8v4M12 16h.01"></path></svg>
                    </div>
                    <div class="recom-content">
                        <span class="recom-tag">{tag}</span>
                        <span class="recom-title">{title}</span>
                        <span class="recom-desc">{description}</span>
                    </div>
                </div>"""
            render_html(f"""
        <div class="content-card">
            <div class="content-card-header">
                <div>
                    <h3 class="content-card-title">Clinical Recommendations</h3>
                    <p class="content-card-subtitle">Evidence-based guidance generated for this assessment</p>
                </div>
            </div>
            
            <div class="recommendations-grid-2x2">
                {cards_html}
            </div>
        </div>
            """, unsafe_allow_html=True)

    elif selected_tab == "Model Insights":
        col_e, col_f = st.columns(2, gap="large")
        with col_e:
            render_html("""
            <div class="content-card" style="margin-bottom:0;">
                <div class="content-card-header">
                    <div>
                        <h3 class="content-card-title">Validation Metrics</h3>
                        <p class="content-card-subtitle">ROC curve - hold-out validation</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Renders Plotly ROC curve
            render_roc_curve(roc_points, auc_val)
            
        with col_f:
            # Training dataset stats card
            render_html(f"""
            <div class="content-card">
                <div class="content-card-header">
                    <h3 class="content-card-title">TRAINING DATASET</h3>
                </div>
                <p style="color:var(--text-primary); font-size:1.4rem; font-weight:700; margin-bottom:0.15rem;">{row_count} patients</p>
                <p style="color:var(--text-muted); font-size:0.75rem; font-weight:500;">Local diabetes dataset &bull; {len(feature_names)} features &bull; stratified split</p>
                
                <div class="dataset-box-grid">
                    <div class="dataset-box">
                        <div class="dataset-box-val">{train_size}</div>
                        <div class="dataset-box-lbl">Train</div>
                    </div>
                    <div class="dataset-box">
                        <div class="dataset-box-val">{val_size}</div>
                        <div class="dataset-box-lbl">Val</div>
                    </div>
                    <div class="dataset-box">
                        <div class="dataset-box-val">{test_size}</div>
                        <div class="dataset-box-lbl">Test</div>
                    </div>
                </div>
            </div>
            
            <div class="content-card" style="margin-top: 1rem;">
                <div class="content-card-header">
                    <h3 class="content-card-title">MODEL</h3>
                </div>
                <p style="color:var(--text-primary); font-size:1.15rem; font-weight:700; margin-bottom:0.15rem;">{model_type}</p>
                <p style="color:var(--text-muted); font-size:0.75rem; font-weight:500; margin-bottom:0.5rem;">Trained from the current local dataset</p>
                
                <div class="model-metrics-grid">
                    <div class="model-metric-item">
                        <span class="model-metric-lbl">AUC</span>
                        <span class="model-metric-val">{auc_val:.3f}</span>
                    </div>
                    <div class="model-metric-item">
                        <span class="model-metric-lbl">F1</span>
                        <span class="model-metric-val">{f1_val:.3f}</span>
                    </div>
                    <div class="model-metric-item">
                        <span class="model-metric-lbl">Brier</span>
                        <span class="model-metric-val">{brier_val:.3f}</span>
                    </div>
                    <div class="model-metric-item">
                        <span class="model-metric-lbl">Accuracy</span>
                        <span class="model-metric-val">{accuracy_pct:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        render_html("""
        <div class="content-card" style="margin-top:1rem; margin-bottom:.5rem;">
            <div class="content-card-header"><div><h3 class="content-card-title">Classical Model Benchmark</h3><p class="content-card-subtitle">Logistic Regression, Random Forest, Gradient Boosting, and SVM on the same deterministic 80/20 split</p></div><span class="tag">CACHED</span></div>
        </div>
        """)
        benchmark_frame = get_benchmark_results(model_metadata.get('dataset_sha256', 'active-dataset'))
        error_column = benchmark_frame.get('Error', pd.Series(index=benchmark_frame.index, dtype=object))
        successful_benchmarks = benchmark_frame[error_column.isna()].copy()
        if not successful_benchmarks.empty:
            metric_columns = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
            display_benchmark = successful_benchmarks[['Model'] + metric_columns].copy()
            for column in metric_columns:
                display_benchmark[column] = display_benchmark[column].map(lambda value: f"{value * 100:.1f}%")
            st.dataframe(display_benchmark, width='stretch', hide_index=True)
        failed_benchmarks = benchmark_frame[error_column.notna()]
        if not failed_benchmarks.empty:
            st.warning("Some benchmark models were unavailable: " + ", ".join(failed_benchmarks['Model']))
            
    elif selected_tab == "History":
        # History table listing assessments with search capability
        render_history_tab()
        
    elif selected_tab == "Documentation":
        render_html(f"""
        <div class="content-card">
            <div class="content-card-header">
                <div>
                    <h3 class="content-card-title">Documentation</h3>
                    <p class="content-card-subtitle">Model, dataset, and clinical-use notes</p>
                </div>
            </div>
            <div style="display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:1rem;">
                <div class="recommendation-card-custom">
                    <div class="recom-content">
                        <span class="recom-tag">Dataset</span>
                        <span class="recom-title">{row_count} patient records loaded</span>
                        <span class="recom-desc">The active model is trained from the current local diabetes dataset with {len(feature_names)} clinical features.</span>
                    </div>
                </div>
                <div class="recommendation-card-custom">
                    <div class="recom-content">
                        <span class="recom-tag">Model</span>
                        <span class="recom-title">{model_type}</span>
                        <span class="recom-desc">Hold-out validation: AUC {auc_val:.3f}, accuracy {accuracy_pct:.1f}%, precision {precision_pct:.1f}%, recall {recall_pct:.1f}%.</span>
                    </div>
                </div>
                <div class="recommendation-card-custom">
                    <div class="recom-content">
                        <span class="recom-tag">Inputs</span>
                        <span class="recom-title">Eight PIMA-style features</span>
                        <span class="recom-desc">Pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, diabetes pedigree function, and age.</span>
                    </div>
                </div>
                <div class="recommendation-card-custom">
                    <div class="recom-content">
                        <span class="recom-tag">Use</span>
                        <span class="recom-title">Clinical decision support</span>
                        <span class="recom-desc">Predictions must be interpreted alongside diagnostic testing and clinician judgment.</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # Footer logo & badges
    render_html(f"""
    <div class="app-footer">
        <div style="display:flex; align-items:center; gap:0.75rem;">
            <img src="data:image/png;base64,{base64.b64encode((ASSET_DIR / 'glucosight_logo.png').read_bytes()).decode()}" alt="GlucoSight AI" style="width: 28px; height: 28px; object-fit: contain;"/>
            <div class="footer-text">&copy; 2026 GlucoSight AI &bull; For clinical decision support only &bull; Not for diagnostic use</div>
        </div>
        <div class="footer-badges">
            <span class="footer-badge">{model_type}</span>
            <span class="footer-badge">HIPAA-aligned</span>
            <span class="footer-badge">HL7 FHIR R4</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
