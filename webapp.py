"""
inqtrix Chat Interface
======================================================
Start with: streamlit run webapp.py
Requires:   pip install streamlit
"""

import base64
from html import escape

import streamlit as st

from inqtrix_webapp.client import (
    call_chat,
    fetch_health,
    fetch_models_fallback,
    fetch_stacks,
    get_api_key,
    get_base_url,
    stream_chat,
)

# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="inqtrix",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS injection
st.markdown("""
<style>
    :root {
        --bg: #1f1d1b;
        --panel: rgba(49, 45, 42, 0.88);
        --panel-soft: rgba(49, 45, 42, 0.72);
        --panel-strong: rgba(39, 36, 34, 0.96);
        --popover-panel: rgba(43, 40, 37, 0.985);
        --line: rgba(244, 236, 225, 0.08);
        --line-strong: rgba(244, 236, 225, 0.14);
        --text: #ece6db;
        --muted: #a59f94;
        --accent: #d97757;
        --accent-soft: rgba(217, 119, 87, 0.16);
        --pill: rgba(42, 39, 36, 0.94);
        --pill-hover: rgba(58, 53, 49, 0.98);
    }

    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(217, 119, 87, 0.14), transparent 28%),
            radial-gradient(circle at top right, rgba(239, 226, 195, 0.08), transparent 24%),
            linear-gradient(180deg, #262320 0%, #1f1d1b 42%, #1a1918 100%) !important;
        color: var(--text) !important;
    }

    [data-testid="stHeader"],
    footer {
        display: none !important;
    }

    .block-container {
        max-width: 900px;
        padding-top: 1.1rem !important;
        padding-bottom: max(1rem, calc(env(safe-area-inset-bottom) + 1rem)) !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stBottomBlock"] {
        background: transparent !important;
    }

    body,
    button,
    input,
    textarea,
    select,
    [data-testid="stMarkdownContainer"],
    [data-testid="stChatMessage"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6,
    .brand-header,
    .welcome-title {
        font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif !important;
        font-weight: 400 !important;
        letter-spacing: -0.02em;
        color: #f3eee5 !important;
    }

    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        color: var(--text);
        line-height: 1.7;
    }

    .brand-kicker,
    .composer-meta,
    .message-label {
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
    }

    .brand-kicker {
        margin-bottom: 0.2rem;
    }

    .prototype-notice {
        display: inline-block;
        padding: 0.5rem 1.15rem;
        margin-bottom: 1.25rem;
        border: 1px solid rgba(235, 84, 70, 0.42);
        background: rgba(235, 84, 70, 0.08);
        border-radius: 14px;
        color: #ef7266;
        font-size: 0.72rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 500;
        backdrop-filter: blur(8px) saturate(140%);
        -webkit-backdrop-filter: blur(8px) saturate(140%);
        box-shadow: 0 2px 10px rgba(235, 84, 70, 0.08);
    }

    .composer-meta .prototype-inline {
        color: #f28b7f;
        font-weight: 600;
    }

    .composer-meta .meta-right {
        margin-left: auto;
    }

    .composer-prototype-pill {
        display: inline-block;
        padding: 0.26rem 0.75rem;
        border: 1px solid rgba(235, 84, 70, 0.42);
        background: rgba(235, 84, 70, 0.08);
        border-radius: 999px;
        color: #ef7266;
        font-size: 0.62rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 500;
        line-height: 1;
        backdrop-filter: blur(8px) saturate(140%);
        -webkit-backdrop-filter: blur(8px) saturate(140%);
        box-shadow: 0 2px 10px rgba(235, 84, 70, 0.08);
    }

    .brand-header {
        font-size: 1.46rem;
        line-height: 1.1;
    }

    .st-key-page_shell {
        min-height: calc(100vh - 92px - 170px);
        display: flex;
        flex-direction: column;
    }

    .st-key-app_header {
        position: fixed;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 920px;
        z-index: 45;
        padding: 0.85rem 1.4rem 1.35rem;
        background: rgba(31, 29, 27, 0.72);
        backdrop-filter: blur(22px) saturate(160%);
        -webkit-backdrop-filter: blur(22px) saturate(160%);
        -webkit-mask-image: linear-gradient(to bottom, black 0%, black 72%, transparent 100%);
        mask-image: linear-gradient(to bottom, black 0%, black 72%, transparent 100%);
    }

    .st-key-app_header [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 0.5rem !important;
    }

    .st-key-conversation_shell {
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
    }

    .st-key-empty_state_shell {
        flex: 1 1 auto;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .st-key-page_shell > [data-testid="stLayoutWrapper"]:has(> .st-key-conversation_shell) {
        flex: 1 1 auto !important;
        min-height: 0 !important;
    }

    .st-key-conversation_shell > [data-testid="stLayoutWrapper"]:has(> .st-key-empty_state_shell) {
        flex: 1 1 auto !important;
        min-height: 0 !important;
    }

    [data-testid="stElementContainer"]:has(> [data-testid="stIFrame"]) {
        position: absolute !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        pointer-events: none !important;
        overflow: hidden !important;
    }

    .user-msg-anchor {
        display: block;
        height: 0;
        scroll-margin-top: 96px;
        pointer-events: none;
    }

    [data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] .user-msg-anchor) {
        margin: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
    }

    .st-key-conversation_shell:has(.user-msg-anchor)::after {
        content: "";
        display: block;
        flex: 0 0 auto;
        min-height: calc(100vh - 92px - 170px - 96px);
        pointer-events: none;
    }

    .welcome-wrap {
        text-align: center;
        padding: 0.2rem 0 0.65rem;
    }

    .welcome-meta {
        font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
        font-size: 0.78rem;
        font-style: italic;
        color: #c9a961;
        opacity: 0.82;
        margin: 0 0 1.35rem;
        letter-spacing: 0.01em;
    }

    .welcome-meta a {
        color: #e3c37a;
        text-decoration: none;
        border-bottom: 1px solid rgba(227, 195, 122, 0.22);
        transition: color 0.15s ease, border-color 0.15s ease;
        padding-bottom: 1px;
    }

    .welcome-meta a:hover {
        color: #f3d68a;
        border-bottom-color: rgba(243, 214, 138, 0.55);
    }

    .welcome-meta .sep {
        margin: 0 0.55rem;
        opacity: 0.5;
    }

    .welcome-title {
        font-size: clamp(2.35rem, 5.2vw, 3.35rem);
        margin-bottom: 0.22rem;
    }

    .welcome-sub {
        font-size: 0.94rem;
        color: var(--muted);
        max-width: 30rem;
        margin: 0 auto 0.8rem;
    }

    .st-key-suggestion_shell {
        margin-bottom: 0.18rem;
    }

    [data-testid="stPills"] > div {
        gap: 0.45rem;
        justify-content: center;
    }

    [data-testid="stPills"] button {
        background: rgba(28, 30, 38, 0.82) !important;
        border: 1px solid rgba(119, 129, 156, 0.24) !important;
        border-radius: 999px !important;
        color: var(--text) !important;
        padding: 0.2rem 0.66rem !important;
        font-size: 0.82rem !important;
    }

    [data-testid="stPills"] button:hover,
    [data-testid="stButton"] button:hover {
        border-color: rgba(217, 119, 87, 0.38) !important;
        background: var(--pill-hover) !important;
        color: #fff4ed !important;
    }

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding: 0.1rem 0 0.45rem !important;
        gap: 0.55rem !important;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] {
        background: var(--panel-soft) !important;
        border: 1px solid var(--line) !important;
        border-radius: 16px !important;
        padding: 1.2rem 1.2rem !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
        font-size: 0.88rem !important;
        line-height: 1.55 !important;
        position: relative !important;
    }

    .copy-btn {
        position: absolute;
        top: 0.55rem;
        right: 0.55rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 28px;
        height: 28px;
        padding: 0;
        border: 1px solid transparent;
        border-radius: 8px;
        background: transparent;
        color: var(--muted);
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.15s ease, background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
        z-index: 4;
    }

    [data-testid="stChatMessage"]:hover .copy-btn,
    .copy-btn:focus-visible,
    .copy-btn.copied {
        opacity: 1;
    }

    .copy-btn:hover {
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(255, 255, 255, 0.08);
        color: var(--text);
    }

    .copy-btn .check-icon {
        display: none;
    }

    .copy-btn.copied {
        color: #6cc276;
        border-color: rgba(108, 194, 118, 0.3);
        background: rgba(108, 194, 118, 0.08);
    }

    .copy-btn.copied .copy-icon {
        display: none;
    }

    .copy-btn.copied .check-icon {
        display: inline-block;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"],
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stElementContainer"],
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stMarkdown"],
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stMarkdown"] > div,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stMarkdownContainer"] {
        display: contents !important;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stMarkdown"]:first-child p:first-child,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stMarkdown"]:first-child h1:first-child,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stMarkdown"]:first-child h2:first-child,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stMarkdown"]:first-child h3:first-child,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] > div > div > [data-testid="stElementContainer"]:first-child p:first-child {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] li {
        font-size: 0.88rem !important;
        line-height: 1.55 !important;
        margin-bottom: 0.5rem !important;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] > div > div > [data-testid="stElementContainer"]:last-child,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stElementContainer"]:last-child,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] [data-testid="stVerticalBlock"]:last-child,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] p:last-child,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] li:last-child,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] ol:last-child,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] ul:last-child {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] h1,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] h2,
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"] h3 {
        margin-top: 0.9rem !important;
        margin-bottom: 0.55rem !important;
        padding: 0 !important;
    }

    [data-testid="stChatMessageAvatar"] {
        background: transparent !important;
        color: var(--accent) !important;
        font-size: 1rem !important;
        margin-top: 0.2rem;
        width: 1.85rem !important;
        height: 1.85rem !important;
    }

    [data-testid="stSelectbox"] label p,
    [data-testid="stNumberInput"] label p,
    [data-testid="stSlider"] label p,
    [data-testid="stToggle"] label p,
    [data-testid="stRadio"] label p {
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    div[data-baseweb="select"] > div {
        background: var(--panel) !important;
        border: 1px solid var(--line) !important;
        border-radius: 18px !important;
        min-height: 2.72rem !important;
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.1);
    }

    div[data-baseweb="select"] span {
        color: var(--text) !important;
    }

    [data-testid="stButton"] button {
        background: rgba(49, 45, 42, 0.88) !important;
        color: var(--text) !important;
        border: 1px solid var(--line) !important;
        min-height: 2.15rem;
        border-radius: 999px !important;
    }

    .st-key-new_chat button,
    .st-key-delete_chat button {
        background: var(--pill) !important;
        border: 1px solid var(--line) !important;
        border-radius: 999px !important;
        color: var(--text) !important;
        min-height: 2.25rem !important;
        height: 2.25rem !important;
        padding: 0 0.95rem !important;
        gap: 0.5rem !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.04em !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18);
        transition: border-color .15s ease, background .15s ease;
    }

    .st-key-new_chat button:hover,
    .st-key-delete_chat button:hover {
        background: var(--pill-hover) !important;
        border-color: rgba(217, 119, 87, 0.38) !important;
        color: #fff4ed !important;
    }

    .st-key-new_chat button [data-testid="stIconMaterial"],
    .st-key-delete_chat button [data-testid="stIconMaterial"] {
        font-size: 1.08rem !important;
        line-height: 1 !important;
        color: var(--accent) !important;
    }

    .st-key-new_chat button p,
    .st-key-delete_chat button p {
        font-size: 0.78rem !important;
        line-height: 1 !important;
        margin: 0 !important;
        letter-spacing: 0.04em !important;
    }

    @media (max-width: 620px) {
        .st-key-new_chat button p,
        .st-key-delete_chat button p {
            display: none !important;
        }
        .st-key-new_chat button,
        .st-key-delete_chat button {
            padding: 0 0.7rem !important;
        }
    }

    .st-key-composer_shell {
        position: fixed;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100%;
        max-width: 860px;
        z-index: 40;
        padding: 1.1rem 1.2rem calc(0.9rem + env(safe-area-inset-bottom));
        background: rgba(31, 29, 27, 0.72);
        backdrop-filter: blur(22px) saturate(160%);
        -webkit-backdrop-filter: blur(22px) saturate(160%);
        -webkit-mask-image: linear-gradient(to top, black 0%, black 94%, transparent 100%);
        mask-image: linear-gradient(to top, black 0%, black 94%, transparent 100%);
    }

    [data-testid="stMainBlockContainer"],
    [data-testid="stAppViewBlockContainer"],
    section.main > .block-container,
    [data-testid="stMain"] > .block-container,
    .block-container {
        max-width: 860px !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding: 92px 1.2rem 170px !important;
        height: 100vh !important;
        max-height: 100vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        overscroll-behavior: contain;
        scroll-behavior: smooth;
    }

    [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"],
    [data-testid="stMain"] > .block-container > [data-testid="stVerticalBlock"],
    section.main > .block-container > [data-testid="stVerticalBlock"],
    .block-container > [data-testid="stVerticalBlock"] {
        gap: 0 !important;
    }

    .st-key-composer_status,
    .st-key-composer_input,
    .st-key-composer_footer {
        max-width: 720px;
        margin: 0 auto;
    }

    .st-key-composer_status {
        padding: 0 0 0.42rem;
        min-height: 1.9rem;
    }

    .st-key-composer_status [data-testid="stHorizontalBlock"] {
        align-items: center !important;
        gap: 0.4rem !important;
    }

    .st-key-composer_status .composer-meta {
        font-size: 0.68rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #c4bcae;
        line-height: 1;
        padding: 0;
    }

    .composer-meta-spacer {
        min-height: 1.2rem;
    }

    .st-key-composer_footer .st-key-composer_settings_detail_popover {
        margin-left: 0.1rem !important;
    }

    .st-key-composer_settings_detail_popover [data-testid="stPopoverButton"] {
        min-width: 0 !important;
        padding: 0.32rem 0.55rem !important;
        height: auto !important;
        background: rgba(236, 230, 219, 0.04) !important;
        border: 1px solid rgba(244, 236, 225, 0.1) !important;
        border-radius: 12px !important;
        color: var(--muted) !important;
        opacity: 0.9;
        transition: opacity .12s ease, border-color .12s ease, color .12s ease, background .12s ease;
    }

    .st-key-composer_settings_detail_popover [data-testid="stPopoverButton"] p {
        display: none !important;
    }

    .st-key-composer_settings_detail_popover [data-testid="stPopoverButton"]:hover {
        opacity: 1;
        border-color: rgba(244, 236, 225, 0.22) !important;
        color: var(--text) !important;
        background: rgba(236, 230, 219, 0.08) !important;
    }

    .st-key-composer_settings_detail_popover [data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {
        font-size: 1.05rem !important;
        color: inherit !important;
    }

    .settings-detail-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.22rem 0;
        font-size: 0.82rem;
        color: var(--text);
        border-bottom: 1px solid rgba(244, 236, 225, 0.04);
    }

    .settings-detail-row:last-child {
        border-bottom: none;
    }

    .settings-detail-bullet {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
        background: rgba(244, 236, 225, 0.22);
        flex-shrink: 0;
    }

    .settings-detail-label {
        color: var(--muted);
        flex: 0 0 6.5rem;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
    }

    .settings-detail-value {
        color: var(--text);
        flex: 1 1 auto;
        font-weight: 400;
    }

    .settings-detail-default {
        font-size: 0.62rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        background: rgba(244, 236, 225, 0.06);
        border: 1px solid rgba(244, 236, 225, 0.08);
        border-radius: 999px;
        padding: 0.05rem 0.45rem;
        opacity: 0.72;
    }

    .st-key-composer_footer {
        margin-top: 0.45rem;
    }

    .st-key-composer_footer > div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        align-items: center !important;
        gap: 0.45rem !important;
    }

    .st-key-composer_footer [data-testid="stToggle"] {
        margin: 0 !important;
    }

    .st-key-composer_footer [data-testid="stToggle"] label {
        gap: 0.5rem !important;
    }

    .st-key-composer_footer [data-testid="stToggle"] label p {
        font-size: 0.72rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    .st-key-composer_footer [data-testid="stPopoverButton"] {
        background: var(--pill) !important;
        border: 1px solid var(--line) !important;
        border-radius: 999px !important;
        color: var(--text) !important;
        min-height: 2.25rem !important;
        height: 2.25rem !important;
        width: auto !important;
        min-width: 0 !important;
        padding: 0 0.95rem !important;
        gap: 0.5rem !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.04em !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.18);
        transition: border-color .15s ease, background .15s ease;
    }

    .st-key-composer_footer [data-testid="stPopoverButton"]:hover {
        background: var(--pill-hover) !important;
        border-color: rgba(217, 119, 87, 0.38) !important;
        color: #fff4ed !important;
    }

    .st-key-composer_footer [data-testid="stPopoverButton"] p {
        font-size: 0.78rem !important;
        line-height: 1 !important;
        margin: 0 !important;
        letter-spacing: 0.04em !important;
    }

    .st-key-composer_footer [data-testid="stPopoverButton"] [data-testid="stIconMaterial"] {
        font-size: 1.08rem !important;
        line-height: 1 !important;
        color: var(--accent) !important;
    }

    @media (max-width: 620px) {
        .st-key-composer_footer [data-testid="stPopoverButton"] p {
            display: none !important;
        }
        .st-key-composer_footer [data-testid="stPopoverButton"] {
            padding: 0 0.7rem !important;
        }
    }

    .st-key-composer_input [data-testid="stChatInput"] {
        max-width: none !important;
        margin: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
    }

    .st-key-composer_input [data-testid="stChatInput"] > div {
        border-radius: 22px !important;
    }

    .st-key-composer_input [data-testid="stChatInput"] [data-baseweb="textarea"] {
        min-height: auto !important;
        border-radius: 22px !important;
    }

    .st-key-composer_input [data-testid="stChatInputTextArea"],
    .st-key-composer_input [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
        min-height: unset !important;
        padding-top: 0.18rem !important;
        padding-bottom: 0.18rem !important;
        font-size: 0.95rem !important;
        line-height: 1.35 !important;
    }

    .st-key-composer_input [data-testid="stChatInputTextArea"]::placeholder,
    .st-key-composer_input [data-testid="stChatInput"] textarea::placeholder {
        color: var(--muted) !important;
    }

    .st-key-composer_input [data-testid="stChatInputFileUploadButton"],
    .st-key-composer_input [data-testid="stChatInputFileUploadButton"] button,
    .st-key-composer_input [data-testid="stChatInputSubmitButton"],
    .st-key-composer_input [data-testid="stChatInputSubmitButton"] button {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        color: var(--text) !important;
        border-radius: 999px !important;
        box-shadow: none !important;
    }

    .st-key-composer_input [data-testid="stChatInputFileUploadButton"]:hover,
    .st-key-composer_input [data-testid="stChatInputFileUploadButton"] button:hover,
    .st-key-composer_input [data-testid="stChatInputSubmitButton"]:hover,
    .st-key-composer_input [data-testid="stChatInputSubmitButton"] button:hover {
        background: rgba(217, 119, 87, 0.16) !important;
        border-color: rgba(217, 119, 87, 0.28) !important;
    }

    .composer-meta {
        display: flex;
        align-items: center;
        text-align: left;
        padding-top: 0.08rem;
        font-size: 0.61rem;
        line-height: 1.25;
        letter-spacing: 0.11em;
    }

    [data-testid="stPopoverBody"] {
        background: var(--popover-panel) !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 16px !important;
        box-shadow: 0 18px 36px rgba(0, 0, 0, 0.32) !important;
        padding: 1rem 1.1rem 1.05rem !important;
        min-width: 240px !important;
        max-height: 82vh !important;
        overflow-y: auto !important;
        overscroll-behavior: contain;
        scrollbar-width: thin;
        scrollbar-color: rgba(244, 236, 225, 0.22) transparent;
    }

    .popover-explainer {
        font-size: 0.78rem !important;
        line-height: 1.45 !important;
        color: var(--muted) !important;
        margin: 0 0 0.7rem !important;
        padding: 0.55rem 0.7rem !important;
        border-radius: 10px !important;
        background: rgba(236, 230, 219, 0.04);
        border: 1px solid rgba(244, 236, 225, 0.06);
    }

    .popover-explainer strong {
        color: var(--text) !important;
        font-weight: 500 !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stSlider"] {
        margin-top: 0.35rem !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stSlider"] label {
        display: block !important;
        font-size: 0.78rem !important;
        color: var(--muted) !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 0.15rem !important;
    }

    .stack-health-row {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.22rem 0;
        font-size: 0.83rem;
        color: var(--text);
    }

    .stack-health-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }

    .stack-health-dot.ready {
        background: #9bba7a;
        box-shadow: 0 0 0 2px rgba(155, 186, 122, 0.18);
    }

    .stack-health-dot.down {
        background: transparent;
        border: 1px solid rgba(235, 84, 70, 0.55);
        box-shadow: 0 0 0 2px rgba(235, 84, 70, 0.08);
    }

    .stack-health-label {
        color: var(--muted);
        font-size: 0.78rem;
    }

    .stack-health-name {
        color: var(--text);
        font-weight: 400;
        letter-spacing: 0.01em;
    }

    [data-testid="stPopoverBody"]::-webkit-scrollbar {
        width: 8px;
    }

    [data-testid="stPopoverBody"]::-webkit-scrollbar-thumb {
        background: rgba(244, 236, 225, 0.18);
        border-radius: 8px;
    }

    [data-testid="stPopoverBody"]::-webkit-scrollbar-track {
        background: transparent;
    }

    .st-key-stopp_row {
        margin: 0 0 0.95rem !important;
    }

    /* Guarantee breathing room between the Stopp button and the
       following status widget (st.status renders as
       <details data-testid="stStatusWidget"> or stVerticalBlock wrapping a
       stExpander). Streamlit's default margin collapses them. */
    .st-key-stopp_row + [data-testid="stStatusWidget"],
    .st-key-stopp_row + [data-testid="stExpander"],
    .st-key-stopp_row + [data-testid="stVerticalBlockBorderWrapper"] {
        margin-top: 0.65rem !important;
    }

    .st-key-stopp_row [data-testid="stButton"] > button {
        background: rgba(235, 84, 70, 0.1) !important;
        border: 1px solid rgba(235, 84, 70, 0.38) !important;
        color: #ef7266 !important;
        border-radius: 12px !important;
        padding: 0.45rem 0.95rem !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.01em !important;
        font-weight: 500 !important;
        box-shadow: none !important;
        transition: background .15s ease, border-color .15s ease, color .15s ease;
    }

    .st-key-stopp_row [data-testid="stButton"] > button:hover {
        background: rgba(235, 84, 70, 0.22) !important;
        border-color: rgba(235, 84, 70, 0.65) !important;
        color: #f28b7f !important;
    }

    .st-key-stopp_row [data-testid="stButton"] > button [data-testid="stIconMaterial"] {
        color: #ef7266 !important;
    }

    /* Persisted progress-log expander inside completed assistant messages. */
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary {
        font-size: 0.78rem !important;
        color: var(--muted) !important;
        letter-spacing: 0.02em !important;
    }

    [data-testid="stChatMessage"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] p code,
    [data-testid="stChatMessage"] [data-testid="stExpander"] [data-testid="stExpanderDetails"] code {
        background: rgba(236, 230, 219, 0.04) !important;
        border: 1px solid rgba(244, 236, 225, 0.06) !important;
        color: var(--muted) !important;
        padding: 0.1rem 0.4rem !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.02em !important;
    }

    [data-testid="stPopoverBody"] > div,
    [data-testid="stPopoverBody"] > div > div,
    [data-testid="stPopoverBody"] [data-testid="stVerticalBlock"],
    [data-testid="stPopoverBody"] [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stPopoverBody"] [data-testid="stHorizontalBlock"],
    [data-testid="stPopoverBody"] [data-testid="stElementContainer"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] {
        font-size: 0.82rem !important;
        letter-spacing: 0.02em !important;
        text-transform: none !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        margin: 0 0 0.5rem !important;
        padding: 0 !important;
        opacity: 1 !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stCaptionContainer"] p {
        font-size: 0.82rem !important;
        letter-spacing: 0.02em !important;
        text-transform: none !important;
        color: var(--text) !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
        margin: 0 !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stRadio"] + [data-testid="stCaptionContainer"],
    [data-testid="stPopoverBody"] [data-testid="stSelectbox"] + [data-testid="stCaptionContainer"] {
        margin-top: 0.95rem !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stSelectbox"],
    [data-testid="stPopoverBody"] [data-testid="stRadio"] {
        margin-bottom: 0.1rem !important;
    }

    [data-testid="stPopoverBody"] [data-testid="stSelectbox"] > label,
    [data-testid="stPopoverBody"] [data-testid="stRadio"] > label {
        display: none !important;
    }

    [data-testid="stPopoverBody"] div[data-baseweb="select"] > div {
        min-height: 2.2rem !important;
        border-radius: 12px !important;
        box-shadow: none !important;
        background: var(--panel) !important;
    }

    [data-testid="stPopoverBody"] [role="radiogroup"] {
        gap: 0.12rem !important;
    }

    [data-testid="stPopoverBody"] [role="radiogroup"] label {
        padding: 0.28rem 0.35rem !important;
        border-radius: 8px !important;
        transition: background .12s ease;
        gap: 0.6rem !important;
    }

    [data-testid="stPopoverBody"] [role="radiogroup"] label:hover {
        background: rgba(236, 230, 219, 0.05) !important;
    }

    [data-testid="stPopoverBody"] [role="radiogroup"] label > div:first-child {
        display: flex !important;
    }

    [data-testid="stPopoverBody"] [role="radiogroup"] p {
        font-size: 0.88rem !important;
        line-height: 1.25 !important;
        letter-spacing: 0.01em !important;
        color: var(--text) !important;
        text-transform: none !important;
        font-weight: 400 !important;
    }

    .message-label {
        margin-bottom: 0.6rem;
    }

    .provider-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        font-size: 0.69rem;
        padding: 0.32rem 0.58rem;
        border-radius: 999px;
        font-weight: 600;
        margin-bottom: 0.65rem;
        background: rgba(255, 255, 255, 0.04);
        color: var(--muted);
        border: 1px solid var(--line);
    }

    .file-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        margin-top: 0.7rem;
    }

    .file-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.26rem 0.56rem;
        border-radius: 999px;
        background: rgba(217, 119, 87, 0.12);
        border: 1px solid rgba(217, 119, 87, 0.18);
        color: #f7d8cb;
        font-size: 0.72rem;
    }

    .citation-box {
        font-size: 0.82rem;
        padding: 0.85rem 0.95rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        margin-top: 1rem;
        color: var(--muted);
        border: 1px solid var(--line);
    }

    .citation-box a {
        color: #efb39d;
        text-decoration: none;
        font-weight: 600;
    }

    .citation-box a:hover {
        text-decoration: underline;
    }

    [data-testid="stSidebar"] {
        background: #24211f !important;
        border-right: 1px solid var(--line) !important;
    }

    @media (max-width: 900px) {
        .block-container {
            padding-top: 0.9rem !important;
        }

        .st-key-empty_state_shell {
            min-height: clamp(15rem, calc(100vh - 21rem), 22rem);
        }

        .st-key-composer_shell {
            padding-top: 0.42rem;
        }

        [data-testid="stMainBlockContainer"] {
            padding-top: 108px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Constants — UI labels + agent_overrides whitelist mappings
# ---------------------------------------------------------------------------

# Fallback avatar / icon used until the server has been discovered or when
# a stack has no operator-friendly description. Chat messages only need
# an icon for st.chat_message; nothing here ever hits the HTTP wire.
DEFAULT_STACK_AVATAR = ":material/auto_awesome:"
DEFAULT_STACK_ICON = "✦"
STACK_AVATARS = {
    "litellm_perplexity": ":material/hub:",
    "anthropic_perplexity": ":material/auto_awesome:",
    "bedrock_perplexity": ":material/memory:",
    "azure_openai_perplexity": ":material/cloud:",
    "azure_openai_web_search": ":material/travel_explore:",
    "azure_openai_bing": ":material/search:",
    "azure_foundry_web_search": ":material/cloud:",
}

# report_profile is the primary "Recherche-Modus" lever on the agent.
# Profile defaults fan out across other settings via
# AgentSettings.with_report_profile_defaults (see src/inqtrix/settings.py).
REPORT_PROFILES = ["compact", "deep"]
REPORT_PROFILE_LABELS = {
    "compact": "Kompakt",
    "deep": "Deep Research",
}

# Effort levels map the UI label onto (max_rounds, min_rounds) pairs from
# the per-request overrides whitelist (src/inqtrix/server/overrides.py).
# "Auto" is the only level that omits both fields — the server then
# applies the AgentSettings defaults (which themselves fan out from
# report_profile).
EFFORT_LEVELS = ["Auto", "Niedrig", "Mittel", "Hoch", "Max"]
EFFORT_ROUNDS: dict[str, tuple[int, int] | None] = {
    "Auto": None,
    "Niedrig": (1, 1),
    "Mittel": (2, 1),
    "Hoch": (4, 2),
    "Max": (6, 3),
}
EFFORT_HELP = (
    "Steuert den Recherche-Aufwand des Agenten über `max_rounds` (obere "
    "Schleifen-Schranke) und `min_rounds` (untere Schranke).\n\n"
    "• **Auto** — Server-Default (ergibt sich aus dem Profil).\n"
    "• **Niedrig** — max=1, min=1 (einziger Durchlauf, schnellste Antwort).\n"
    "• **Mittel** — max=2, min=1.\n"
    "• **Hoch** — max=4, min=2 (solide Standardwahl für Deep).\n"
    "• **Max** — max=6, min=3 (maximal tief, langsamste Antwort)."
)

SUGGESTIONS = {
    "Suche aktuelle KI-Nachrichten": "Suche nach den aktuellsten Entwicklungen im Bereich künstliche Intelligenz diese Woche.",
    "Erkläre RAG": "Erkläre mir Retrieval Augmented Generation einfach und verständlich. Was sind die Vorteile gegenüber Fine-Tuning?",
    "Vergleiche Claude vs GPT-4": "Vergleiche Claude 3.5 Sonnet mit GPT-4o. Wo liegen die Stärken und Schwächen?",
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

@st.cache_data(ttl=5, show_spinner=False)
def load_stacks(base_url: str) -> tuple[dict[str, dict], str]:
    """Discover stacks from the server.

    Returns ``({stack_name: payload}, default_stack_name)``. On a
    single-stack server (``/v1/stacks`` → 404) falls back to
    ``/v1/models`` and synthesises a single pseudo-stack. On a fully
    unreachable server the pseudo-stack carries ``ready=False`` so the
    UI can render the selector without crashing.
    """
    data = fetch_stacks(base_url)
    if data is not None:
        stacks = {entry["name"]: entry for entry in data.get("stacks", [])}
        default = data.get("default") or (next(iter(stacks), "") if stacks else "")
        return stacks, default
    models = fetch_models_fallback(base_url)
    synthetic_name = models[0] if models else "default"
    return (
        {
            synthetic_name: {
                "name": synthetic_name,
                "llm": "",
                "search": "",
                "ready": bool(models),
                "description": "Single-stack server (kein /v1/stacks)",
                "models": {},
            }
        },
        synthetic_name,
    )


@st.cache_data(ttl=5, show_spinner=False)
def load_health(base_url: str) -> dict:
    """Discover health from the server (5 s cache)."""
    return fetch_health(base_url)


def format_stack_option(stack_name: str, stacks: dict[str, dict]) -> str:
    """Format the stack selectbox option with an elegant ready indicator.

    Ready stacks get a solid bullet (``●``), unavailable ones an open
    circle (``○``) — monochrome and understated so the selectbox stays
    typographically coherent. The actual coloring (muted green for
    ready, muted red for unavailable) is applied via CSS targeting the
    selectbox options via ``.stack-dot-ready`` / ``.stack-dot-down``.
    """
    entry = stacks.get(stack_name, {})
    marker = "●" if entry.get("ready") else "○"
    return f"{marker}  {stack_name}"


def get_mode_display(profile_value: str) -> str:
    """Returns a compact label for the active report profile."""
    return REPORT_PROFILE_LABELS.get(profile_value, profile_value.title())


def shorten_label(label: str, max_length: int = 24) -> str:
    """Shortens long labels for composer pills without losing context."""
    if len(label) <= max_length:
        return label
    return f"{label[:max_length - 1]}..."


def get_settings_summary() -> str:
    """Compact primary summary shown permanently in the composer-meta row.

    Only the two most stable decisions stay visible (stack and profile);
    every other setting is collapsed into the ``···`` info popover so
    the meta row never wraps into the composer area.
    """
    stack_label = shorten_label(st.session_state.active_stack or "—", 22)
    return (
        f'{stack_label} · '
        f'{get_mode_display(st.session_state.report_profile)}'
    )


def render_settings_detail_popover() -> None:
    """Detailed active-settings view inside the composer-meta info popover.

    Shown only when the user opens the ``···`` popover. Uses the same
    CSS-dot pattern as the stack-health row so on/off states are
    instantly readable without colored emoji. Default values are
    labelled as such to help operators tell tuned vs untuned runs
    apart at a glance.
    """

    def _row(label: str, value: str, *, is_default: bool, state: str = "neutral") -> None:
        dot = (
            '<span class="stack-health-dot ready"></span>'
            if state == "on"
            else '<span class="stack-health-dot down"></span>'
            if state == "off"
            else '<span class="settings-detail-bullet"></span>'
        )
        default_tag = (
            '<span class="settings-detail-default">Default</span>'
            if is_default else ''
        )
        st.markdown(
            f'<div class="settings-detail-row">'
            f'{dot}'
            f'<span class="settings-detail-label">{escape(label)}</span>'
            f'<span class="settings-detail-value">{escape(value)}</span>'
            f'{default_tag}'
            f'</div>',
            unsafe_allow_html=True,
        )

    stack = st.session_state.active_stack or "—"
    profile = get_mode_display(st.session_state.report_profile)
    effort = st.session_state.reasoning_effort
    web_on = bool(st.session_state.get("web_search_enabled", True))
    policy_on = bool(st.session_state.get("de_policy_bias_enabled", True))
    streaming_on = bool(st.session_state.streaming_enabled)
    include_progress = bool(st.session_state.include_progress)

    _row("Stack", stack, is_default=False)
    _row("Profil", profile, is_default=st.session_state.report_profile == "compact")
    _row("Aufwand", effort, is_default=(effort == "Auto"))
    _row(
        "Websuche",
        "an" if web_on else "aus",
        is_default=web_on,
        state="on" if web_on else "off",
    )
    _row(
        "DE-Policy",
        "an" if policy_on else "aus",
        is_default=policy_on,
        state="on" if policy_on else "off",
    )
    _row(
        "Streaming",
        "an" if streaming_on else "aus",
        is_default=streaming_on,
        state="on" if streaming_on else "off",
    )
    _row(
        "Progress-Events",
        "an" if include_progress else "aus",
        is_default=include_progress,
        state="on" if include_progress else "off",
    )
    st.caption(
        f"Feintuning: confidence_stop = {int(st.session_state.confidence_stop)} · "
        f"max_total_seconds = {int(st.session_state.max_total_seconds)} · "
        f"first_round_queries = {int(st.session_state.first_round_queries)}"
    )


def get_stack_avatar(stack_name: str) -> str:
    """Pick an avatar icon for the given stack name."""
    return STACK_AVATARS.get(stack_name, DEFAULT_STACK_AVATAR)


def build_agent_overrides() -> dict:
    """Collect the per-request overrides from the current session state.

    When the effort level is ``Auto`` both ``max_rounds`` and
    ``min_rounds`` are omitted so the server-side AgentSettings default
    (derived from ``report_profile``) applies.

    When the Websuche toggle is off ``skip_search=True`` is emitted so
    the server short-circuits the research graph and answers purely
    from the LLM provider.

    The ``enable_de_policy_bias`` flag is always emitted so the active
    state is visible in the server-side audit log even when it matches
    the server default — that makes it easy to correlate a chat with
    the policy-heuristic configuration used.
    """
    effort = EFFORT_ROUNDS.get(st.session_state.reasoning_effort)
    web_search = bool(st.session_state.get("web_search_enabled", True))
    overrides: dict = {
        "report_profile": st.session_state.report_profile,
        "confidence_stop": int(st.session_state.confidence_stop),
        "max_total_seconds": int(st.session_state.max_total_seconds),
        "first_round_queries": int(st.session_state.first_round_queries),
        "enable_de_policy_bias": bool(
            st.session_state.get("de_policy_bias_enabled", True)
        ),
    }
    if effort is not None:
        overrides["max_rounds"], overrides["min_rounds"] = effort
    if not web_search:
        overrides["skip_search"] = True
    return overrides


def normalize_prompt_input(prompt_value, fallback_prompt: str | None = None) -> tuple[str | None, list[str]]:
    """Normalizes text/file input returned by st.chat_input."""
    if prompt_value is None:
        return fallback_prompt, []

    if hasattr(prompt_value, "text"):
        file_names = [uploaded_file.name for uploaded_file in getattr(
            prompt_value, "files", [])]
        prompt_text = prompt_value.text.strip()
        if not prompt_text and file_names:
            prompt_text = "Bitte analysiere die hochgeladenen Dateien."
        return prompt_text or None, file_names

    prompt_text = str(prompt_value).strip()
    return prompt_text or None, []


def render_uploaded_files(file_names: list[str]) -> None:
    """Renders uploaded files as compact chips below the user message."""
    if not file_names:
        return

    chips = "".join(
        f'<span class="file-chip">{escape(file_name)}</span>'
        for file_name in file_names
    )
    st.markdown(f'<div class="file-chip-row">{chips}</div>', unsafe_allow_html=True)


def render_message(message: dict) -> None:
    """Renders a single chat message with the custom visual style."""
    if message["role"] == "user":
        avatar_icon = "user"
    else:
        avatar_icon = get_stack_avatar(message.get("stack", ""))
    content = message.get("content", "")
    if not isinstance(content, str):
        content = getattr(content, "text", str(content))

    if message["role"] == "user":
        st.markdown(
            '<div class="user-msg-anchor" data-anchor="user"></div>',
            unsafe_allow_html=True,
        )

    with st.chat_message(message["role"], avatar=avatar_icon):
        copy_payload = base64.b64encode(content.encode("utf-8")).decode("ascii")
        st.markdown(
            f'<button type="button" class="copy-btn" data-copy-b64="{copy_payload}" '
            f'aria-label="Kopieren" title="Kopieren">'
            f'<svg class="copy-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>'
            f'<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>'
            f'</svg>'
            f'<svg class="check-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
            f'<polyline points="20 6 9 17 4 12"></polyline>'
            f'</svg>'
            f'</button>',
            unsafe_allow_html=True,
        )

        progress_lines = message.get("progress") or []
        if progress_lines:
            with st.expander(
                f"Rechercheverlauf · {len(progress_lines)} Schritt(e)",
                expanded=False,
                icon=":material/manage_search:",
            ):
                for line in progress_lines:
                    st.markdown(f"`{line}`")

        st.markdown(content)

        if message["role"] == "user":
            render_uploaded_files(message.get("files", []))


def effort_summary(effort: str) -> str:
    """Compact human-readable summary of an Aufwand level."""
    value = EFFORT_ROUNDS.get(effort)
    if value is None:
        return "Server-Default (richtet sich nach Profil)"
    max_r, min_r = value
    return f"max_rounds = {max_r} · min_rounds = {min_r}"


# ---------------------------------------------------------------------------
# Server discovery + session state init
# ---------------------------------------------------------------------------

SERVER_BASE_URL = get_base_url()
API_KEY = get_api_key()
AVAILABLE_STACKS, DEFAULT_STACK = load_stacks(SERVER_BASE_URL)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "active_stack" not in st.session_state or st.session_state.active_stack not in AVAILABLE_STACKS:
    st.session_state.active_stack = DEFAULT_STACK or (next(iter(AVAILABLE_STACKS), ""))

if "report_profile" not in st.session_state:
    st.session_state.report_profile = "deep"

if "trigger_prompt" not in st.session_state:
    st.session_state.trigger_prompt = None

if "reasoning_effort" not in st.session_state:
    st.session_state.reasoning_effort = "Auto"

if "confidence_stop" not in st.session_state:
    st.session_state.confidence_stop = 8

if "max_total_seconds" not in st.session_state:
    st.session_state.max_total_seconds = 600

if "first_round_queries" not in st.session_state:
    st.session_state.first_round_queries = 6

if "streaming_enabled" not in st.session_state:
    st.session_state.streaming_enabled = True

if "include_progress" not in st.session_state:
    st.session_state.include_progress = True

if "web_search_enabled" not in st.session_state:
    st.session_state.web_search_enabled = True

if "de_policy_bias_enabled" not in st.session_state:
    st.session_state.de_policy_bias_enabled = True

if "pending_response" not in st.session_state:
    st.session_state.pending_response = None

if "cancel_requested" not in st.session_state:
    st.session_state.cancel_requested = False


# ---------------------------------------------------------------------------
# Sidebar (Minimalist)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### Verlauf")

    st.caption("Heute")
    st.markdown("Vergleich LLM-Architekturen")
    st.markdown("Azure Bing Grounding Setup")

    st.caption("Gestern")
    st.markdown("Perplexity Sonar Integration")

    st.divider()

    st.markdown("### Server")
    health = load_health(SERVER_BASE_URL)
    health_status = health.get("status", "unreachable")
    health_indicator = {
        "ok": ("🟢", "bereit"),
        "degraded": ("🟠", "degradiert"),
        "unreachable": ("⚪", "nicht erreichbar"),
    }.get(health_status, ("⚪", health_status))
    st.caption(f"{health_indicator[0]} {health_indicator[1]}")
    st.caption(f"URL: `{SERVER_BASE_URL}`")
    st.caption("Auth: Bearer gesetzt" if API_KEY else "Auth: keine")
    if st.button("Aktualisieren", key="refresh_discovery", icon=":material/refresh:"):
        load_stacks.clear()
        load_health.clear()
        st.rerun()

    st.divider()

    st.markdown("### Einstellungen")
    st.toggle("Streaming", key="streaming_enabled")
    st.toggle("Progress-Events", key="include_progress")


with st.container(key="page_shell"):
    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    with st.container(key="app_header"):
        brand_col, actions_col = st.columns([6, 3], vertical_alignment="center")
        with brand_col:
            st.markdown(
                '<div class="brand-kicker">Conversational Research Agent</div>'
                '<div class="brand-header">inqtrix</div>',
                unsafe_allow_html=True,
            )
        with actions_col:
            with st.container(horizontal=True, gap="small", vertical_alignment="center", horizontal_alignment="right"):
                if st.button(
                    "Neu",
                    key="new_chat",
                    icon=":material/edit_note:",
                    help="Neuen Chat starten",
                ):
                    st.session_state.messages = []
                    st.session_state.trigger_prompt = None
                    st.session_state.pending_response = None
                    st.rerun()
                if st.button(
                    "Löschen",
                    key="delete_chat",
                    icon=":material/delete:",
                    help="Alle Nachrichten löschen",
                    disabled=len(st.session_state.messages) == 0,
                ):
                    st.session_state.messages = []
                    st.session_state.trigger_prompt = None
                    st.session_state.pending_response = None
                    st.rerun()

    # -----------------------------------------------------------------------
    # Main Chat Logic
    # -----------------------------------------------------------------------
    has_history = len(st.session_state.messages) > 0

    conversation_shell = st.container(key="conversation_shell")

    with conversation_shell:
        if not has_history:
            with st.container(key="empty_state_shell"):
                st.markdown(
                    '<div class="welcome-wrap">'
                    '<div class="prototype-notice">Hinweis: Hierbei handelt sich um einen Prototypen</div>'
                    '<div class="welcome-meta">'
                    'GitHub: <a href="https://github.com/BZandi/inqtrix" target="_blank" rel="noopener">github.com/BZandi/inqtrix</a>'
                    '<span class="sep">·</span>'
                    'Lizenz: Apache 2.0'
                    '</div>'
                    '<div class="welcome-title">Guten Tag.</div>'
                    '<div class="welcome-sub">Wie kann inqtrix Ihnen heute helfen?</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )

                with st.container(key="suggestion_shell"):
                    c1, c2, c3 = st.columns([1.15, 3.7, 1.15])
                    with c2:
                        sel_suggestion = st.pills(
                            "Vorschlaege:",
                            options=list(SUGGESTIONS.keys()),
                            label_visibility="collapsed"
                        )
                        if sel_suggestion:
                            st.session_state.trigger_prompt = SUGGESTIONS[sel_suggestion]
                            st.rerun()

        for msg in st.session_state.messages:
            render_message(msg)

        if has_history:
            st.iframe(
                """
                <script>
                  (function() {
                    const parentDoc = window.parent.document;
                    if (parentDoc.__copyHandlerAttached) return;
                    parentDoc.__copyHandlerAttached = true;
                    const s = parentDoc.createElement('script');
                    s.textContent = `
                      (function() {
                        function markCopied(btn) {
                          btn.classList.add('copied');
                          setTimeout(function() { btn.classList.remove('copied'); }, 1400);
                        }
                        function fallbackCopy(text) {
                          const ta = document.createElement('textarea');
                          ta.value = text;
                          ta.setAttribute('readonly', '');
                          ta.style.position = 'fixed';
                          ta.style.top = '-1000px';
                          ta.style.opacity = '0';
                          document.body.appendChild(ta);
                          ta.select();
                          let ok = false;
                          try { ok = document.execCommand('copy'); } catch (e) {}
                          document.body.removeChild(ta);
                          return ok;
                        }
                        document.addEventListener('click', function(e) {
                          const btn = e.target.closest('.copy-btn');
                          if (!btn) return;
                          e.preventDefault();
                          e.stopPropagation();
                          const b64 = btn.getAttribute('data-copy-b64');
                          if (!b64) return;
                          let text = '';
                          try {
                            const bin = atob(b64);
                            const bytes = Uint8Array.from(bin, function(c) { return c.charCodeAt(0); });
                            text = new TextDecoder('utf-8').decode(bytes);
                          } catch (err) { return; }
                          if (navigator.clipboard && navigator.clipboard.writeText) {
                            navigator.clipboard.writeText(text).then(function() {
                              markCopied(btn);
                            }).catch(function() {
                              if (fallbackCopy(text)) markCopied(btn);
                            });
                          } else {
                            if (fallbackCopy(text)) markCopied(btn);
                          }
                        });
                      })();
                    `;
                    parentDoc.head.appendChild(s);
                  })();
                </script>
                """,
                height=1,
            )

        scroll_to_user = (
            st.session_state.pending_response is not None
            or st.session_state.pop("scroll_to_last_user", False)
        )
        if scroll_to_user:
            st.iframe(
                """
                <script>
                  (function() {
                    const doc = window.parent.document;
                    const run = () => {
                      const anchors = doc.querySelectorAll('.user-msg-anchor');
                      if (!anchors.length) return;
                      const last = anchors[anchors.length - 1];
                      last.scrollIntoView({ behavior: 'auto', block: 'start' });
                    };
                    requestAnimationFrame(run);
                  })();
                </script>
                """,
                height=1,
            )

    # -----------------------------------------------------------------------
    # Composer
    # -----------------------------------------------------------------------
    with st.container(key="composer_shell"):
        with st.container(key="composer_status"):
            if has_history:
                st.markdown(
                    '<div class="composer-meta">'
                    '<span class="composer-prototype-pill">Hinweis: Prototyp</span>'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="composer-meta-spacer"></div>',
                    unsafe_allow_html=True,
                )

        with st.container(key="composer_input"):
            chat_input_val = st.chat_input(
                "Wie kann ich helfen?",
                key="composer_input_field",
                accept_file="multiple",
                file_type=["pdf", "png", "jpg", "txt", "csv"],
            )

        with st.container(key="composer_footer"):
            with st.container(horizontal=True, gap="small", vertical_alignment="center"):
                with st.popover(
                    ":material/list_alt: Profil",
                    help="Report-Profil, Antworttiefe und DE-Policy-Heuristik.",
                    key="composer_profile_popover",
                    width="content",
                ):
                    st.caption("Profil")
                    st.markdown(
                        '<div class="popover-explainer">'
                        'Bestimmt <strong>Tiefe und Länge</strong> der Antwort. '
                        '<strong>Kompakt</strong> liefert eine schnelle, knappe '
                        'Synthese mit kleinerem Kontextfenster und wenigen Zitaten. '
                        '<strong>Deep Research</strong> fächert Suchanfragen '
                        'breiter auf, sammelt deutlich mehr Quellen und erzeugt '
                        'eine ausführlichere, gegliederte Synthese — dauert aber '
                        'entsprechend länger.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    st.radio(
                        "Recherchemodus",
                        REPORT_PROFILES,
                        key="report_profile",
                        format_func=get_mode_display,
                        label_visibility="collapsed",
                    )

                    st.caption("DE-Policy-Heuristik")
                    st.markdown(
                        '<div class="popover-explainer">'
                        'Aktiviert die <strong>deutsche Gesundheits- und '
                        'Sozialpolitik-Kalibrierung</strong> des Agenten: '
                        'Quality-Site-Injection für DE-Policy-Domänen, '
                        'Utility-Stop-Unterdrückung bei politischen Themen '
                        'und einen <strong>Risiko-Bonus (+2)</strong> für '
                        'einschlägige Keywords. <strong>Ausschalten</strong> '
                        'für allgemeine oder nicht-deutsche Einsätze, um den '
                        'DE-spezifischen Bias zu entfernen.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    st.toggle(
                        "DE-Policy-Bias aktiv",
                        key="de_policy_bias_enabled",
                        help=(
                            "Serverseitiges Flag `enable_de_policy_bias`. "
                            "Wirkt auf `KeywordRiskScorer` und die Stop-"
                            "Kriterien. Default: an."
                        ),
                    )
                with st.popover(
                    ":material/memory: Modell",
                    help="Aktiver Stack und Provider-Verfügbarkeit.",
                    key="composer_stack_popover",
                    width="content",
                ):
                    st.caption("Stack")
                    if AVAILABLE_STACKS:
                        st.selectbox(
                            "Stack",
                            options=list(AVAILABLE_STACKS.keys()),
                            key="active_stack",
                            format_func=lambda value: format_stack_option(value, AVAILABLE_STACKS),
                            label_visibility="collapsed",
                            help=(
                                "**●** — Provider-Readiness-Check war erfolgreich.\n\n"
                                "**○** — Provider antwortet aktuell nicht.\n\n"
                                "Die Liste stammt aus `GET /v1/stacks` und wird "
                                "bei jedem Seitenaufbau serverseitig (5 s Cache) "
                                "über den ADR-WS-12-Readiness-Probe erneuert."
                            ),
                        )
                        active_entry = AVAILABLE_STACKS.get(st.session_state.active_stack, {})
                        ready_count = sum(1 for s in AVAILABLE_STACKS.values() if s.get("ready"))
                        status_class = "ready" if active_entry.get("ready") else "down"
                        status_label = "bereit" if active_entry.get("ready") else "nicht erreichbar"
                        st.markdown(
                            f'<div class="stack-health-row">'
                            f'<span class="stack-health-dot {status_class}"></span>'
                            f'<span class="stack-health-name">{escape(status_label)}</span>'
                            f'<span class="stack-health-label">'
                            f'· {ready_count} / {len(AVAILABLE_STACKS)} verfügbar'
                            f'</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        if active_entry.get("description"):
                            st.caption(active_entry["description"])
                        models = active_entry.get("models") or {}
                        chip_parts: list[str] = []
                        if active_entry.get("llm"):
                            chip_parts.append(f"LLM: `{active_entry['llm']}`")
                        if active_entry.get("search"):
                            chip_parts.append(f"Search: `{active_entry['search']}`")
                        if models.get("reasoning_model"):
                            chip_parts.append(f"Reasoning: `{models['reasoning_model']}`")
                        if models.get("search_model"):
                            chip_parts.append(f"Search-Modell: `{models['search_model']}`")
                        if chip_parts:
                            st.caption(" · ".join(chip_parts))
                    else:
                        st.warning("Kein Stack verfügbar — Server nicht erreichbar?")
                with st.popover(
                    ":material/tune: Aufwand",
                    help="Recherche-Tiefe und Feintuning-Schranken.",
                    key="composer_effort_popover",
                    width="content",
                ):
                    st.caption("Aufwand")
                    st.markdown(
                        '<div class="popover-explainer">'
                        'Steuert, wie viele Recherche-Runden der Agent höchstens '
                        'und mindestens ausführt. Jede Runde verfeinert Suchanfragen '
                        'und sammelt weitere Quellen. <strong>Mehr Runden</strong> '
                        '= tiefere Recherche, aber längere Laufzeit.'
                        '</div>',
                        unsafe_allow_html=True,
                    )
                    st.radio(
                        "Aufwand",
                        EFFORT_LEVELS,
                        key="reasoning_effort",
                        label_visibility="collapsed",
                        help=EFFORT_HELP,
                    )
                    st.caption(f"Aktuell: {effort_summary(st.session_state.reasoning_effort)}")

                    st.caption("Feintuning")
                    st.slider(
                        "Confidence Stop",
                        1, 10,
                        key="confidence_stop",
                        help="Evaluator-Confidence (1-10), ab der die Stop-Kaskade 'done' setzen darf.",
                    )
                    st.slider(
                        "Max Total Seconds",
                        30, 1800,
                        key="max_total_seconds",
                        step=30,
                        help="Hartes Wall-Clock-Limit pro Durchlauf (Sekunden).",
                    )
                    st.slider(
                        "First Round Queries",
                        1, 20,
                        key="first_round_queries",
                        help="Anzahl der breiten Suchanfragen in Runde 0.",
                    )
                st.toggle(
                    "Websuche",
                    key="web_search_enabled",
                    help=(
                        "**An** (Standard): Der Agent führt Websuche, "
                        "Quellensynthese und Claim-Verifikation durch.\n\n"
                        "**Aus**: Reiner Chat — der Server ruft nur den "
                        "LLM-Provider direkt mit Frage + Historie, ohne "
                        "Recherche. Ideal für Definitions- oder "
                        "Konzeptfragen, die keine aktuellen Quellen brauchen."
                    ),
                )
                with st.popover(
                    ":material/fact_check:",
                    help="Alle aktiven Einstellungen anzeigen",
                    key="composer_settings_detail_popover",
                    width="content",
                ):
                    st.caption("Aktive Einstellungen")
                    render_settings_detail_popover()

    actual_prompt, uploaded_files = normalize_prompt_input(
        chat_input_val,
        st.session_state.trigger_prompt,
    )

    # Handle new user interaction
    if actual_prompt:
        st.session_state.trigger_prompt = None
        st.session_state.cancel_requested = False

        user_message = {
            "role": "user",
            "content": actual_prompt,
            "files": uploaded_files,
        }
        st.session_state.messages.append(user_message)
        st.session_state.pending_response = {
            "stack": st.session_state.active_stack,
            "overrides": build_agent_overrides(),
            "stream": bool(st.session_state.streaming_enabled),
            "include_progress": bool(st.session_state.include_progress),
        }

        st.rerun()

    # If a cancel was queued, swallow it *before* the pending-response block
    # so the UI does not start another stream against the server.
    if st.session_state.cancel_requested and not st.session_state.pending_response:
        st.session_state.cancel_requested = False
        st.session_state.messages.append({
            "role": "assistant",
            "content": "_Recherche abgebrochen._",
            "stack": st.session_state.active_stack,
        })
        st.session_state.scroll_to_last_user = True
        st.rerun()

    if st.session_state.pending_response:
        pending = st.session_state.pending_response
        stack_name = pending["stack"]
        overrides = pending["overrides"]
        stream_requested = pending["stream"]
        include_progress = pending["include_progress"]

        wire_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
            if isinstance(m.get("content"), str) and m.get("content")
        ]

        # The stop-button ``on_click`` callback: sets a cancel flag and
        # clears the pending_response so that the next rerun does not
        # immediately re-enter this block. Streamlit's built-in
        # widget-interaction rerun aborts the currently running script,
        # which closes the ``httpx.stream(...)`` context manager, which
        # in turn causes the server-side ``_watch_disconnect`` task to
        # set ``cancel_event`` and surface it into the LangGraph probe.
        def _on_stop_click() -> None:
            st.session_state.cancel_requested = True
            st.session_state.pending_response = None

        with conversation_shell:
            pending_avatar = get_stack_avatar(stack_name)
            with st.chat_message("assistant", avatar=pending_avatar):
                # The Stopp button sits at the top of the in-progress
                # assistant bubble, right next to the running status
                # widget, so the user can find it without hunting.
                with st.container(key="stopp_row"):
                    st.button(
                        "Recherche stoppen",
                        key="stop_stream_button",
                        icon=":material/stop_circle:",
                        help=(
                            "Bricht den laufenden Recherche-Request ab. "
                            "Der Server erkennt den Disconnect und cancelt "
                            "den Agent-Run innerhalb weniger hundert "
                            "Millisekunden."
                        ),
                        on_click=_on_stop_click,
                        type="secondary",
                        use_container_width=True,
                    )

                full_response = ""
                error_state: dict[str, str] = {}
                progress_log: list[str] = []

                if stream_requested:
                    status_ctx = (
                        st.status("Recherche läuft...", expanded=True)
                        if include_progress
                        else None
                    )

                    def _stream_generator(
                        state=error_state,
                        status=status_ctx,
                        log=progress_log,
                    ):
                        for kind, payload in stream_chat(
                            wire_messages,
                            stack=stack_name,
                            agent_overrides=overrides,
                            include_progress=include_progress,
                            api_key=API_KEY,
                            base_url=SERVER_BASE_URL,
                        ):
                            if kind == "progress":
                                log.append(payload)
                                if status is not None:
                                    status.write(f"`{payload}`")
                            elif kind == "delta":
                                yield payload
                            elif kind == "error":
                                state["message"] = payload
                                return
                            elif kind == "done":
                                return

                    try:
                        streamed = st.write_stream(_stream_generator())
                    except Exception as exc:  # noqa: BLE001
                        error_state["message"] = f"Streaming-Fehler: {exc}"
                        streamed = ""

                    if isinstance(streamed, list):
                        full_response = "".join(str(chunk) for chunk in streamed)
                    else:
                        full_response = str(streamed or "")

                    if status_ctx is not None:
                        if error_state.get("message"):
                            status_ctx.update(
                                label=f"Fehler: {error_state['message']}",
                                state="error",
                                expanded=True,
                            )
                        else:
                            # Leave the status widget visible and collapsed so
                            # the operator can re-open the progress log at any
                            # time. After the rerun the dropdown is re-rendered
                            # from ``message["progress"]`` via render_message.
                            status_ctx.update(
                                label=f"Rechercheverlauf · {len(progress_log)} Schritt(e)",
                                state="complete",
                                expanded=False,
                            )
                else:
                    with st.spinner("Recherche läuft..."):
                        response = call_chat(
                            wire_messages,
                            stack=stack_name,
                            agent_overrides=overrides,
                            api_key=API_KEY,
                            base_url=SERVER_BASE_URL,
                        )
                    if "error" in response:
                        error_state["message"] = response["error"].get(
                            "message", "Unbekannter Fehler"
                        )
                    else:
                        choices = response.get("choices") or []
                        if choices:
                            full_response = (
                                choices[0].get("message", {}).get("content", "")
                            )
                        else:
                            error_state["message"] = "Antwort enthielt keine 'choices'"

                    if full_response:
                        st.markdown(full_response)

                error_message = error_state.get("message")
                if error_message and not full_response:
                    st.error(error_message)
                elif error_message and full_response:
                    st.warning(f"Vorzeitig beendet: {error_message}")

        if not error_message or full_response:
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response or (error_message or ""),
                "stack": stack_name,
                "progress": list(progress_log),
            })
        st.session_state.pending_response = None
        st.session_state.scroll_to_last_user = True

        st.rerun()
