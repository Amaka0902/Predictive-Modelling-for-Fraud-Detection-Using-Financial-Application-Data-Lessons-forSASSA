"""Look and feel of the dashboard: pastel colours, cards, and a Plotly style."""
import plotly.graph_objects as go
import streamlit as st

import config

P = config.PASTEL
PASTEL_SEQUENCE = [P["lavender"], P["pink"], P["blue"], P["mint"], P["peach"], P["butter"]]

CSS = f"""
<style>
.stApp {{ background: linear-gradient(180deg, #FBF8FF 0%, #F6FAFF 100%); }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #EFE6FA 0%, #E6F0FF 100%); }}
[data-testid="stSidebar"] * {{ color: {P['text']}; }}
h1, h2, h3, h4 {{ color: {P['text']}; letter-spacing: -0.01em; }}
.hero {{
  background: linear-gradient(120deg, {P['lavender']} 0%, {P['pink']} 55%, {P['blue']} 100%);
  border-radius: 24px; padding: 30px 34px; margin-bottom: 22px;
  box-shadow: 0 8px 24px rgba(155,126,200,0.18);
}}
.hero h1 {{ margin: 0 0 6px 0; font-size: 2.0rem; color: #34304a; }}
.hero p {{ margin: 0; font-size: 1.02rem; color: #4a4663; }}
.card {{
  background: #ffffff; border-radius: 20px; padding: 20px 22px; margin-bottom: 16px;
  border: 1px solid #EFE8F7; box-shadow: 0 4px 16px rgba(120,100,160,0.08);
}}
.card h4 {{ margin-top: 0; margin-bottom: 8px; }}
.tile {{ border-radius: 20px; padding: 16px 18px; text-align: left; box-shadow: 0 3px 12px rgba(120,100,160,0.08); }}
.tile .label {{ font-size: 0.80rem; text-transform: uppercase; letter-spacing: 0.06em; color: #5c5877; }}
.tile .value {{ font-size: 1.7rem; font-weight: 700; color: {P['text']}; line-height: 1.25; }}
.tile .note {{ font-size: 0.80rem; color: #6d6988; }}
.pill {{ display: inline-block; padding: 4px 16px; border-radius: 999px; font-weight: 700; color: #3D3A50; }}
.step {{ background: #fff; border-radius: 16px; padding: 12px 14px; border: 1px solid #EFE8F7; height: 100%; }}
.step b {{ color: #7d5fb0; }}
.stButton > button, .stDownloadButton > button {{
  border-radius: 999px; border: 1px solid #D9C9EE; background: linear-gradient(90deg, #E8DAF6, #FFD9E6);
  color: {P['text']}; font-weight: 600; padding: 0.45rem 1.3rem;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{ border-color: #B79CD6; filter: brightness(0.98); }}
.stTabs [data-baseweb="tab"] {{ border-radius: 999px; padding: 6px 16px; }}
.stTabs [aria-selected="true"] {{ background: #EFE6FA; }}
[data-testid="stMetric"] {{ background: #fff; border-radius: 16px; padding: 10px 14px; border: 1px solid #EFE8F7; }}
.small {{ font-size: 0.85rem; color: {P['muted']}; }}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(title, body_html):
    st.markdown(f'<div class="card"><h4>{title}</h4>{body_html}</div>', unsafe_allow_html=True)


def tile(label, value, note="", colour=None):
    colour = colour or P["lavender"]
    st.markdown(
        f'<div class="tile" style="background:{colour}55;border:1px solid {colour};">'
        f'<div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>',
        unsafe_allow_html=True)


def pill(text, colour):
    return f'<span class="pill" style="background:{colour};">{text}</span>'


def style_fig(fig, height=360, legend=True):
    """Same pastel look for every plotly chart."""
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.0)",
        font=dict(color=P["text"], family="Nunito, Segoe UI, sans-serif"),
        margin=dict(l=10, r=10, t=40, b=10), showlegend=legend,
        colorway=PASTEL_SEQUENCE, hoverlabel=dict(bgcolor="white"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E6DDF2")
    fig.update_yaxes(gridcolor="#EFE8F7", zeroline=False)
    return fig


def need_results():
    """Pages 4-8 stop here with a friendly message if there are no predictions yet."""
    if "results" not in st.session_state:
        st.info("No predictions yet. Upload a dataset (page 2), check it (page 3) and press **Run predictions**.")
        st.stop()
