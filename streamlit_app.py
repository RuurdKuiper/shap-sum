import streamlit as st
import plotly.graph_objects as go
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
from text_viz import render_shap_heatmap
from graph_viz import create_shap_figure
from transformers import AutoTokenizer
from huggingface_hub import login

st.set_page_config(layout="wide")

# ================================
# AUTHENTICATE & LOAD TOKENIZER
# ================================
@st.cache_resource
def authenticate_huggingface():
    hf_token = st.secrets["HUGGINGFACE_TOKEN"]
    login(token=hf_token)
    return hf_token

@st.cache_resource
def load_tokenizer():
    authenticate_huggingface()
    return AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

tokenizer = load_tokenizer()

# ===================
# 🔹 LOAD SHAP DATA
# ===================
SHAP_FILE_PATH = "shap_results.json"

@st.cache_data
def load_shap_data():
    if not os.path.exists(SHAP_FILE_PATH):
        st.error(f"SHAP data file not found at {SHAP_FILE_PATH}. Please generate the data first.")
        return []
    with open(SHAP_FILE_PATH, "r") as f:
        return json.load(f)

# ===============================
# 🔹 STREAMLIT APP MAIN INTERFACE
# ===============================
st.title("SHAP Dependency Visualization for LLM Summaries")

# Load SHAP data
shap_data = load_shap_data()
if not shap_data:
    st.stop()

# Store SHAP results in session state
if "shap_results" not in st.session_state:
    st.session_state.shap_results = shap_data
num_summaries = len(shap_data)

# ==========================
# 🔹 SUMMARY DROPDOWN LIST
# ==========================
dropdown_options = []
for i, summary in enumerate(shap_data):
    # Check if summary contains hallucinations
    eval_data = summary.get("evaluation", {})
    hallucinated = "hallucination_triggers" in eval_data and eval_data["hallucination_triggers"]
    
    # Add 🔴 marker if hallucinated
    marker = "🔴" if hallucinated else ""
    dropdown_options.append(f"{i}: {summary['article'][:50]}... {marker}")

# Select summary
selected_option = st.selectbox("Select a summary (summaries with hallucinations are marked with 🔴):", dropdown_options)
index = int(selected_option.split(":")[0])  # Extract index

# Load selected summary
selected_summary = shap_data[index]

# ===================
# 🔹 RADIO MENU
# ===================
viz_mode = st.radio("Select View Mode:", ["Text & Summary", "SHAP Heatmap", "Graph"], index=0)

# ===================
# 📌 MODE 1: TEXT SUMMARY
# ===================
if viz_mode == "Text & Summary":
    st.subheader("Original Article")
    st.write(selected_summary["article"][:1000])

    st.subheader("Generated Summary")
    st.write(selected_summary["generated_summary"])

    st.subheader("Reference Summary")
    st.write(selected_summary["reference_summary"])

    st.subheader("GPT-4 Evaluation")
    st.write(selected_summary["evaluation"])

# ===================
# 📌 MODE 2: SHAP HEATMAP
# ===================
elif viz_mode == "SHAP Heatmap":

    # Call function to render heatmap
    render_shap_heatmap(selected_summary, tokenizer)

# ===================
# 📌 MODE 3: GRAPH VISUALIZATION
# ===================
elif viz_mode == "Graph":
    st.subheader("Click on a Token to See Its Connections")

    plot_spot = st.empty()  # Placeholder for graph

    if 'selected_data' not in st.session_state:
        st.session_state.selected_data = []

    if st.session_state.selected_data:
        selection = st.session_state.selected_data.get("selection", [])
        if selection:
            points = selection.get("points", [])
            if points:
                point_index = points[0]["point_index"]
                x_position = points[0]["x"]  # 0 for input, 1 for output
                is_input = (x_position == 0)

                # Generate updated figure with token highlighted
                fig = create_shap_figure(index, st.session_state.shap_results, highlight=(point_index, is_input))
            else:
                fig = create_shap_figure(index, st.session_state.shap_results)
        else:
            fig = create_shap_figure(index, st.session_state.shap_results)
    else:
        fig = create_shap_figure(index, st.session_state.shap_results)

    # Display graph plot
    with plot_spot:
        st.session_state.selected_data = st.plotly_chart(fig, use_container_width=False, on_select="rerun", selection_mode='points')
