import streamlit as st
import plotly.graph_objects as go
import numpy as np
import json
import os
import re
import matplotlib.pyplot as plt
from text_viz import visualize_shap_multiline
from graph_viz import create_shap_figure

st.set_page_config(layout="wide")

# Set up file path for SHAP data
SHAP_FILE_PATH = "shap_results.json"

# Load SHAP data from file
@st.cache_data
def load_shap_data():
    if not os.path.exists(SHAP_FILE_PATH):
        st.error(f"SHAP data file not found at {SHAP_FILE_PATH}. Please generate the data first.")
        return []
    
    with open(SHAP_FILE_PATH, "r") as f:
        return json.load(f)


# Streamlit App
st.title("SHAP Dependency Visualization for LLM Summaries")

# Load SHAP data
shap_data = load_shap_data()
if not shap_data:
    st.stop()

# Load data
if "shap_results" not in st.session_state:
    st.session_state.shap_results = load_shap_data()
num_summaries = len(st.session_state.shap_results)

# Create dropdown options: (Index + First Few Words of Article)
dropdown_options = [f"{i}: {shap_data[i]['article'][:50]}..." for i in range(num_summaries)]

# Display the dropdown menu
selected_option = st.selectbox("Select a summary:", dropdown_options)

# Extract the selected index
index = int(selected_option.split(":")[0])  # Get index from string

# Load the selected summary
selected_summary = shap_data[index]

# Tokenize
input_tokens = selected_summary["input_tokens"]
input_tokens = [re.sub(r'^[▁Ġ]', '', token) for token in input_tokens]
output_tokens = selected_summary["output_tokens"]
output_tokens = [re.sub(r'^[▁Ġ]', '', token) for token in output_tokens]
shap_matrix = np.array(selected_summary["shap_matrix"])
original_probs = np.array(selected_summary["original_probs"])
masked_probs = np.array(selected_summary["masked_probs"])

# Create a dictionary to map output tokens to their indices
output_token_dict = {token: i for i, token in enumerate(output_tokens)}

# **🔹 Add a radio button for visualization modes**
viz_mode = st.radio("Select View Mode:", ["Text & Summary", "SHAP Heatmap", "Graph"], index=0)

# **📌 Mode 1: Show text, summaries, and GPT-4 evaluation**
if viz_mode == "Text & Summary":
    st.subheader("Original Article")
    st.write(selected_summary["article"][:1000])

    st.subheader("Generated Summary")
    st.write(selected_summary["generated_summary"])

    st.subheader("Reference Summary")
    st.write(selected_summary["reference_summary"])

    st.subheader("GPT-4 Evaluation")
    st.write(selected_summary["evaluation"])

# **📌 Mode 2: SHAP Heatmap Visualization**
elif viz_mode == "SHAP Heatmap":
    # Dropdown for selecting the output token
    selected_output_token_text = st.selectbox(
        "Select an output token:", 
        list(output_token_dict.keys())  # Show only token names
    )

    # Retrieve the corresponding index
    output_token_index = output_token_dict[selected_output_token_text]

    # **Create a figure before calling `visualize_shap_multiline`**
    fig, ax = plt.subplots(figsize=(12, len(input_tokens)//12))  # Create figure explicitly
    visualize_shap_multiline(
        input_tokens, 
        shap_matrix[:, output_token_index], 
        original_probs[output_token_index], 
        masked_probs[:, output_token_index], 
        output_tokens[output_token_index],
        ax=ax  # **Pass the existing axis to the function**
    )

    # **Use `st.pyplot(fig)` to display the Matplotlib figure**
    st.pyplot(fig)

# **📌 Mode 3: Graph-based Token Dependency**
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

                # Generate an updated figure with the selected token highlighted
                fig = create_shap_figure(index, st.session_state.shap_results, highlight=(point_index, is_input))
            else:
                fig = create_shap_figure(index, st.session_state.shap_results)
        else:
            fig = create_shap_figure(index, st.session_state.shap_results)
    else:
        fig = create_shap_figure(index, st.session_state.shap_results)

    # Display the plot with selection enabled
    with plot_spot:
        st.session_state.selected_data = st.plotly_chart(fig, use_container_width=False, on_select="rerun", selection_mode='points')
