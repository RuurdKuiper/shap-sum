import plotly.graph_objects as go
import re
import numpy as np
import streamlit as st

# Function to create the SHAP visualization figure
def create_shap_figure(index, shap_results, highlight=None):
    """
    Create a Plotly figure for SHAP token dependency visualization.

    - index: Index of the summary in the dataset.
    - highlight: Tuple (token_index, is_input) specifying which token to highlight.
    """
    if index >= len(shap_results):
        st.error(f"Invalid index {index}. Choose from 0 to {len(shap_results)-1}.")
        return None

    data = shap_results[index]
    input_tokens = data["input_tokens"]
    input_tokens = [re.sub(r'^[▁Ġ]', '', token) for token in input_tokens]
    output_tokens = data["output_tokens"]
    output_tokens = [re.sub(r'^[▁Ġ]', '', token) for token in output_tokens]
    shap_matrix = np.array(data["shap_matrix"])  # Convert back to numpy array

    # Normalize SHAP values for better visualization
    max_shap = shap_matrix.max()
    if max_shap > 0:
        shap_matrix /= max_shap  # Scale between 0 and 1

    # Define token positions (spaced out for clarity)
    input_positions = np.linspace(0, len(input_tokens), len(input_tokens))
    output_positions = np.linspace(0, len(input_tokens), len(output_tokens))

    fig = go.Figure()

    # Add input tokens (left side)
    fig.add_trace(go.Scatter(
        x=[0] * len(input_tokens),
        y=input_positions,
        mode="markers+text",
        text=input_tokens,
        textposition="middle left",
        marker=dict(size=10, color="blue"),
        name="Input Tokens",
        customdata=list(range(len(input_tokens))),
        hoverinfo="text"
    ))

    # Add output tokens (right side)
    fig.add_trace(go.Scatter(
        x=[0.6] * len(output_tokens),
        y=output_positions,
        mode="markers+text",
        text=output_tokens,
        textposition="middle right",
        marker=dict(size=16, color="red"),
        name="Output Tokens",
        customdata=list(range(len(output_tokens))),
        hoverinfo="text"
    ))

    # Add connections with default low opacity
    for i, input_pos in enumerate(input_positions):
        for j, output_pos in enumerate(output_positions):
            opacity = 0.2
            width = 4 * abs(shap_matrix[i, j])

            # Highlight specific connections when a token is selected
            if highlight:
                token_index, is_input = highlight
                if (is_input and i == token_index) or (not is_input and j == token_index):
                    opacity = 1
                    if is_input:
                        width = 6 * abs(shap_matrix[i, j])/max(abs(shap_matrix[:,j]))  # Make highlighted lines thicker
                    if not is_input:
                        width = 6 * abs(shap_matrix[i, j])/max(abs(shap_matrix[i,:]))

            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[input_pos, output_pos],
                mode="lines",
                line=dict(width=width, color=f"rgba(50,50,50,{opacity})"),
                hoverinfo="none",
            ))

    # Remove legend and adjust layout
    fig.update_layout(
        title=f"SHAP Influence Graph for Summary {index}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.2, 0.8]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        height=1200
    )

    return fig