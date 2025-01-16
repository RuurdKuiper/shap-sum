import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math
import re

def visualize_shap_multiline(input_tokens, shap_values, original_value, masked_values, output_token, ax=None):
    """
    Visualize SHAP values for a specific output token using Matplotlib in Streamlit.
    - Splits tokens into sentences, placing each on a new line.
    - Displays SHAP values under each token.
    - If `ax` is provided, plots on an existing axis. Otherwise, creates a new figure.
    """

    # Remove special characters at token start
    input_tokens = [re.sub(r'^[▁Ġ]', '', token) for token in input_tokens]

    # Normalize SHAP values to range [-1,1] for consistent color mapping
    norm = colors.Normalize(vmin=-1, vmax=1)
    cmap = plt.get_cmap("RdYlGn")  # Red (negative SHAP) → Yellow (neutral) → Green (positive)

    # Define number of tokens per line
    tokens_per_line = 12
    num_lines = (len(input_tokens) // tokens_per_line) + 1

    # **Use existing axis if provided, otherwise create a new figure**
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, num_lines))  # Adjust height dynamically
    else:
        fig = ax.figure  # Get the parent figure

    y_pos = num_lines  # Start plotting from top (each row is a sentence)
    total_width = 0

    for i, token in enumerate(input_tokens):
        if i >= len(shap_values):
            break  # Prevent indexing errors

        # Compute SHAP difference
        shap_value = shap_values[i]
        masked_value = masked_values[i]
        shap_difference = math.exp(original_value) - math.exp(masked_value)

        # Convert SHAP difference to color
        color = cmap(norm(shap_difference))

        # Adjust token width (approximate based on text length)
        text_width = len(token)

        # Place token text on figure
        ax.text(total_width, y_pos, token, fontsize=12, color="black",
                bbox=dict(facecolor=color, alpha=0.8, boxstyle="round,pad=0.3"))

        # Add SHAP differences below tokens
        ax.text(total_width + 0.1, y_pos - 0.45, f"{shap_difference:.2f}", fontsize=6, color="black", ha="center")
        #  ax.text(total_width+0.3, y_pos - 0.5, f"{shap_difference:.2f}|{math.exp(original_value):.2f}|{math.exp(masked_value):.2f}", fontsize=6, color="black", ha="center")

        total_width += text_width * 0.1 + 0.5  # Add extra spacing between tokens

        # Move to next line after 12 tokens
        if i % tokens_per_line == 11:
            y_pos -= 1
            total_width = 0

    # Adjust figure layout
    ax.set_xlim(0, 12)  # Limit X range to fit sentences
    ax.set_ylim(0, num_lines + 0.5)  # Adjust height dynamically
    ax.axis("off")  # Hide axes
    ax.set_title(f"SHAP values for output token: '{output_token}', with original probability: {math.exp(original_value):.2f}")

    return fig  # **Return the figure for display**

def render_shap_heatmap(summary_data, tokenizer):
    """ Renders SHAP heatmap visualization. """
    
    # Extract hallucination triggers
    hallucinated_words = set()
    evaluation_data = summary_data.get("evaluation", {})
    if isinstance(evaluation_data, dict):
        hallucinated_words = set(evaluation_data.get("hallucination_triggers", []))

    # Tokenize hallucinated words
    hallucinated_tokens = set()
    for word in hallucinated_words:
        hallucinated_tokens.update(tokenizer.tokenize(word))
    hallucinated_tokens = [re.sub(r'^[▁Ġ]', '', token) for token in hallucinated_tokens]

    # Tokenize output tokens
    output_tokens = [re.sub(r'^[▁Ġ]', '', token) for token in summary_data["output_tokens"]]
    hallucinated_token_indices = {i for i, token in enumerate(output_tokens) if token in hallucinated_tokens}

    # Store both plain and styled versions
    plain_output_tokens = output_tokens[:]
    styled_output_tokens = [
        f"🔴**{token.upper()}**🔴" if i in hallucinated_token_indices else token
        for i, token in enumerate(output_tokens)
    ]

    # Highlight hallucinations in pill selection
    selected_output_token_text = st.pills(
        "Select an output token (tokens associated with hallucinations are marked with 🔴):", 
        options=styled_output_tokens,
        selection_mode="single",
        key="output_token_selection"
    )

    # Find the selected token index
    token_map = {styled: plain for styled, plain in zip(styled_output_tokens, plain_output_tokens)}
    if selected_output_token_text:
        selected_plain_token = token_map[selected_output_token_text]
        output_token_index = plain_output_tokens.index(selected_plain_token)
    else:
        output_token_index = 0  # Default to first token

    # SHAP Visualization
    shap_matrix = np.array(summary_data["shap_matrix"])
    original_probs = np.array(summary_data["original_probs"])
    masked_probs = np.array(summary_data["masked_probs"])
    input_tokens = [re.sub(r'^[▁Ġ]', '', token) for token in summary_data["input_tokens"]]

    # Create Matplotlib figure
    fig, ax = plt.subplots(figsize=(12, len(input_tokens) // 18))
    visualize_shap_multiline(
        input_tokens, 
        shap_matrix[:, output_token_index], 
        original_probs[output_token_index], 
        masked_probs[:, output_token_index], 
        output_tokens[output_token_index],
        ax=ax
    )

    # Display figure
    st.pyplot(fig)
