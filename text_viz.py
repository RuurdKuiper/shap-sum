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
        ax.text(total_width+0.3, y_pos - 0.5, f"{shap_difference:.2f}", fontsize=6, color="black", ha="center")
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
