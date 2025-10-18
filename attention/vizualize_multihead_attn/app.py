import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from MultiHeadAttnViz import MultiHeadAttentionViz

# Set page config
st.set_page_config(
    page_title="Multi-Head Attention Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {padding: 2rem 3rem;}
    .stSlider {max-width: 500px;}
    .stButton>button {width: 100%;}
    .stTextInput>div>div>input {font-family: monospace;}
    </style>
""", unsafe_allow_html=True)

def generate_sample_data(seq_len, d_model, num_heads):
    """Generate sample data for visualization"""
    x = torch.randn(1, seq_len, d_model)  # [batch_size, seq_len, d_model]
    tokens = [f"token_{i}" for i in range(seq_len)]
    return x, tokens

def plot_attention_heads(attn_weights, tokens, head_idx=None, figsize=(10, 8)):
    """
    Plot attention weights for visualization.

    attn_weights: Tensor [B, H, T, S]
    """
    attn = attn_weights[0].detach().cpu().numpy()  # [num_heads, seq_len, seq_len]
    num_heads, seq_len, _ = attn.shape
    
    if head_idx is not None:
        # Plot single head
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(attn[head_idx], cmap='viridis')
        ax.set_title(f'Attention Head {head_idx}')
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=90, ha='right')
        ax.set_yticklabels(tokens)
        plt.colorbar(im)
    else:
        # Plot all heads in a grid
        n_cols = min(4, num_heads)
        n_rows = (num_heads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        
        for h in range(num_heads):
            row = h // n_cols
            col = h % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            im = ax.imshow(attn[h], cmap='viridis')
            ax.set_title(f'Head {h}')
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.tight_layout()
    
    return fig

# Sidebar controls
st.sidebar.header("Model Configuration")
d_model = st.sidebar.slider("Model Dimension (d_model)", min_value=16, max_value=128, value=64, step=16)
num_heads = st.sidebar.slider("Number of Attention Heads", min_value=1, max_value=8, value=4)
seq_len = st.sidebar.slider("Sequence Length", min_value=4, max_value=20, value=8)
dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)

# Add a run button
run_button = st.sidebar.button("Run Model")

# Check if d_model is divisible by num_heads
if d_model % num_heads != 0:
    st.sidebar.error(f"Error: d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    st.stop()

# Generate sample data
x, tokens = generate_sample_data(seq_len, d_model, num_heads)

# Initialize model
model = MultiHeadAttentionViz(
    d_model=d_model,
    num_heads=num_heads,
    dropout=dropout
)

# Forward pass
with torch.no_grad():
    output, attn_weights = model(x, x, x, return_attn=True)

# Main content
st.title("Multi-Head Attention Visualization")
st.markdown("""
This app visualizes attention patterns in a multi-head attention layer. 
Adjust the parameters in the sidebar to see how different configurations affect the attention weights.
""")

# Display model configuration
st.subheader("Model Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Model Dimension (d_model)", d_model)
with col2:
    st.metric("Number of Heads", num_heads)
with col3:
    st.metric("Sequence Length", seq_len)

# Visualization options
st.subheader("Visualization Options")
viz_mode = st.radio(
    "Select visualization mode:",
    ["All Heads", "Single Head"],
    horizontal=True
)

if viz_mode == "Single Head":
    head_idx = st.slider("Select Head", 0, num_heads-1, 0)
    fig = plot_attention_heads(attn_weights, tokens, head_idx=head_idx)
else:
    fig = plot_attention_heads(attn_weights, tokens)

st.pyplot(fig)

# Display raw attention weights (ensure 2D for DataFrame)
if st.checkbox("Show raw attention weights"):
    st.subheader("Attention Weights")

    # Shapes: [B, H, T, S] -> use the first item in batch
    H = int(attn_weights.shape[1])
    T = int(attn_weights.shape[2])
    S = int(attn_weights.shape[3])

    raw_mode = st.radio(
        "Display as:",
        ["Per head (T x S matrix)", "Flattened across heads (T x (H*S))"],
        horizontal=True
    )

    if raw_mode == "Per head (T x S matrix)":
        head_idx_raw = st.slider("Head to view", 0, H - 1, 0)
        mat = attn_weights[0, head_idx_raw].detach().cpu().numpy()  # [T, S]
        df = pd.DataFrame(
            mat.round(3),
            index=[f"q_{i}" for i in range(T)],
            columns=[f"k_{j}" for j in range(S)],
        )
        st.dataframe(df)
    else:
        # Flatten [H, T, S] -> [T, H*S] with MultiIndex columns (head, key)
        arr = attn_weights[0].detach().cpu().numpy()  # [H, T, S]
        flat = arr.transpose(1, 0, 2).reshape(T, H * S)  # [T, H*S]
        cols = pd.MultiIndex.from_product(
            [range(H), range(S)], names=["head", "key"]
        )
        df = pd.DataFrame(
            flat.round(3),
            index=[f"q_{i}" for i in range(T)],
            columns=cols,
        )
        st.dataframe(df)

# Add some explanation
with st.expander("How to interpret these visualizations"):
    st.markdown("""
    - Each square shows how much attention each position (y-axis) pays to other positions (x-axis)
    - Brighter colors indicate higher attention weights
    - In self-attention, the diagonal is often strong but not always
    - Different heads often learn to focus on different aspects of the input
    """)

# Add a button to regenerate with new random data
if st.button("Generate New Random Input"):
    x, tokens = generate_sample_data(seq_len, d_model, num_heads)
    with torch.no_grad():
        output, attn_weights = model(x, x, x, return_attn=True)
    st.rerun()