import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MultiHeadAttnViz import MultiHeadAttentionViz  # Your class must accept mask=... and return_attn=True
from attention_masks import make_causal_mask



# Set page config
st.set_page_config(
    page_title="Multi-Head Attention Visualizer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {padding: 2rem 3rem;}
    .stSlider {max-width: 500px;}
    .stButton>button {width: 100%;}
    .stTextInput>div>div>input {font-family: monospace;}
    </style>
""", unsafe_allow_html=True)

def generate_sample_data(seq_len, d_model):
    """Generate sample data for visualization"""
    x = torch.randn(1, seq_len, d_model)  # [batch_size, seq_len, d_model]
    tokens = [f"token_{i}" for i in range(seq_len)]
    return x, tokens

def plot_attention_heads(attn_weights, tokens, head_idx=None, figsize=(10, 8)):
    attn = attn_weights[0].detach().cpu().numpy()  # [H,T,S]
    num_heads, seq_len, _ = attn.shape
    
    # Calculate dynamic font size based on sequence length
    base_font_size = 8
    font_size = max(4, base_font_size - seq_len // 10)
    
    if head_idx is not None:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(attn[head_idx], cmap='viridis')
        ax.set_title(f'Attention Head {head_idx}', fontsize=font_size + 2)
        
        # Set ticks and labels with dynamic font size
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(tokens, rotation=90, ha='right', fontsize=font_size)
        ax.set_yticklabels(tokens, fontsize=font_size)
        
        plt.colorbar(im)
    else:
        n_cols = min(4, num_heads)
        n_rows = (num_heads + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else np.array([axes])
        
        for h in range(num_heads):
            ax = axes[h]
            im = ax.imshow(attn[h], cmap='viridis')
            ax.set_title(f'Head {h}', fontsize=font_size + 2)
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add colorbar to each subplot
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
        # Hide unused subplots
        for h in range(num_heads, len(axes)):
            axes[h].axis('off')
            
        plt.tight_layout()
    
    return fig

# Sidebar
st.sidebar.header("Model Configuration")
d_model = st.sidebar.slider("Model Dimension (d_model)", min_value=16, max_value=128, value=64, step=16)
num_heads = st.sidebar.slider("Number of Attention Heads", min_value=1, max_value=8, value=4)
seq_len = st.sidebar.slider("Sequence Length", min_value=4, max_value=64, value=16)
# Update input data when sequence length changes
if 'prev_seq_len' not in st.session_state:
    st.session_state.prev_seq_len = seq_len

if st.session_state.prev_seq_len != seq_len:
    st.session_state.prev_seq_len = seq_len
    st.session_state.base_x, st.session_state.tokens = generate_sample_data(seq_len, d_model)
    st.rerun()
dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.5, value=0.0, step=0.05)
use_causal = st.sidebar.checkbox("Use causal mask (auto-regressive)", value=True)

# Add a run button
run_button = st.sidebar.button("Run Model")

# Check if d_model is divisible by num_heads
if d_model % num_heads != 0:
    st.sidebar.error(f"Error: d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    st.stop()


# Generate or keep base input for verification
if "base_x" not in st.session_state or run_button:
    st.session_state.base_x, st.session_state.tokens = generate_sample_data(seq_len, d_model)

# Initialize model
model = MultiHeadAttentionViz(d_model=d_model, num_heads=num_heads, dropout=dropout)
model.eval()  # turn off dropout for stable visualization

# Build mask (float 1 keep / 0 mask-out)
mask = None
if use_causal:
    mask = make_causal_mask(T=seq_len, S=seq_len, batch_size=1, num_heads=num_heads, device=st.session_state.base_x.device, dtype=torch.float32)

# Forward pass
with torch.no_grad():
    output, attn_weights = model(st.session_state.base_x, st.session_state.base_x, st.session_state.base_x, mask=mask, return_attn=True)

# Main content
st.title("Multi-Head Attention Visualization (with Causal Masking)")
st.markdown("""
Toggle causal masking to enforce auto-regressive attention (no look-ahead). 
With causal masking ON, attention above the diagonal should be zero and outputs up to a position are unaffected by future tokens.
""")

# Display configuration
col1, col2, col3, col4 = st.columns(4)
with col1: st.metric("d_model", d_model)
with col2: st.metric("# Heads", num_heads)
with col3: st.metric("Seq Len", seq_len)
with col4: st.metric("Causal", "ON" if use_causal else "OFF")

# Visualization mode
st.subheader("Visualization")
viz_mode = st.radio("Select visualization mode:", ["All Heads", "Single Head"], horizontal=True)
if viz_mode == "Single Head":
    head_idx = st.slider("Select Head", 0, num_heads-1, 0)
    fig = plot_attention_heads(attn_weights, st.session_state.tokens, head_idx=head_idx)
else:
    fig = plot_attention_heads(attn_weights, st.session_state.tokens)
st.pyplot(fig)

# Raw attention weights (ensure 2D)
if st.checkbox("Show raw attention weights"):
    st.subheader("Attention Weights")
    H = int(attn_weights.shape[1]); T = int(attn_weights.shape[2]); S = int(attn_weights.shape[3])
    raw_mode = st.radio("Display as:", ["Per head (T x S)", "Flattened (T x H*S)"], horizontal=True)
    if raw_mode == "Per head (T x S)":
        head_idx_raw = st.slider("Head to view", 0, H - 1, 0)
        mat = attn_weights[0, head_idx_raw].detach().cpu().numpy()
        df = pd.DataFrame(mat.round(3), index=[f"q_{i}" for i in range(T)], columns=[f"k_{j}" for j in range(S)])
        st.dataframe(df)
    else:
        arr = attn_weights[0].detach().cpu().numpy()  # [H,T,S]
        flat = arr.transpose(1, 0, 2).reshape(T, H * S)
        cols = pd.MultiIndex.from_product([range(H), range(S)], names=["head", "key"])
        df = pd.DataFrame(flat.round(3), index=[f"q_{i}" for i in range(T)], columns=cols)
        st.dataframe(df)

# Verification section: Auto-regressive property
with st.expander("Verify auto-regressive property"):
    st.markdown("""
    We check two things:
    1) Attention shows no look-ahead: all weights above the diagonal are 0 when causal masking is ON.
    2) Outputs up to a chosen prefix length L do not change if we perturb future tokens (> L).
    """)
    L = st.slider("Prefix length L (inclusive index)", min_value=0, max_value=seq_len-2, value=min(4, seq_len-2))
    colA, colB = st.columns(2)

    # 1) No-lookahead metric
    with colA:
        if use_causal:
            # Compute max weight above diagonal
            attn = attn_weights[0]  # [H,T,S]
            T = attn.shape[1]; S = attn.shape[2]
            future_mask = torch.triu(torch.ones((T, S), dtype=torch.bool), diagonal=1)
            max_future = attn[:, future_mask].abs().max().item()
            st.metric("Max future attention (should be 0)", f"{max_future:.2e}")
        else:
            st.info("Enable causal masking to check future-attention = 0.")

    # 2) Output invariance test
    with colB:
        if st.button("Run invariance test"):
            # Build a perturbed input: change tokens after L
            x_base = st.session_state.base_x
            x_pert = x_base.clone()
            x_pert[:, L+1:, :] = torch.randn_like(x_pert[:, L+1:, :])
            with torch.no_grad():
                out_base, _ = model(x_base, x_base, x_base, mask=mask, return_attn=True)
                out_pert, _ = model(x_pert, x_pert, x_pert, mask=mask, return_attn=True)
            prefix_diff = (out_base[:, :L+1, :] - out_pert[:, :L+1, :]).abs().max().item()
            if use_causal:
                st.metric("Max difference in outputs up to L", f"{prefix_diff:.2e}")
                if prefix_diff < 1e-6:
                    st.success("Pass: outputs up to L are unchanged by future-token changes.")
                else:
                    st.warning("Outputs up to L changed; check mask wiring or dropout.")
            else:
                st.info("Enable causal masking to expect invariance of prefix outputs.")

# Add some explanation
with st.expander("How to interpret causal masking"):
    st.markdown("""
    - **Causal Masking** ensures each position can only attend to previous positions
    - The upper triangle of attention weights is masked (set to -inf)
    - This enforces auto-regressive generation (no looking ahead)
    - The diagonal shows self-attention to the current position
    - Lower triangle shows attention to previous positions
    - All heads should respect the causal constraint when masking is enabled
    """)

