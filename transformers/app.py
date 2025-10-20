import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from typing import Optional
import time
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn

from mini_tranformers import MiniTransformer
from toy_tasks import get_dataloader
from attention import make_causal_mask
from train_mini_transformer import train_epoch, evaluate, plot_training_curves

# page config
st.set_page_config(
    page_title = "mini-transformer",
    layout = "wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 1rem 2rem;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
    .success-box {background-color: #d4edda; padding: 15px; border-radius: 5px; color: #155724;}
    .info-box {background-color: #d1ecf1; padding: 15px; border-radius: 5px; color: #0c5460;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = {
        'train_loss': [],
        'train_acc' : [],
        'val_loss' : [],
        'val_acc': []
    }
if 'is_training' not in st.session_state:
    st.session_state.is_training = False
if 'model_config' not in st.session_state:
    st.session_state.model_config = None

# Title
st.title("Mini Transformer Playground")
st.markdown("**Train, test, and visualize transformers on toy tasks** — All in one place!")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Config",
    "Training",
    "Testing",
    "Vizualization",
    "Tutorial"
])

# ---- Tab 1: Model Configuration ----
with tab1:
    st.header("Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Architecture")
        vocab_size = st.number_input("Vocabulary Size", min_value=10, max_value=100, value=20, step=5)
        d_model = st.select_slider("d_model", options=[32, 64, 128, 256], value=64)
        num_heads = st.select_slider("Number of Heads", options=[2, 4, 8], value=4)
        d_ff = st.select_slider("d_ff (FFN hidden)", options=[128, 256, 512, 1024], value=256)
        num_layers = st.slider("Number of Layers", min_value=1, max_value=6, value=2)
        dropout = st.slider("Dropout", min_value=0.0, max_value=0.5, value=0.1, step=0.05)
    
    with col2:
        st.subheader("Training Settings")
        task = st.selectbox("Task", ["copy", "reverse", "next_token"])
        
        # Add pattern selection for next_token task
        pattern = None
        if task == 'next_token':
            pattern = st.selectbox(
                "Pattern Type",
                ["fibonacci", "primes", "alternating", "multiplicative", "random_walk", "skip"],
                index=0
            )
        seq_len = st.slider("Sequence Length", min_value=4, max_value=32, value=10)
        batch_size = st.slider("Batch Size", min_value=8, max_value=128, value=32, step=8)
        learning_rate = st.select_slider("Learning Rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, format_func=lambda x: f"{x:.0e}")
        num_epochs = st.slider("Number of Epochs", min_value=5, max_value=100, value=20, step=5)
        use_causal_mask = st.checkbox("Use Causal Masking", value=(task == "next_token"))
    
    # Build Model Button

    if st.button("Build Model", type = "primary", use_container_width=True):
        with st.spinner("Building Model..."):
            try:
                model = MiniTransformer(
                    vocab_size = vocab_size,
                    d_model = d_model,
                    num_heads = num_heads,
                    d_ff = d_ff,
                    num_layers = num_layers,
                    max_seq_len=seq_len,
                    dropout = dropout,
                    pos_encoding='sinusoidal',
                    tie_weights=False
                )
                st.session_state.model = model
                st.session_state.model_config = {
                    'vocab_size': vocab_size,
                    'd_model': d_model,
                    'num_heads': num_heads,
                    'd_ff': d_ff,
                    'num_layers': num_layers,
                    'seq_len': seq_len,
                    'dropout': dropout,
                    'task': task,
                    'pattern': pattern,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'use_causal_mask': use_causal_mask
                }

                st.session_state.training_history = {
                    'train_loss': [],
                    'train_acc' : [],
                    'val_loss' : [],
                    'val_acc': []
                }

                st.success("✅ Model Built successfully!!")

                # Display model info with custom styling
                st.markdown("""
                <style>
                .metric-container {
                    background-color: #0e1117;
                    border-radius: 8px;
                    padding: 10px;
                    margin: 5px 0;
                }
                .metric-label {
                    color: #9e9e9e;
                    font-size: 0.9rem;
                }
                .metric-value {
                    color: #f0f0f0;
                    font-size: 1.2rem;
                    font-weight: bold;
                }
                </style>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                
                def metric_box(label, value):
                    return f"""
                    <div class="metric-container">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """
                
                with col1:
                    st.markdown(metric_box("Total Parameters", f"{model.count_parameters():,}"), unsafe_allow_html=True)
                with col2:
                    st.markdown(metric_box("Layers", num_layers), unsafe_allow_html=True)
                with col3:
                    st.markdown(metric_box("Heads per Layer", num_heads), unsafe_allow_html=True)
                with col4:
                    st.markdown(metric_box("Model Dim", d_model), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error building model: {str(e)}")
        # show current model status
        if st.session_state.model is not None:
            st.markdown("---")
            st.markdown('<div class="info-box">✅ Model is ready for training!</div>', unsafe_allow_html=True)
        with st.expander("📐 Model Architecture"):
            st.code(f"""
                Mini-Transformer Architecture:
                {'='*50}
                Input: [batch_size, seq_len] token indices
                ↓
                Token Embedding: {vocab_size} → {d_model}
                ↓
                Positional Encoding (Sinusoidal)
                ↓
                {'─'*50}
                TransformerBlock × {num_layers}
                │  ├─ LayerNorm
                │  ├─ MultiHeadAttention ({num_heads} heads, d_k={d_model//num_heads})
                │  ├─ Residual Connection
                │  ├─ LayerNorm
                │  ├─ FeedForward ({d_model} → {d_ff} → {d_model})
                │  └─ Residual Connection
                {'─'*50}
                ↓
                Output Projection: {d_model} → {vocab_size}
                ↓
                Logits: [batch_size, seq_len, {vocab_size}]
                {'='*50}
                Total Parameters: {model.count_parameters():,}
            """, language="text")
# ---- Tab 2: Training ----
with tab2:
    st.header("Training")

    if st.session_state.model is None:
        st.warning("⚠️ Please build a model first in the 'Model Config' tab!")
    else:
        config = st.session_state.model_config

        # Training controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Task:** {config['task'].upper()} | **Seq Len:** {config['seq_len']} | **Batch:** {config['batch_size']}")
        with col2:
            train_samples = st.number_input("Train Samples", 1000, 10000, 3000, 500)
        with col3:
            val_samples = st.number_input("Val Samples", 100, 2000, 500, 100)
        
        # Start training button
        if st.button("🚀 Start Training", type="primary", use_container_width=True, disabled=st.session_state.is_training):
            st.session_state.is_training = True
            

            # Create dataloaders
            train_loader = get_dataloader(
                config['task'], config['vocab_size'], config['seq_len'],
                train_samples, config['batch_size'], shuffle=True,
                pattern=config['pattern'] if config['task'] == 'next_token' else None
            )
            val_loader = get_dataloader(
                config['task'], config['vocab_size'], config['seq_len'],
                val_samples, config['batch_size'], shuffle=False,
                pattern=config['pattern'] if config['task'] == 'next_token' else None
            )
            
            # Store the dataset in session state for reference in expected output generation
            if hasattr(train_loader, 'dataset'):
                st.session_state.dataset = train_loader.dataset
            
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
            # Training progress
            pbar = st.progress(0)
            status_text = st.empty()
            metric_placeholder = st.empty()
            chart_placeholder = st.empty()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(st.session_state.model.parameters(), lr=config['learning_rate'])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.session_state.model.to(device)
            # Training loop
            for epoch in range(1, config['num_epochs'] + 1):
                # Train epoch
                train_loss, train_acc = train_epoch(
                    st.session_state.model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    use_casual_mask=config['use_causal_mask']
                )
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                # Validate
                val_loss, val_acc = evaluate(
                    st.session_state.model,
                    val_loader,
                    criterion,
                    device,
                    use_casual_mask=config['use_causal_mask']
                )
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)

                # Update progress
                pbar.progress(epoch / config['num_epochs'])
                status_text.write(f"**Epoch {epoch}/{config['num_epochs']}**  | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                # Update metrics with custom styling
                with metric_placeholder.container():
                    st.markdown("""
                    <style>
                    .train-metric-container {
                        background-color: #1a1a2e;
                        border-radius: 8px;
                        padding: 10px;
                        margin: 5px;
                    }
                    .train-metric-label {
                        color: #a1a1a1;
                        font-size: 0.9rem;
                        margin-bottom: 5px;
                    }
                    .train-metric-value {
                        color: #ffffff;
                        font-size: 1.1rem;
                        font-weight: bold;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    def train_metric_box(label, value):
                        return f"""
                        <div class="train-metric-container">
                            <div class="train-metric-label">{label}</div>
                            <div class="train-metric-value">{value}</div>
                        </div>
                        """
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(train_metric_box("Train Loss", f"{train_loss:.4f}"), unsafe_allow_html=True)
                    with col2:
                        st.markdown(train_metric_box("Train Acc", f"{train_acc:.4f}"), unsafe_allow_html=True)
                    with col3:
                        st.markdown(train_metric_box("Val Loss", f"{val_loss:.4f}"), unsafe_allow_html=True)
                    with col4:
                        st.markdown(train_metric_box("Val Acc", f"{val_acc:.4f}"), unsafe_allow_html=True)
                                
                # update chart
                if epoch > 1:
                    fig = plot_training_curves(
                        history['train_loss'],
                        history['train_acc'],
                        history['val_loss'],
                        history['val_acc']
                    )
                    chart_placeholder.pyplot(fig)
                    plt.close(fig)
            
            # Training complete
            st.session_state.training_history = history
            st.session_state.is_training = False
            st.session_state.model.cpu() # Move back to CPU

            st.markdown('<div class="success-box">🎉 Training completed successfully!</div>', unsafe_allow_html=True)
            
            # Final metrics with custom styling
            st.markdown("### Final Results")
            st.markdown("""
            <style>
            .final-metric-container {
                background-color: #1e1e2f;
                border-radius: 8px;
                padding: 12px;
                margin: 5px;
                border-left: 4px solid #4f46e5;
            }
            .final-metric-label {
                color: #a1a1a1;
                font-size: 0.9rem;
                margin-bottom: 5px;
            }
            .final-metric-value {
                color: #ffffff;
                font-size: 1.2rem;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)

            def final_metric_box(label, value):
                return f"""
                <div class="final-metric-container">
                    <div class="final-metric-label">{label}</div>
                    <div class="final-metric-value">{value}</div>
                </div>
                """

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(final_metric_box("Best Val Loss", f"{min(history['val_loss']):.4f}"), unsafe_allow_html=True)
            with col2:
                st.markdown(final_metric_box("Best Val Acc", f"{max(history['val_acc']):.4f}"), unsafe_allow_html=True)
            with col3:
                best_epoch = np.argmax(history['val_acc']) + 1
                st.markdown(final_metric_box("Best Epoch", best_epoch), unsafe_allow_html=True)
            

        # Show training history if available
        if st.session_state.training_history['train_loss']:
            st.markdown("---")
            st.subheader("Training History")
            
            history = st.session_state.training_history
            
            # Training History Metrics with custom styling
            st.markdown("""
            <style>
            .history-metric-container {
                background-color: #1a1a2e;
                border-radius: 8px;
                padding: 12px;
                margin: 5px;
                transition: all 0.3s ease;
            }
            .history-metric-container:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            .history-metric-label {
                color: #a1a1a1;
                font-size: 0.9rem;
                margin-bottom: 5px;
            }
            .history-metric-value {
                color: #ffffff;
                font-size: 1.2rem;
                font-weight: bold;
            }
            </style>
            """, unsafe_allow_html=True)

            def history_metric_box(label, value):
                return f"""
                <div class="history-metric-container">
                    <div class="history-metric-label">{label}</div>
                    <div class="history-metric-value">{value}</div>
                </div>
                """

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(history_metric_box("Final Train Loss", f"{history['train_loss'][-1]:.4f}"), unsafe_allow_html=True)
            with col2:
                st.markdown(history_metric_box("Final Train Acc", f"{history['train_acc'][-1]:.4f}"), unsafe_allow_html=True)
            with col3:
                st.markdown(history_metric_box("Final Val Loss", f"{history['val_loss'][-1]:.4f}"), unsafe_allow_html=True)
            with col4:
                st.markdown(history_metric_box("Final Val Acc", f"{history['val_acc'][-1]:.4f}"), unsafe_allow_html=True)

            # Training curves
            fig = plot_training_curves(
                history['train_loss'],
                history['train_acc'],
                history['val_loss'],
                history['val_acc']
            )
            st.pyplot(fig)
            plt.close(fig)

        # Save model if available
        if st.session_state.model is not None:
            st.markdown("---")
            st.subheader("Model")
            
            # Save model
            if st.button("💾 Save Model"):
                model_path = "mini_transformer.pt"
                torch.save(st.session_state.model.state_dict(), model_path)
                st.success(f"Model saved to {model_path}")
            
            # Load model

# ---- Tab 3: Inference ----
with tab3:
    st.header("Model Testing")

    if st.session_state.model is None:
        st.warning("⚠️ Please build and train a model first in the 'Model Config' tab!")
        st.stop()

    config = st.session_state.model_config
    model = st.session_state.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Input mode selection
    input_mode = st.radio("Input Mode", 
                         ["Preset Patterns", "Manual Input", "Random Sequence"], 
                         horizontal=True,
                         help="Choose how to generate test sequences")
    
    # Initialize test input in session state if not exists
    if 'test_input' not in st.session_state:
        st.session_state.test_input = torch.zeros((1, config['seq_len']), dtype=torch.long)
    
    # Input generation based on mode
    if input_mode == "Preset Patterns":
        st.markdown("### Select a Pattern")
        
        # Get pattern type if available
        pattern = getattr(getattr(st.session_state, 'dataset', None), 'pattern', 'increment')
        
        # Pattern selection
        pattern_type = st.selectbox(
            "Pattern Type",
            ["Fibonacci", "Primes", "Incrementing", "Alternating"],
            index=0
        )
        
        # Generate sequence based on pattern
        if st.button("Generate Sequence"):
            if pattern_type == "Fibonacci":
                # Generate Fibonacci sequence
                a, b = 1, 1
                seq = [a, b]
                for _ in range(config['seq_len'] - 2):
                    a, b = b, (a + b) % config['vocab_size']
                    seq.append(b)
                st.session_state.test_input = torch.tensor([seq[:config['seq_len']]])
                
            elif pattern_type == "Primes":
                # Generate prime numbers
                def is_prime(n):
                    if n < 2:
                        return False
                    for i in range(2, int(n**0.5) + 1):
                        if n % i == 0:
                            return False
                    return True
                
                primes = [2]
                n = 3
                while len(primes) < config['seq_len']:
                    if is_prime(n):
                        primes.append(n % config['vocab_size'])
                    n += 1
                st.session_state.test_input = torch.tensor([primes[:config['seq_len']]])
                
            elif pattern_type == "Incrementing":
                # Simple incrementing sequence
                start = st.number_input("Start value", 1, config['vocab_size']-1, 1)
                seq = [(start + i) % config['vocab_size'] for i in range(config['seq_len'])]
                st.session_state.test_input = torch.tensor([seq])
                
            elif pattern_type == "Alternating":
                # Alternating pattern
                a, b = st.slider("First value", 1, config['vocab_size']-1, 1), \
                       st.slider("Second value", 1, config['vocab_size']-1, 2)
                seq = [a if i % 2 == 0 else b for i in range(config['seq_len']) ]
                st.session_state.test_input = torch.tensor([seq])
    
    elif input_mode == "Manual Input":
        st.markdown("### Enter Custom Sequence")
        
        # Create a row of number inputs for each position
        cols = st.columns(config['seq_len'])
        seq = []
        for i in range(config['seq_len']):
            with cols[i]:
                token = st.number_input(
                    f"Pos {i}",
                    min_value=0,
                    max_value=config['vocab_size']-1,
                    value=i % config['vocab_size'],
                    key=f"token_{i}",
                    step=1
                )
                seq.append(token)
        
        if st.button("Set Sequence"):
            st.session_state.test_input = torch.tensor([seq])
    
    else:  # Random Sequence
        st.markdown("### Random Sequence")
        
        if st.button("Generate New Random Sequence"):
            st.session_state.test_input = torch.randint(1, config['vocab_size'], (1, config['seq_len']))
    
    # Display current input sequence
    st.markdown("### Current Input Sequence")
    if 'test_input' in st.session_state and st.session_state.test_input is not None:
        input_seq = st.session_state.test_input
        st.code(f"{input_seq[0].tolist()}", language="python")
        
        # Visual representation of the sequence
        st.markdown("#### Sequence Visualization")
        st.bar_chart(pd.DataFrame({"Tokens": input_seq[0].tolist()}))
    
    # Run inference button
    if st.button("Run Inference", type="primary") and 'test_input' in st.session_state:
        input_seq = st.session_state.test_input

        # Display input
        st.markdown("### Input Sequence")
        st.code(f"{input_seq[0].tolist()}", language="python")

        # Expected output for toy tasks
        if config['task'] == 'copy':
            expected = input_seq.clone()
        elif config['task'] == 'reverse':
            expected = torch.flip(input_seq, [1])
        elif config['task'] == 'next_token':
            if hasattr(st.session_state, 'dataset') and hasattr(st.session_state.dataset, 'pattern'):
                # For complex patterns, get the next token according to the pattern
                pattern = st.session_state.dataset.pattern
                input_seq_np = input_seq[0].numpy()  # Get the first sequence in batch
                
                if pattern == 'fibonacci':
                    # Next number is sum of last two numbers
                    expected = input_seq.clone()
                    for i in range(len(input_seq_np) - 1):
                        expected[0, i] = (input_seq_np[i] + input_seq_np[i-1]) % config['vocab_size'] if i > 0 else input_seq_np[1]
                    expected[0, -1] = (input_seq_np[-1] + input_seq_np[-2]) % config['vocab_size']
            # Expected output for toy tasks
            if config['task'] == 'copy':
                expected = input_seq.clone()
            elif config['task'] == 'reverse':
                expected = torch.flip(input_seq, [1])
            elif config['task'] == 'next_token':
                if hasattr(st.session_state, 'dataset') and hasattr(st.session_state.dataset, 'pattern'):
                    # For complex patterns, get the next token according to the pattern
                    pattern = st.session_state.dataset.pattern
                    input_seq_np = input_seq[0].numpy()  # Get the first sequence in batch
                    
                    if pattern == 'fibonacci':
                        # Next number is sum of last two numbers
                        expected = input_seq.clone()
                        for i in range(len(input_seq_np) - 1):
                            expected[0, i] = (input_seq_np[i] + input_seq_np[i-1]) % config['vocab_size'] if i > 0 else input_seq_np[1]
                        expected[0, -1] = (input_seq_np[-1] + input_seq_np[-2]) % config['vocab_size']
                        
                    elif pattern == 'primes':
                        # Find next prime number
                        primes = st.session_state.dataset.primes
                        expected = input_seq.clone()
                        for i in range(len(input_seq_np)):
                            current = input_seq_np[i]
                            next_p = next((p for p in primes if p > current), primes[0])
                            expected[0, i] = next_p % config['vocab_size']
                            
                    elif pattern == 'alternating':
                        # Apply alternating operations
                        expected = input_seq.clone()
                        for i in range(1, len(input_seq_np)):
                            if i % 3 == 0:
                                expected[0, i-1] = (input_seq_np[i-1] + 3) % config['vocab_size']
                            elif i % 2 == 0:
                                expected[0, i-1] = (input_seq_np[i-1] * 2) % config['vocab_size']
                            else:
                                expected[0, i-1] = max(1, (input_seq_np[i-1] - 1) % config['vocab_size'])
                        # For the last position, continue the pattern
                        if len(input_seq_np) % 3 == 0:
                            expected[0, -1] = (input_seq_np[-1] + 3) % config['vocab_size']
                        elif len(input_seq_np) % 2 == 0:
                            expected[0, -1] = (input_seq_np[-1] * 2) % config['vocab_size']
                        else:
                            expected[0, -1] = max(1, (input_seq_np[-1] - 1) % config['vocab_size'])
                            
                    elif pattern == 'multiplicative':
                        # Next is double the current value
                        expected = (input_seq * 2) % config['vocab_size']
                        expected[expected == 0] = 1  # Avoid 0s
                        
                    elif pattern == 'random_walk':
                        # Can't perfectly predict random walk, so just shift
                        expected = torch.roll(input_seq, -1, dims=1)
                        
                    elif pattern == 'skip':
                        # Skip pattern: every 3rd element follows a different pattern
                        expected = input_seq.clone()
                        for i in range(len(input_seq_np)):
                            if (i+1) % 3 == 0:
                                expected[0, i] = (input_seq_np[0] + input_seq_np[1] * (1 if (i % 2) else 0)) % config['vocab_size']
                            else:
                                expected[0, i] = input_seq_np[0] % config['vocab_size']
                    else:
                        # Default to simple shift if pattern not recognized
                        expected = torch.roll(input_seq, -1, dims=1)
                else:
                    # Fallback to simple shift if no pattern information is available
                    expected = torch.roll(input_seq, -1, dims=1)
            
            st.markdown("### Expected Output")
            st.code(f"{expected[0].tolist()}", language="python")

            # Inference
            with torch.no_grad():
                B, T = input_seq.shape
                mask = None
                if config['use_causal_mask']:
                    mask = make_causal_mask(T, T, B, config['num_heads'], device=device)
                
                logits, attn_list = model(input_seq, mask=mask, return_attn=True)
                predictions = logits.argmax(dim=-1)
            
            st.markdown("### Model Prediction")
            st.code(f"{predictions[0].cpu().tolist()}", language="python")

            # Accuracy
            correct = (predictions == expected).sum().item()
            total = expected.numel()
            accuracy = correct / total
            
            # Prediction metrics with custom styling
            st.markdown("""
            <style>
            .prediction-metric-container {
                background-color: #1a1a2e;
                border-radius: 8px;
                padding: 12px;
                margin: 5px;
                border-top: 3px solid #4f46e5;
            }
            .prediction-metric-label {
                color: #a1a1a1;
                font-size: 0.9rem;
                margin-bottom: 5px;
            }
            .prediction-metric-value {
                color: #ffffff;
                font-size: 1.2rem;
                font-weight: bold;
            }
            .status-correct {
                color: #10b981;
            }
            .status-incorrect {
                color: #ef4444;
            }
            </style>
            """, unsafe_allow_html=True)

            def prediction_metric_box(label, value, status=None):
                status_class = "status-correct" if status == "✅" else "status-incorrect" if status else ""
                value_class = f'class="{status_class}"' if status else ""
                return f"""
                <div class="prediction-metric-container">
                    <div class="prediction-metric-label">{label}</div>
                    <div {value_class} class="prediction-metric-value">{value}</div>
                </div>
                """

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(prediction_metric_box("Correct Tokens", f"{correct}/{total}"), unsafe_allow_html=True)
            with col2:
                st.markdown(prediction_metric_box("Accuracy", f"{accuracy:.2%}"), unsafe_allow_html=True)
            with col3:
                match_status = "✅ Perfect Match!" if accuracy == 1.0 else "❌ Some Errors"
                st.markdown(prediction_metric_box("Status", match_status, "✅" if accuracy == 1.0 else "❌"), unsafe_allow_html=True)
            
            # Token-by-token comparison
            st.markdown("### Token-by-Token Comparison")
            comparison_df = pd.DataFrame({
                'Position': range(config['seq_len']),
                'Input': input_seq[0].tolist(),
                'Expected': expected[0].tolist(),
                'Predicted': predictions[0].tolist(),
                'Correct': ['✅' if predictions[0, i] == expected[0, i] else '❌' 
                           for i in range(config['seq_len'])]
            })
            st.dataframe(comparison_df, use_container_width=True)
            
            # Store attention for visualization tab
            st.session_state.last_attention = attn_list
            st.session_state.last_input = input_seq

# ==================== TAB 4: VISUALIZATION ====================
with tab4:
    st.header("Attention Visualization")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please build a model first!")
    elif 'last_attention' not in st.session_state:
        st.info("ℹ️ Run a test in the 'Testing' tab first to generate attention patterns!")
    else:
        config = st.session_state.model_config
        attn_list = st.session_state.last_attention
        input_seq = st.session_state.last_input
        
        st.markdown(f"**Visualizing attention for input:** `{input_seq[0].tolist()}`")
        
        # Layer selection
        layer_idx = st.slider("Select Layer", 0, config['num_layers']-1, 0)
        
        # Head visualization mode
        viz_mode = st.radio("Visualization Mode:", ["All Heads (Grid)", "Single Head", "Average Across Heads"], horizontal=True)
        
        attn = attn_list[layer_idx][0].cpu().numpy()  # [H, T, T]
        H, T, _ = attn.shape
        
        if viz_mode == "All Heads (Grid)":
            # Plot all heads
            n_cols = min(4, H)
            n_rows = (H + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
            axes = np.array(axes).reshape(-1) if isinstance(axes, np.ndarray) else [axes]
            
            for h in range(H):
                ax = axes[h]
                im = ax.imshow(attn[h], cmap='viridis', aspect='auto', vmin=0, vmax=1)
                ax.set_title(f'Head {h}', fontsize=12, fontweight='bold')
                ax.set_xlabel('Key Position', fontsize=10)
                ax.set_ylabel('Query Position', fontsize=10)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # Hide extra axes
            for h in range(H, len(axes)):
                axes[h].axis('off')
            
            plt.suptitle(f'Layer {layer_idx} - All Attention Heads', fontsize=16, fontweight='bold', y=1.00)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        elif viz_mode == "Single Head":
            head_idx = st.slider("Select Head", 0, H-1, 0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(attn[head_idx], cmap='viridis', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Key Position', fontsize=12)
            ax.set_ylabel('Query Position', fontsize=12)
            
            # Add token labels
            token_labels = [f"{i}\n({input_seq[0, i].item()})" for i in range(T)]
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(token_labels, fontsize=9)
            ax.set_yticklabels(token_labels, fontsize=9)
            
            plt.colorbar(im, ax=ax, label='Attention Weight')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Show attention matrix values
            if st.checkbox("Show attention values"):
                df = pd.DataFrame(
                    attn[head_idx].round(3),
                    columns=[f"K{i}" for i in range(T)],
                    index=[f"Q{i}" for i in range(T)]
                )
                st.dataframe(df, use_container_width=True)
        
        else:  # Average across heads
            attn_avg = attn.mean(axis=0)  # Average over heads
            
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(attn_avg, cmap='viridis', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Layer {layer_idx} - Average Attention (across {H} heads)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Key Position', fontsize=12)
            ax.set_ylabel('Query Position', fontsize=12)
            
            token_labels = [f"{i}\n({input_seq[0, i].item()})" for i in range(T)]
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(token_labels, fontsize=9)
            ax.set_yticklabels(token_labels, fontsize=9)
            
            plt.colorbar(im, ax=ax, label='Attention Weight')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        # Layer-by-layer comparison
        st.markdown("---")
        if st.checkbox("Compare Attention Across All Layers"):
            st.subheader("Attention Patterns Across Layers")
            
            head_to_compare = st.slider("Head to compare", 0, H-1, 0)
            
            fig, axes = plt.subplots(1, config['num_layers'], figsize=(5*config['num_layers'], 4))
            if config['num_layers'] == 1:
                axes = [axes]
            
            for layer in range(config['num_layers']):
                attn_layer = attn_list[layer][0, head_to_compare].cpu().numpy()
                im = axes[layer].imshow(attn_layer, cmap='viridis', aspect='auto', vmin=0, vmax=1)
                axes[layer].set_title(f'Layer {layer}', fontsize=12, fontweight='bold')
                axes[layer].set_xlabel('Keys', fontsize=10)
                axes[layer].set_ylabel('Queries', fontsize=10)
                plt.colorbar(im, ax=axes[layer], fraction=0.046, pad=0.04)
            
            plt.suptitle(f'Head {head_to_compare} Across All Layers', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# ==================== TAB 5: TUTORIAL ====================
with tab5:
    st.header("📚 Tutorial & Documentation")
    
    st.markdown("""## How to Use This App

### Step 1: Configure Model ⚙️
1. Go to the **Model Config** tab
2. Choose your architecture parameters:
   - `vocab_size`: How many unique tokens (e.g., 20)
   - `d_model`: Model dimension (e.g., 64)
   - `num_heads`: Number of attention heads (e.g., 4)
   - `num_layers`: Stack depth (e.g., 2)
3. Choose training settings:
   - **Task**: 
     - `copy`: Copy the input sequence
     - `reverse`: Reverse the input sequence
     - `next_token`: Predict next token in sequence
   - **Pattern Type** (for next_token task):
     - `fibonacci`: Generate Fibonacci sequence
     - `primes`: Sequence of prime numbers
     - `alternating`: Alternating between two values
     - `multiplicative`: Multiplicative patterns
     - `random_walk`: Random walk sequence
     - `skip`: Skip patterns
   - **Sequence length**: Length of input/output sequences
   - **Batch size**, **learning rate**, **epochs**
4. Click **🏗️ Build Model**

### Testing Your Model 🧪

In the **Testing** tab, you can evaluate your model using different input modes:

1. **Preset Patterns**:
   - **Fibonacci**: Generates a Fibonacci sequence
     - Example: [1, 1, 2, 3, 5, 8, 13, 21] → [1, 2, 3, 5, 8, 13, 21, 34]
   - **Primes**: Sequence of prime numbers
     - Example: [2, 3, 5, 7, 11, 13, 17, 19] → [3, 5, 7, 11, 13, 17, 19, 23]
   - **Incrementing**: Simple counting sequence
     - Example: [1, 2, 3, 4, 5] → [2, 3, 4, 5, 6]
   - **Alternating**: Alternates between two values
     - Example: [1, 2, 1, 2, 1] → [2, 1, 2, 1, 2]

2. **Manual Input**:
   - Directly enter token values for each position
   - Useful for testing specific sequences
   - Each token must be within vocabulary range (0 to vocab_size-1)

3. **Random Sequence**:
   - Generates a completely random sequence
   - Good for quick testing with varied inputs

### Understanding the Output

For each test sequence, you'll see:
- **Input Sequence**: The original sequence fed to the model
- **Model Prediction**: The model's output sequence
- **Expected Output**: What the model should predict
- **Accuracy**: Percentage of correct predictions
- **Visualization**: Bar chart of the input sequence

### Pattern-Specific Notes

- **Fibonacci**:
  - Starts with [1, 1, ...]
  - Each number is sum of previous two
  - Wraps around using modulo vocab_size

- **Primes**:
  - Sequence of prime numbers
  - Wraps around using modulo vocab_size
  - Good for testing number theory understanding

- **Alternating**:
  - Toggle between two values
  - Tests pattern recognition

- **Incrementing**:
  - Simple counting sequence
  - Tests basic sequential understanding

### Interpreting Results

1. **Copy Task**:
   - Model should reproduce input exactly
   - Look for perfect diagonal in attention maps

2. **Reverse Task**:
   - Output should be input in reverse order
   - Check for anti-diagonal patterns in attention

3. **Next Token**:
   - Each output token should be the next in the sequence
   - Pattern determines the exact relationship
   - Check if model learned the underlying rule

### Tips for Testing

1. Start with simple patterns (incrementing) before moving to complex ones
2. Try different sequence lengths to test generalization
3. Compare attention patterns across different input types
4. For next_token, verify the model learns the pattern, not just memorization
5. Use manual input to test edge cases""")

# Sidebar: Model download
if st.session_state.model is not None:
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Export Model")
    
    if st.sidebar.button("Download Model", use_container_width=True):
        # Save model to bytes
        buffer = BytesIO()
        torch.save({
            'model_state_dict': st.session_state.model.state_dict(),
            'config': st.session_state.model_config,
            'training_history': st.session_state.training_history
        }, buffer)
        buffer.seek(0)
        
        st.sidebar.download_button(
            label="📥 Download .pt file",
            data=buffer,
            file_name=f"mini_transformer_{config['task']}.pt",
            mime="application/octet-stream",
            use_container_width=True
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.9em;">
    <b>Mini-Transformer Playground v1.0</b><br>
    User: {st.session_state.get('user', '23f2002284')}<br>
    Session: {time.strftime('%Y-%m-%d %H:%M UTC')}
</div>
""", unsafe_allow_html=True)