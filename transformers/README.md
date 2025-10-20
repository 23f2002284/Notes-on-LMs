# Single Transformer
x → MultiHeadAttention → residual → LayerNorm → 
→ FeedForward → residual → LayerNorm → output
- Original Transformer paper implementation
- Can be harder to train
- Included for completeness

## Key Features

- **Modular Design**: Each component is cleanly separated
- **Type Hints**: Full Python type annotations
- **Documentation**: Detailed docstrings and comments
- **Configurable**:
  - Hidden dimension (d_model)
  - Feed-forward dimension (d_ff)
  - Number of attention heads
  - Dropout rates
  - Activation functions
  - Layer normalization epsilon

## Usage Example

```python
# Initialize a pre-norm transformer block
block = TransformerBlock(
    d_model=512,
    num_heads=8,
    d_ff=2048,
    dropout=0.1,
    activation="gelu"
)

# Forward pass
x = torch.randn(32, 100, 512)  # [batch_size, seq_len, d_model]
output = block(x)
```

### Key Concepts Explained
1. Multi-Head Attention
- Splits the input into multiple "heads" to attend to different aspects
- Each head computes attention independently
- Outputs are concatenated and projected back to d_model
2. Layer Normalization
- Normalizes across the feature dimension
- Helps stabilize training
- Pre-norm vs post-norm affects gradient flow
3. Residual Connections
- Helps prevent vanishing gradients
- Enables training of very deep networks
- Added before layer norm in pre-norm architecture
### Best Practices
- Prefer Pre-Norm for better stability
- Learning Rate Warmup helps with training
- Gradient Clipping prevents exploding gradients (yet to implement)
- Learning Rate Scheduling improves convergence(yet to implement)

# Mini n-block Transformer

A complete transformer model that stacks multiple transformer blocks for sequence-to-sequence tasks. Built with educational clarity and practical applications in mind.

## Key Features

- **Flexible Architecture**: Stack any number of transformer blocks
- **Multiple Positional Encodings**:
  - Sinusoidal (fixed)
  - Learned (trainable)
- **Weight Tying**: Optional sharing of input/output embeddings
- **Configurable**:
  - Vocabulary size
  - Model dimension (d_model)
  - Number of attention heads
  - Feed-forward dimension (d_ff)
  - Number of transformer blocks
  - Dropout rates
  - Maximum sequence length

## Architecture Overview
```python
# Initialize a mini-transformer
model = MiniTransformer(
    vocab_size=10000,
    d_model=512,
    num_heads=8,
    d_ff=2048,
    num_layers=6,  # Number of transformer blocks
    max_seq_len=512,
    dropout=0.1,
    pos_encoding='sinusoidal',  # or 'learned'
    tie_weights=True  # Share input/output embeddings
)

# Forward pass
x = torch.randint(0, 10000, (32, 100))  # [batch_size, seq_len]
logits = model(x)  # [batch_size, seq_len, vocab_size]
```
### Key Components
1. Token Embeddings
- Maps discrete tokens to continuous vectors
- Scaled by √d_model for stable training
2. Positional Encoding
- Sinusoidal: Fixed, non-trainable patterns
- Learned: Trainable position embeddings
3. Transformer Blocks
- Stack of N identical layers
- Each with multi-head attention and feed-forward network
- residual connections and layer normalization
4. Output Projection
- Maps back to vocabulary space
- Optional weight tying with input embeddings
### Best Practices
- Initialization: Uses Xavier/Glorot uniform by default
- Dropout: Applied to attention and feed-forward layers
- Gradient Flow: Pre-norm architecture for stable training
- Memory Efficiency: Attention masking for variable-length sequences


## How to Use This App

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
5. Use manual input to test edge cases