# LLM Engineering Projects
Inspired by [TheAhmadOsman's post](https://x.com/TheAhmadOsman/status/1975783287961313362)

## 1. Tokenization & Embeddings
- [x] Build a byte-pair encoder and train your own subword vocabulary
- [ ] Create a token visualizer to map words/chunks to their corresponding token IDs
- [ ] Compare one-hot encoding vs learned embeddings by plotting cosine distances

## 2. Positional Embeddings
- [ ] Implement and compare different positional encoding methods:
  - Classic sinusoidal
  - Learned positional embeddings
  - RoPE (Rotary Positional Embeddings)
  - ALiBi (Attention with Linear Biases)
- [ ] Create a 3D animation showing how a toy sequence is position-encoded
- [ ] Perform ablation studies by removing positional encodings and observe attention collapse

## 3. Self-Attention & Multihead Attention
- [ ] Implement dot-product attention for a single token
- [ ] Scale to multi-head attention and visualize attention weight heatmaps per head
- [ ] Implement causal masking and verify the auto-regressive property

## 4. Transformers, QKV, & Stacking
- [ ] Build a single transformer block combining:
  - Multi-head attention
  - Layer normalization
  - Residual connections
  - Feed-forward network
- [ ] Scale to an n-block mini-transformer on toy data
- [ ] Experiment with Q, K, V matrices: swap them, break them, and observe the effects

## 5. Sampling Parameters: Temperature/Top-k/Top-p
- [ ] Create an interactive sampling dashboard to tune parameters:
  - Temperature
  - Top-k
  - Top-p (nucleus sampling)
- [ ] Plot entropy vs. output diversity across different parameter settings
- [ ] Test with temperature=0 (argmax sampling) and observe repetition patterns

## 6. KV Cache (Fast Inference)
- [ ] Implement KV state caching and measure inference speedup
- [ ] Build a visualizer to track cache hit/miss rates during token generation
- [ ] Profile memory usage for different sequence lengths

## 7. Long-Context Tricks
- [ ] Implement sliding window attention and evaluate on long documents
- [ ] Benchmark different attention variants:
  - Full attention
  - Memory-efficient attention
  - Flash attention
- [ ] Plot perplexity vs. context length to identify context collapse points

## 8. Mixture of Experts (MoE)
- [ ] Implement a 2-expert routing mechanism
- [ ] Analyze expert utilization patterns across different datasets
- [ ] Simulate sparse vs. dense expert routing and measure FLOP efficiency

## 9. Grouped Query Attention
- [ ] Modify a transformer to use grouped query attention
- [ ] Benchmark performance against vanilla multi-head attention
- [ ] Study the impact of different group sizes on model performance

## 10. Normalization & Activations
- [ ] Implement and compare:
  - LayerNorm
  - RMSNorm
  - SwiGLU
  - GELU
- [ ] Perform ablation studies and track training dynamics
- [ ] Visualize activation distributions across network depth

## 11. Pretraining Objectives
- [ ] Compare different objectives:
  - Masked Language Modeling (MLM)
  - Causal Language Modeling (CLM)
  - Prefix Language Modeling
- [ ] Analyze learning curves and sample quality
- [ ] Document model behaviors specific to each objective

## 12. Finetuning vs Instruction Tuning vs RLHF
- [ ] Fine-tune a base model on a custom dataset
- [ ] Implement instruction tuning using task prefixes
- [ ] Set up a basic RLHF pipeline:
  - Train a reward model
  - Implement PPO for policy optimization
  - Track reward improvements

## 13. Scaling Laws & Model Capacity
- [ ] Train models of varying sizes
- [ ] Profile:
  - Training time
  - VRAM usage
  - Throughput
- [ ] Analyze the relationship between model size and performance

## 14. Quantization
- [ ] Implement and compare:
  - Post-Training Quantization (PTQ)
  - Quantization-Aware Training (QAT)
- [ ] Export models to efficient formats (GGUF/AWQ)
- [ ] Measure accuracy vs. model size trade-offs

## 15. Inference/Training Stacks
- [ ] Port models between frameworks:
  - HuggingFace Transformers
  - DeepSpeed
  - vLLM
  - ExLlama
- [ ] Benchmark:
  - Throughput
  - Memory usage
  - Latency

## 16. Synthetic Data
- [ ] Generate synthetic datasets
- [ ] Implement data cleaning and deduplication
- [ ] Compare model performance on real vs. synthetic data

## Development Principles
- Focus on one core insight per project
- Build → Plot → Analyze → Iterate
- Document findings and visualizations
- Share learnings with the community