# LLM Engineering Projects
Inspired by [TheAhmadOsman's post](https://x.com/TheAhmadOsman/status/1975783287961313362)



## 1. Tokenization & Embeddings
- [x] Build a byte-pair encoder and train your own subword vocabulary
- [X] Create a token visualizer to map words/chunks to their corresponding token IDs
- [ ] Compare one-hot encoding vs learned embeddings by plotting cosine distances

### Implementation Files
| File | Description |
|------|-------------|
| [1-1.ipynb](tokenization_n_embeddings/1-1.ipynb) | Main notebook for Task 1.1 |
| [1-1.md](tokenization_n_embeddings/1-1.md) | Documentation for Task 1.1 |
| [task_1_1_bpe.py](tokenization_n_embeddings/task_1_1_bpe.py) | Byte-pair encoding implementation |
| [task_1_2_token_mapping.ipynb](tokenization_n_embeddings/task_1_2_token_mapping.ipynb) | Token visualization implementation |
| [task_1_3_ohe_vs_learned_emb.ipynb](tokenization_n_embeddings/task_1_3_ohe_vs_learned_emb.ipynb) | One-hot vs learned embeddings comparison |


## 2. Positional Embeddings
- [x] Implement and compare different positional encoding methods:
  - Classic sinusoidal
  - Learned positional embeddings
  - RoPE (Rotary Positional Embeddings)
  - ALiBi (Attention with Linear Biases)
- [ ] Create a 3D animation showing how a toy sequence is position-encoded
- [ ] Perform ablation studies by removing positional encodings and observe attention collapse

### Implementation Files
| File | Description |
|------|-------------|
| [sinusodial_PE.py](possitional_emb/sinusodial_PE.py) | Implementation of classic sinusoidal positional encoding |
| [learned_PE.py](possitional_emb/learned_PE.py) | Implementation of learned positional embeddings |
| [RoPE.py](possitional_emb/RoPE.py) | Implementation of Rotary Positional Embeddings |
| [ALiBi_PE.py](possitional_emb/ALiBi_PE.py) | Implementation of Attention with Linear Biases |
| [toy_seq_pe_vizualization.py](possitional_emb/toy_seq_pe_vizualization.py) | 3D visualization of position encoding |
| [app](possitional_emb/app) | Streamlit app for positional encoding comparison |
| [ablation_studies.py](possitional_emb/ablation_studies.py) | Ablation studies on positional encodings |


## 3. Self-Attention & Multihead Attention
- [X] Implement dot-product attention for a single token
- [X] Scale to multi-head attention and visualize attention weight heatmaps per head
- [X] Implement causal masking and verify the auto-regressive property

### Implementation Files
| File | Description |
|------|-------------|
| [single_token_attn.py](attention/single_token_attn.py) | Basic dot-product attention implementation for a single token |
| [vizualize_multihead_attn/](attention/vizualize_multihead_attn/) | Multi-head attention visualization and implementation |
| [vizualize_casualM_autoreg/](attention/vizualize_casualM_autoreg/) | Causal masking and auto-regressive property verification |
| [vizualize_multihead_attn/app.py](attention/vizualize_multihead_attn/app.py) | Streamlit app for visualizing multi-head attention weights |
| [vizualize_casualM_autoreg/app.py](attention/vizualize_casualM_autoreg/app.py) | Interactive demo of causal masking in attention |
| [vizualize_casualM_autoreg/MultiHeadAttnViz.py](attention/vizualize_casualM_autoreg/MultiHeadAttnViz.py) | Multi-head attention implementation with causal masking |
| [vizualize_casualM_autoreg/attention_masks.py](attention/vizualize_casualM_autoreg/attention_masks.py) | Utilities for creating attention masks |



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


# Next Steps
- [ ] separate requirements.txt files for each tasks folders
- [ ] will add existing good repositories on topic
- [ ] a good readme file like a roadmap and with good file structure mapping
- [ ] write notes and q&a in md files
- [ ] Seq2Seq Model for 
  - [ ] classification
  - [ ] translation
  - [ ] summarization
  - [ ] question answering ?
  - [ ] entity extraction
  - [ ] bottleneck explanation for achilles heel
- [ ] Generative Deep learning
- [ ] Build a LLM from Scratch
- [ ] Test yourself on Build a LLM from scratch
- [ ] LLM Engineering Handbook
- [ ] GANs in Action
- [ ] NLP with Transformers models
- [ ] AI Engineering
- [ ] Reinforcement learning
- [ ] Development for interface building and deployment Practices (ops)
- [ ] Sebastian Raschka Sir's new book on Reasoning Models
- [ ] New and old Papers on LLMs development
- [ ] Upcoming GSoc Contribution and projects Notes
- [ ] it will be like a knowledge graph anyone would like to have in LLM development
- [ ] State Management: Memory (Mem0, MemGPT)

> ( soon to add more as i learn ) ⬇️

## Development Principles 
- Focus on one core insight per project
- Build → Plot → Analyze → Iterate
- Document findings and visualizations
- Share learnings with the community

## Development Practices

To maintain high code quality and consistency across the project, we adhere to the following development practices:

### Code Structure
- **Object-Oriented Design**: Leveraging Python classes for better organization and encapsulation
- **Modular Architecture**: Breaking down complex systems into reusable, single-responsibility components

### Code Quality
- **Type Annotations**: Comprehensive type hints for better code clarity and IDE support
- **Documentation**: Detailed docstrings following Google style guide for all public interfaces
- **Error Handling**: Robust exception handling with meaningful error messages
- **Code Comments**: Clear, concise comments explaining non-obvious logic and complex algorithms

### Best Practices
- **Encapsulation**: Using private methods (prefixed with `_`) for internal implementation details
- **Callable Objects**: Implementing `__call__` for classes where instance-as-function behavior makes sense
- **Value-Added Scripts**: Ensuring each script provides clear utility and can be easily reused or extended

### Development Workflow
- **Code Reviews**: All changes undergo thorough code review
- **Testing**: Comprehensive test coverage for critical functionality
- **Continuous Integration**: Automated testing and quality checks on every commit