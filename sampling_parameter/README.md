# Qwen-VL Image Captioning with Advanced Sampling Parameters

## Overview
This project demonstrates the use of different text generation sampling parameters with the Qwen-VL (Vision-Language) model. It includes both a Python API and an interactive Streamlit web interface for experimenting with various generation settings.

## Concepts

### Text Generation Sampling Parameters
1. **Temperature**
   - Controls the randomness of predictions by scaling the logits before applying softmax
   - Lower values (closer to 0) make the output more deterministic and focused
   - Higher values (closer to 1) increase randomness and creativity

2. **Top-k Sampling**
   - Limits the sampling pool to the top-k most likely next tokens
   - Reduces the chance of sampling low-probability tokens
   - Helps prevent nonsensical outputs while maintaining some randomness

3. **Top-p (Nucleus) Sampling**
   - Samples from the smallest set of tokens whose cumulative probability exceeds p
   - Dynamically adjusts the number of tokens considered
   - More flexible than top-k as it adapts to the distribution shape

4. **Beam Search vs. Sampling**
   - Beam search maintains multiple sequences and expands the most promising ones
   - Sampling generates text by randomly selecting from the predicted probability distribution
   - This implementation focuses on sampling-based approaches for more diverse outputs

## Project Structure

```
sampling_parameter/
├── README.md                 # This file
├── inference.py              # Basic inference implementation
├── inference_optimized.py    # Optimized implementation with quantization ( under working )
├── sampling_dashboard/       # Streamlit web interface
    ├── app.py               # Basic Streamlit app
    └── app_optimized.py     # Optimized Streamlit app with model caching ( under working )
```

## Features

- **Optimized Model Loading**
  - Supports 4-bit and 8-bit quantization for reduced memory usage
  - Automatic fallback to CPU if GPU is unavailable
  - Efficient attention mechanisms (FlashAttention-2 when available)

- **Flexible Generation**
  - Adjustable temperature, top-k, and top-p parameters
  - Support for multiple return sequences
  - Configurable maximum token length

- **Interactive Web Interface**
  - Real-time parameter adjustment
  - Sidebar controls for generation settings
  - Responsive layout for different screen sizes

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/23f2002284/Practical-Notes-on-LMs.git
   cd sampling_parameter
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Streamlit App

1. Navigate to the project directory:
   ```bash
   cd sampling_dashboard
   ```

2. Run the optimized Streamlit app:
   ```bash
   streamlit run app.py
   ```
   or
   ```bash
   streamlit run app_optimized.py
   ```

3. The app will open in your default web browser. If it doesn't, navigate to `http://localhost:8501`

## Usage

1. **Using the Streamlit Interface**
   - Upload an image or use the sample image
   - Adjust the sampling parameters using the sidebar sliders
   - Enter a prompt in the text area
   - Click "Generate Caption" to see the results

2. **Using the Python API**
   ```python
   from inference_optimized import load_model, prepare_inputs, generate_text
   
   # Load model (automatically uses GPU if available)
   model, processor = load_model()
   
   # Prepare inputs
   inputs = prepare_inputs(
       image_path="path/to/your/image.jpg",
       prompt="Describe this image in detail:",
       model=model,
       processor=processor
   )
   
   # Generate text with custom parameters
   captions = generate_text(
       inputs,
       model=model,
       processor=processor,
       temperature=0.7,
       top_p=0.9,
       top_k=50,
       max_new_tokens=100,
       num_return_sequences=1
   )
   
   print(captions[0])
   ```

## Best Practices

1. **For Factual Outputs**
   - Use lower temperature (0.3-0.7)
   - Higher top-p values (0.8-0.95)
   - Moderate top-k (20-50)

2. **For Creative Outputs**
   - Higher temperature (0.7-1.0)
   - Slightly lower top-p (0.7-0.9)
   - Higher top-k (50-100)

3. **Performance Optimization**
   - Use the optimized implementation for better performance
   - Enable quantization to reduce memory usage
   - Consider using a GPU for faster inference

## Troubleshooting

- **CUDA Out of Memory**: Try reducing `max_gpu_mem_gb` or enable quantization
- **Slow Performance**: Ensure you're using GPU acceleration if available
- **Installation Issues**: Make sure all dependencies are installed correctly


## Acknowledgments
- Qwen-VL model and its contributors
- Hugging Face Transformers library
- Streamlit for the web interface