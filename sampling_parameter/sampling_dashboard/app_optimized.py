import streamlit as st
import os
from PIL import Image
import sys
from pathlib import Path
from time import perf_counter

# Add parent directory to path to import inference
sys.path.append(str(Path(__file__).parent.parent))
from inference_optimized import load_model, prepare_inputs, generate_text

st.set_page_config(
    page_title="Image Captioning with Qwen-VL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        padding: 2rem;
    }
    .generated-text {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
        border-left: 4px solid #4a90e2;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=True)
def get_model_and_processor():
    # Try GPU 4-bit; if it fails (OOM/missing bnb), fall back to CPU
    try:
        return load_model(prefer_gpu=True, quantization="4bit", max_gpu_mem_gb=3.5)
    except Exception as e:
        st.warning(f"Falling back to CPU: {e}")
        return load_model(prefer_gpu=False, quantization=None)

# Initialize session state
if 'model' not in st.session_state:
    with st.spinner('Loading model... This may take a moment.'):
        st.session_state.model, st.session_state.processor = get_model_and_processor()

# Sidebar for parameters
with st.sidebar:
    st.title("🛠️ Generation Parameters")

    st.subheader("Sampling")
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    col1, col2 = st.columns(2)
    with col1:
        top_p = st.slider("Top-p (nucleus)", 0.1, 1.0, 0.9, 0.05)
    with col2:
        top_k = st.slider("Top-k", 1, 100, 20, 1)
    max_tokens = st.slider("Max Tokens", 16, 256, 96, 16)
    num_sequences = st.slider("Number of Sequences", 1, 3, 1, 1)
    greedy = st.checkbox("Greedy decoding (faster, deterministic)", value=False, help="Disable sampling for slightly faster decoding")

# Main content
st.title("🖼️ Image Captioning with Qwen-VL")
st.markdown("Generate captions for your images with settings optimized for small GPUs.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    temp_image_dir = "temp"
    os.makedirs(temp_image_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_image_dir, uploaded_file.name)

    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Text input for prompt
    prompt = st.text_area(
        "Prompt (optional)",
        value="Describe this image in detail.",
        help="Customize the prompt to guide the model's generation"
    )

    # Generate button
    if st.button("Generate Caption", type="primary"):
        with st.spinner("Generating captions..."):
            try:
                t0 = perf_counter()
                inputs = prepare_inputs(
                    image_path=temp_image_path,
                    prompt=prompt,
                    model=st.session_state.model,
                    processor=st.session_state.processor
                )

                captions = generate_text(
                    inputs=inputs,
                    model=st.session_state.model,
                    processor=st.session_state.processor,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_sequences,
                    do_sample=not greedy,
                )
                dt = perf_counter() - t0

                st.caption(f"Inference time: {dt:.2f} s")
                st.subheader("Generated Captions")
                for i, caption in enumerate(captions, 1):
                    st.markdown(f"<div class='generated-text'><strong>Caption {i}:</strong> {caption}</div>", 
                               unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    # Clean up temporary file
    try:
        os.remove(temp_image_path)
    except Exception:
        pass
else:
    st.info("Please upload an image to get started.")

# Add some instructions
expander = st.expander("ℹ️ How to use this tool")
with expander:
    st.markdown("""
    ### Image Captioning with Qwen-VL

    1. Upload an image using the file uploader
    2. (Optional) Modify the prompt to guide the generation
    3. Adjust the generation parameters in the sidebar
    4. Click "Generate Caption" to create captions

    ### Tips for faster inference on small GPUs (e.g., 4 GB)
    - Use 4-bit quantization (enabled by default here).
    - Keep Max Tokens <= 96 and Number of Sequences = 1.
    - Use Greedy decoding when you don’t need diverse outputs.
    - Prefer smaller images; very high-resolution inputs slow the vision encoder.
    """)