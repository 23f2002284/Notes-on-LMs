import streamlit as st
import os
from PIL import Image
import sys
from pathlib import Path

# Add parent directory to path to import inference
sys.path.append(str(Path(__file__).parent.parent))
from inference import load_model, prepare_inputs, generate_text

# Page configuration
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
    .stSlider > div {
        width: 100%;
    }
    .stTextArea > div > div > textarea {
        min-height: 100px;
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

# Initialize session state
if 'model' not in st.session_state:
    with st.spinner('Loading model... This may take a moment.'):
        st.session_state.model, st.session_state.processor = load_model()

# Sidebar for parameters
with st.sidebar:
    st.title("🛠️ Generation Parameters")
    
    st.subheader("Sampling")
    temperature = st.slider(
        "Temperature", 
        min_value=0.1, 
        max_value=2.0, 
        value=0.7, 
        step=0.1,
        help="Controls randomness: Lower = more deterministic, Higher = more creative"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        top_p = st.slider(
            "Top-p (nucleus)", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.9, 
            step=0.05,
            help="Controls diversity via nucleus sampling: 0.5 means half of all likelihood-weighted options are considered"
        )
    
    with col2:
        top_k = st.slider(
            "Top-k", 
            min_value=1, 
            max_value=100, 
            value=50, 
            step=1,
            help="Controls diversity: 1 = greedy sampling, 50 = consider top 50 tokens"
        )
    
    max_tokens = st.slider(
        "Max Tokens", 
        min_value=32, 
        max_value=512, 
        value=256, 
        step=32,
        help="Maximum number of tokens to generate"
    )
    
    num_sequences = st.slider(
        "Number of Sequences", 
        min_value=1, 
        max_value=5, 
        value=1, 
        step=1,
        help="Number of different sequences to generate"
    )

# Main content
st.title("🖼️ Image Captioning with Qwen-VL")
st.markdown("Generate captions for your images using the Qwen-VL model with configurable sampling parameters.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Display uploaded image
if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True, width='stretch')
    
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
                # Prepare inputs
                inputs = prepare_inputs(
                    image_path=temp_image_path,
                    prompt=prompt,
                    model=st.session_state.model,
                    processor=st.session_state.processor
                )
                
                # Generate text
                captions = generate_text(
                    inputs=inputs,
                    model=st.session_state.model,
                    processor=st.session_state.processor,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_sequences
                )
                
                # Display results
                st.subheader("Generated Captions")
                for i, caption in enumerate(captions, 1):
                    st.markdown(f"<div class='generated-text'><strong>Caption {i}:</strong> {caption}</div>", 
                               unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.exception(e)
    
    # Clean up temporary file
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
else:
    st.info("Please upload an image to get started.")

# Add some instructions
expander = st.expander("ℹ️ How to use this tool")
with expander:
    st.markdown("""
    ### Image Captioning with Qwen-VL
    
    1. **Upload an image** using the file uploader
    2. (Optional) Modify the prompt to guide the generation
    3. Adjust the generation parameters in the sidebar
    4. Click "Generate Caption" to create captions
    
    ### About the Parameters
    - **Temperature**: Controls randomness (0.1-2.0)
    - **Top-p**: Controls diversity via nucleus sampling (0.1-1.0)
    - **Top-k**: Limits the model to consider only the top k tokens (1-100)
    - **Max Tokens**: Maximum length of the generated caption
    - **Number of Sequences**: How many different captions to generate
    """)
