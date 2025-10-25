import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import os
from typing import List, Dict, Any, Optional, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name: str="Qwen/Qwen3-VL-2B-Instruct", device: torch.device = device) -> Tuple[Qwen3VLForConditionalGeneration, Any]:
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def prepare_inputs(
    image_path: str, 
    prompt: str,
    model = None,
    processor= None
    )->Dict[str, Any]:
    """

    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    if model is None or processor is None:
        model, processor = load_model()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                     "text": prompt
                },
            ],
        }
    ]


    
    image_input, _ = process_vision_info(messages)

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # NOTE: use return_tensors (plural)
    inputs = processor(
        text=text,
        images=image_input,
        return_tensors="pt"
    )

    # Move tensors to device safely
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
    return inputs
    

def generate_text(
    inputs: Dict[str, Any],
    model: Optional[Qwen3VLForConditionalGeneration] = None,
    processor: Optional[Any] = None,
    max_new_tokens:int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    num_return_sequences: int = 1
) -> List[str]:
    if model is None or processor is None:
        model, processor = load_model()

    # Now call generate with keyword arguments (unpack the dict)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,           # Enable sampling
            temperature=temperature,          # Controls randomness: lower means more deterministic
            top_p=top_p,                # Nucleus sampling: 0.9 means top 90% of probability mass
            top_k=top_k,                 # Top-k sampling: consider top 50 tokens
            num_return_sequences=num_return_sequences,    # Number of sequences to generate
            pad_token_id=processor.tokenizer.eos_token_id  # Set pad token
        )

    # Trim the prompt (assumes 'input_ids' present in inputs)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], outputs)
    ]

    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts