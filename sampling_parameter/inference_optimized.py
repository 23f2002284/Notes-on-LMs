import os
from typing import List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False

from qwen_vl_utils import process_vision_info

DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct-FP8"

def _select_attn_impl():
    # Prefer FlashAttention-2 if available, else SDPA, else eager
    attn_impl = "eager"
    try:
        from transformers.utils import is_flash_attn_2_available
        if is_flash_attn_2_available():
            attn_impl = "flash_attention_2"
        elif torch.backends.cuda.sdp_kernel.is_available():
            attn_impl = "sdpa"
    except Exception:
        if torch.cuda.is_available():
            attn_impl = "sdpa"
    return attn_impl

def load_model(
    model_name: str = DEFAULT_MODEL_ID,
    prefer_gpu: bool = True,
    quantization: Optional[str] = "4bit",  # "4bit" | "8bit" | None
    max_gpu_mem_gb: float = 3.5,
) -> Tuple[Qwen3VLForConditionalGeneration, Any]:
    use_cuda = prefer_gpu and torch.cuda.is_available()
    attn_impl = _select_attn_impl()

    kwargs = dict(
        low_cpu_mem_usage=True,
    )

    if use_cuda:
        # Ampere (RTX 3050) supports BF16
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.get_device_properties(0).major >= 8 else torch.float16
        kwargs["attn_implementation"] = attn_impl
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: f"{max_gpu_mem_gb}GiB", "cpu": "48GiB"}
        if quantization in {"4bit", "8bit"} and _HAS_BNB:
            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["quantization_config"] = bnb_config
        # else: fall back to full precision on GPU (likely OOM on 4GB)
    else:
        kwargs["torch_dtype"] = torch.float32  # CPU

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        **kwargs,
    )
    model.eval()

    # CPU dynamic quantization (best-effort) for some speed if no GPU
    if not use_cuda:
        try:
            model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        except Exception:
            pass

    processor = AutoProcessor.from_pretrained(model_name)

    # Torch performance knobs
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    return model, processor

def prepare_inputs(
    image_path: str,
    prompt: str,
    model=None,
    processor=None,
) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")

    if model is None or processor is None:
        model, processor = load_model()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    image_input, _ = process_vision_info(messages)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=text,
        images=image_input,
        return_tensors="pt",
    )

    # With device_map="auto", keep inputs on CPU so Accelerate can dispatch.
    if not hasattr(model, "hf_device_map"):
        device = next(model.parameters()).device
        inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    return inputs

@torch.inference_mode()
def generate_text(
    inputs: Dict[str, Any],
    model: Optional[Qwen3VLForConditionalGeneration] = None,
    processor: Optional[Any] = None,
    max_new_tokens: int = 96,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 20,
    num_return_sequences: int = 1,
    do_sample: bool = True,
) -> List[str]:
    if model is None or processor is None:
        model, processor = load_model()

    kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_return_sequences=num_return_sequences,
        pad_token_id=processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
    )

    use_cuda = torch.cuda.is_available() and any(p.is_cuda for p in model.parameters())
    if use_cuda:
        dtype = torch.bfloat16 if any(p.dtype == torch.bfloat16 for p in model.parameters()) else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            outputs = model.generate(**inputs, **kwargs)
    else:
        outputs = model.generate(**inputs, **kwargs)

    input_ids = inputs.get("input_ids")
    if input_ids is not None:
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, outputs)]
    else:
        generated_ids_trimmed = outputs

    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts