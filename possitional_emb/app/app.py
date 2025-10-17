import json
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import torch

from tasks import (
    make_modular_sum_task,
    make_k_back_lm_task,
    make_shift_invariant_pattern_task,
    make_long_range_match_task,
)
from experiments import TrainConfig, run_experiments


st.set_page_config(page_title="Positional Encodings: Comparative Sandbox", layout="wide")


@st.cache_data(show_spinner=False)
def device_options():
    gpu = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    opts = ["cpu"]
    if gpu:
        opts.append("cuda")
    elif mps:
        opts.append("mps")
    return opts


def sidebar_controls():
    st.sidebar.header("Configuration")

    # Task selection
    task_name = st.sidebar.selectbox(
        "Task",
        [
            "Modular Sum (Classification, extrapolation)",
            "k-back LM (Locality)",
            "Shift-Invariant Pattern (Relative)",
            "Long-Range Match (First==Last)",
        ],
        index=0,
    )

    # Encodings
    encodings = st.sidebar.multiselect(
        "Positional encodings to compare",
        ["sinusoidal", "learned", "rope", "alibi"],
        default=["sinusoidal", "learned", "rope", "alibi"],
    )

    # Basic hyperparameters
    if task_name.startswith("k-back"):
        k_back = st.sidebar.slider("k (steps back)", min_value=1, max_value=10, value=3)
    else:
        k_back = None
    steps = st.sidebar.slider("Training steps", min_value=50, max_value=2000, value=300, step=50)
    batch_size = st.sidebar.slider("Batch size", min_value=16, max_value=256, value=64, step=16)
    d_model = st.sidebar.select_slider("Model dim (d_model)", options=[64, 128, 192, 256], value=128)
    n_heads = st.sidebar.select_slider("Num heads", options=[2, 4, 8], value=4)
    n_layers = st.sidebar.select_slider("Layers", options=[1, 2, 3, 4], value=2)
    dropout = st.sidebar.slider("Dropout", min_value=0.0, max_value=0.4, value=0.1, step=0.05)
    lr = st.sidebar.select_slider("Learning rate", options=[1e-4, 3e-4, 5e-4, 1e-3], value=3e-4)
    device = st.sidebar.selectbox("Device", device_options(), index=0)
    seed = st.sidebar.number_input("Seed", min_value=0, max_value=10_000, value=1337, step=1)

    st.sidebar.caption("Tip: Increase steps for clearer separation; defaults are CPU-friendly.")

    return {
        "task_name": task_name,
        "encodings": encodings,
        "steps": steps,
        "batch_size": batch_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "lr": float(lr),
        "device": device,
        "seed": seed,
        "k_back": k_back,
    }


def build_task(task_name: str):
    if task_name.startswith("Modular"):
        spec = make_modular_sum_task(vocab_size=20, train_seq_len=64, val_seq_len=64, long_seq_len=256)
        meta = {
            "headline": "Extrapolation to longer sequences (train 64, test 256)",
            "expect": "Absolute learned often degrades OOD; sinusoidal/RoPE/ALiBi typically generalize better.",
        }
        k_back = None
    elif task_name.startswith("k-back"):
        spec, k_back = make_k_back_lm_task(vocab_size=50, k_back=3, train_seq_len=64, val_seq_len=64, long_seq_len=128)
        meta = {
            "headline": "Locality bias for short-range dependencies",
            "expect": "ALiBi's recency bias can help; others may need more steps to match.",
        }
    elif task_name.startswith("Shift-Invariant"):
        spec = make_shift_invariant_pattern_task(vocab_size=50, train_seq_len=64, val_seq_len=64, long_seq_len=256)
        meta = {
            "headline": "Relative position generalization (train near start, test near end)",
            "expect": "RoPE/ALiBi often transfer better; absolute may overfit to positions.",
        }
        k_back = None
    else:
        spec = make_long_range_match_task(vocab_size=50, train_seq_len=64, val_seq_len=64, long_seq_len=256)
        meta = {
            "headline": "Long-range dependency (first vs last)",
            "expect": "Locality bias can hurt; RoPE/sinusoidal may hold up better.",
        }
        k_back = None
    return spec, meta, k_back


def cfg_to_key(cfg: Dict) -> str:
    # Deterministic string for comparing UI config dictionaries
    return json.dumps(cfg, sort_keys=True)


def init_state():
    defaults = {
        "results": None,
        "last_cfg": None,
        "last_cfg_key": None,
        "last_task_name": None,
        "last_meta": None,
        "last_k_back": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_results(results: Dict, label_prefix: str = ""):
    if not results:
        return

    # Summaries table
    rows = []
    for enc, res in results.items():
        rows.append(
            {
                "encoding": enc,
                "in_loss": round(res["in_dist"]["loss"], 4),
                "in_acc": round(res["in_dist"]["acc"], 4),
                "out_loss": round(res["out_dist"]["loss"], 4),
                "out_acc": round(res["out_dist"]["acc"], 4),
            }
        )
    st.subheader(f"{label_prefix}Summary (In-Distribution vs Out-of-Distribution/Longer Sequences)")
    st.dataframe(pd.DataFrame(rows).set_index("encoding"))

    # Learning curves
    st.subheader(f"{label_prefix}Learning curves (validation)")
    curve_df = []
    for enc, res in results.items():
        val_acc = res["train_logs"]["val_acc"]
        xs = list(range(1, len(val_acc) + 1))
        for i, a in enumerate(val_acc):
            curve_df.append({"encoding": enc, "checkpoint": i, "val_acc": a})
    curve_df = pd.DataFrame(curve_df)
    st.line_chart(curve_df.pivot(index="checkpoint", columns="encoding", values="val_acc"))

    # Attention heatmaps for a small example
    st.subheader(f"{label_prefix}Attention maps (last layer, one example)")
    for enc, res in results.items():
        st.markdown(f"#### {enc}")
        attn = res["attn_example"]
        if attn is None:
            st.write("No attention captured.")
            continue
        H, T, _ = attn.shape
        head_idx = st.slider(f"Head for {enc}", min_value=0, max_value=H - 1, value=0, key=f"head_{enc}")
        st.caption(f"Head {head_idx}, sequence length {T}")
        st.dataframe(
            pd.DataFrame(attn[head_idx], columns=[f"t{j}" for j in range(T)]).style.background_gradient(cmap="Blues")
        )


def main():
    init_state()

    st.title("Comparing Positional Encodings: Sinusoidal vs Learned vs RoPE vs ALiBi")
    st.write(
        "Run quick, controlled experiments to see where each positional encoding shines or struggles. "
        "These synthetic tasks illustrate tradeoffs without implying a single best choice."
    )

    cfg_ui = sidebar_controls()
    spec, meta, k_back_default = build_task(cfg_ui["task_name"])
    if cfg_ui["k_back"] is not None:
        k_back = cfg_ui["k_back"]
        spec.name = f"k_back_lm_k{k_back}"
    else:
        k_back = k_back_default

    # Show current selection context
    st.subheader(spec.name)
    st.caption(meta["headline"])
    st.info(meta["expect"])

    run_btn = st.button("Run experiment")

    if run_btn:
        with st.spinner("Training small models..."):
            cfg = TrainConfig(
                steps=cfg_ui["steps"],
                batch_size=cfg_ui["batch_size"],
                lr=cfg_ui["lr"],
                device=cfg_ui["device"],
                seed=cfg_ui["seed"],
                d_model=cfg_ui["d_model"],
                n_heads=cfg_ui["n_heads"],
                n_layers=cfg_ui["n_layers"],
                d_ff=4 * cfg_ui["d_model"],
                dropout=cfg_ui["dropout"],
                max_seq_len=max(1024, spec.long_seq_len + 8),
            )
            results = run_experiments(
                positional_types=cfg_ui["encodings"],
                spec=spec,
                cfg=cfg,
                k_back=k_back,
                capture_attn_example=True,
            )

        st.success("Done.")

        # Persist results and context so UI tweaks won't drop them on rerun
        st.session_state["results"] = results
        st.session_state["last_cfg"] = dict(cfg_ui)  # shallow copy OK (primitives/lists)
        st.session_state["last_cfg_key"] = cfg_to_key(cfg_ui)
        st.session_state["last_task_name"] = spec.name
        st.session_state["last_meta"] = meta
        st.session_state["last_k_back"] = k_back

    # Always render from stored results if available
    stored_results = st.session_state.get("results")
    if stored_results:
        # Detect if sidebar config changed since last run (but keep showing prior results)
        current_key = cfg_to_key(cfg_ui)
        if st.session_state.get("last_cfg_key") and current_key != st.session_state["last_cfg_key"]:
            st.warning(
                "Sidebar configuration changed since last run. Showing results from the previous run. "
                "Click 'Run experiment' to recompute with the new settings."
            )

        # Label that these results come from the last completed run
        label_prefix = ""
        last_task = st.session_state.get("last_task_name")
        if last_task:
            st.markdown(f"##### Showing results from last run: {last_task}")

        render_results(stored_results, label_prefix=label_prefix)

        # Notes
        st.subheader("Interpretation guide")
        st.write(
            "- Sinusoidal (absolute): parameter-free and extrapolates well to longer sequences; may be less strong on purely relative generalization.\n"
            "- Learned (absolute): can excel within trained lengths/positions and memorize exact positions; often drops OOD on longer lengths since unseen position embeddings remain untrained.\n"
            "- RoPE (relative): good at relative offsets and extrapolation; widely used in modern LLMs; robust to insertions and length changes.\n"
            "- ALiBi (bias-only): adds a recency bias helping short-range dependencies; can hurt when long-range attention is required."
        )
    else:
        st.info("Configure the experiment in the sidebar, then click 'Run experiment'.")


if __name__ == "__main__":
    main()