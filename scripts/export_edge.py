import torch
import os
import argparse
import json
import platform
import statistics
import sys
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.multimodal_fusion_models import TemporalFusionTransformer
from src.core.prism_breath_model import PrismBreathConfig, build_prism_breath_model_from_config

try:
    from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False

def export_edge_model(checkpoint_path, output_dir="data/models/edge"):
    """
    Exports the TemporalFusionTransformer to TorchScript and ONNX,
    applying dynamic quantization for Edge/Mobile deployment.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading Fusion Transformer from {checkpoint_path}...")
    
    # Initialize Quad-Modal Model (no VLM by default for strict edge)
    model = TemporalFusionTransformer(num_layers=4, vlm_dim=0)
    
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu")
        if 'model' in state: state = state['model']
        
        # Remove VLM-specific keys for edge export if present
        keys_to_remove = [k for k in state.keys() if 'vlm_proj' in k]
        for k in keys_to_remove:
            del state[k]
            
        model.load_state_dict(state, strict=False)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Exporting randomly initialized model.")
        
    model.eval()
    
    # 1. Apply Dynamic Quantization (INT8)
    print("\n[1] Applying Dynamic INT8 Quantization...")
    if HAS_TORCHAO:
        print("  Using modern `torchao` quantization backend...")
        # Deepcopy to avoid modifying the original model instance in place if needed multiple times,
        # but here we can just quantize it.
        import copy
        quantized_model = copy.deepcopy(model)
        quantize_(quantized_model, int8_dynamic_activation_int8_weight())
    else:
        print("  `torchao` not found. Falling back to legacy `torch.quantization`...")
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {torch.nn.Linear, torch.nn.Conv1d, torch.nn.TransformerEncoderLayer},
            dtype=torch.qint8
        )
    
    q_out_path = os.path.join(output_dir, "multimodal_fusion_quantized.pth")
    # TorchScripting or saving the architecture
    if HAS_TORCHAO:
        # torchao quantized models are currently best saved and loaded via state_dict or Export
        torch.save(quantized_model.state_dict(), q_out_path)
    else:
        torch.save(quantized_model.state_dict(), q_out_path)
    
    # Compare sizes
    original_size = os.path.getsize(checkpoint_path) if os.path.exists(checkpoint_path) else 0
    quantized_size = os.path.getsize(q_out_path)
    
    print(f"  Saved Quantized Weights to: {q_out_path}")
    if original_size > 0:
        print(f"  Compression: {original_size / 1024 / 1024:.2f} MB -> {quantized_size / 1024 / 1024:.2f} MB")
    
    # 2. Export to TorchScript (for C++ / Mobile integration)
    print("\n[2] Exporting to TorchScript...")
    seq_len = 300
    dummy_depth = torch.randn(1, seq_len)
    dummy_flow = torch.randn(1, seq_len)
    dummy_audio = torch.randn(1, seq_len)
    
    try:
        # Trace requires concrete inputs
        traced_script_module = torch.jit.trace(quantized_model, (dummy_depth, dummy_flow, dummy_audio))
        ts_out_path = os.path.join(output_dir, "multimodal_fusion_mobile.pt")
        traced_script_module.save(ts_out_path)
        print(f"  Saved TorchScript Mobile model to: {ts_out_path}")
    except Exception as e:
        print(f"  TorchScript export failed: {e}")
        
    # 3. Export to ONNX (for generic Edge TPUs / WebAssembly)
    print("\n[3] Exporting to ONNX...")
    onnx_out_path = os.path.join(output_dir, "multimodal_fusion.onnx")
    try:
        torch.onnx.export(
            model,                                 # Normal unquantized model for ONNX
            (dummy_depth, dummy_flow, dummy_audio),# model inputs
            onnx_out_path,                         # where to save the model
            export_params=True,                    # store the trained parameter weights inside the model file
            opset_version=14,                      # the ONNX version to export the model to
            do_constant_folding=True,              # whether to execute constant folding for optimization
            input_names = ['depth_in', 'flow_in', 'audio_in'],   # the model's input names
            output_names = ['bpm_out'],            # the model's output names
            dynamic_axes={'depth_in' : {0 : 'batch_size'},    # variable length axes
                          'flow_in' : {0 : 'batch_size'},
                          'audio_in' : {0 : 'batch_size'},
                          'bpm_out' : {0 : 'batch_size'}}
        )
        print(f"  Saved ONNX model to: {onnx_out_path}")
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        
    print("\nEdge Quantization Pipeline Complete!")

# ---------------------------------------------------------------------------
# Edge latency benchmark (PRISM-Breath family, incl. mamba/cliffphys baselines)
# ---------------------------------------------------------------------------
def _count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def _fp32_size_mb(model) -> float:
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())
    bytes_total += sum(b.numel() * b.element_size() for b in model.buffers())
    return bytes_total / 1024 / 1024


def _build_prism_family_model(variant: str, seq_len: int, d_model: int, checkpoint: str = ""):
    """Build a PRISM-Breath-family model for benchmarking.

    Latency depends on architecture/size, not on trained weights, so a fresh
    init is a valid latency proxy; a checkpoint is loaded when provided so the
    exact deployed config/size is measured.
    """
    if checkpoint and os.path.exists(checkpoint):
        state = torch.load(checkpoint, map_location="cpu")
        cfg = state.get("config", {})
        model_config = cfg.get("model_config", cfg) if isinstance(cfg, dict) else {}
        model_config = dict(model_config or {})
        model_config.setdefault("model_variant", variant)
        model = build_prism_breath_model_from_config(model_config)
        model.load_state_dict(state.get("state_dict", state), strict=False)
    else:
        config = PrismBreathConfig.from_variant(model_variant=variant, seq_len=seq_len, d_model=d_model)
        model = build_prism_breath_model_from_config(config.to_dict())
    return model


def _prism_inputs(batch_size: int, seq_len: int, device: torch.device):
    return {
        "depth": torch.randn(batch_size, seq_len, device=device),
        "flow": torch.randn(batch_size, seq_len, device=device),
        "audio": torch.randn(batch_size, seq_len, device=device),
        "age_months": torch.rand(batch_size, device=device) * 24.0,
        "modality_mask": torch.ones(batch_size, 3, device=device),
        "dataset_ids": torch.zeros(batch_size, dtype=torch.long, device=device),
    }


@torch.no_grad()
def _measure(model, inputs, device: torch.device, warmup: int, iters: int):
    is_cuda = device.type == "cuda"
    for _ in range(max(1, warmup)):
        model(**inputs)
    if is_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(max(1, iters)):
        start = time.perf_counter()
        model(**inputs)
        if is_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)  # ms
    samples.sort()
    p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
    mean_ms = statistics.fmean(samples)
    batch_size = inputs["depth"].shape[0]
    return {
        "latency_ms_mean": round(mean_ms, 4),
        "latency_ms_median": round(statistics.median(samples), 4),
        "latency_ms_p95": round(p95, 4),
        "per_sample_ms": round(mean_ms / batch_size, 4),
        "throughput_sps": round(batch_size / (mean_ms / 1000.0), 2),
    }


def _quantize_dynamic_cpu(model):
    if HAS_TORCHAO:
        import copy

        q = copy.deepcopy(model)
        quantize_(q, int8_dynamic_activation_int8_weight())
        return q
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


def benchmark_edge_latency(
    variants,
    seq_len=300,
    d_model=64,
    batch_sizes=(1,),
    devices=("cpu",),
    warmup=10,
    iters=50,
    threads=0,
    checkpoint="",
    quantize=False,
    report="",
):
    if threads and threads > 0:
        torch.set_num_threads(int(threads))

    env = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seq_len": seq_len,
        "d_model": d_model,
    }
    print(f"[env] {env['platform']} | torch {env['torch']} | threads={env['torch_threads']} | "
          f"cuda={'yes ('+env['cuda_device']+')' if env['cuda_available'] else 'no'}")

    rows = []
    for variant in variants:
        model = _build_prism_family_model(variant, seq_len, d_model, checkpoint=checkpoint).eval()
        params = _count_params(model)
        size_mb = _fp32_size_mb(model)
        print(f"\n=== {variant} :: {model.model_name()} | params={params/1e3:.1f}K | fp32={size_mb:.2f} MB ===")
        for device_str in devices:
            if device_str == "cuda" and not torch.cuda.is_available():
                print("  [skip] cuda requested but not available")
                continue
            device = torch.device(device_str)
            model.to(device)
            for bs in batch_sizes:
                stats = _measure(model, _prism_inputs(bs, seq_len, device), device, warmup, iters)
                row = {"variant": variant, "model_name": model.model_name(), "device": device_str,
                       "precision": "fp32", "batch_size": bs, "params": params,
                       "size_mb": round(size_mb, 3), **stats}
                rows.append(row)
                print(f"  [{device_str} fp32 bs={bs:>2}] "
                      f"{stats['latency_ms_mean']:.2f} ms/batch (p95 {stats['latency_ms_p95']:.2f}), "
                      f"{stats['per_sample_ms']:.2f} ms/sample, {stats['throughput_sps']:.1f} sps")
            model.to("cpu")

        if quantize:
            qmodel = _quantize_dynamic_cpu(model).eval()
            device = torch.device("cpu")
            for bs in batch_sizes:
                stats = _measure(qmodel, _prism_inputs(bs, seq_len, device), device, warmup, iters)
                q_size = _fp32_size_mb(qmodel)
                row = {"variant": variant, "model_name": model.model_name(), "device": "cpu",
                       "precision": "int8_dynamic", "batch_size": bs, "params": params,
                       "size_mb": round(q_size, 3), **stats}
                rows.append(row)
                print(f"  [cpu int8 bs={bs:>2}] "
                      f"{stats['latency_ms_mean']:.2f} ms/batch, {stats['per_sample_ms']:.2f} ms/sample, "
                      f"{stats['throughput_sps']:.1f} sps")

    payload = {"env": env, "results": rows}
    if report:
        os.makedirs(os.path.dirname(os.path.abspath(report)), exist_ok=True)
        with open(report, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\n[OK] Saved latency report to {report}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Export/benchmark respiratory models for edge devices")
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run the PRISM-family edge latency benchmark instead of the legacy transformer export.",
    )
    # Latency benchmark options
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["prism_breath", "mamba", "cliffphys", "airflow_compact"],
        help="PRISM-family variants to benchmark (accuracy-vs-latency comparison).",
    )
    parser.add_argument("--seq-len", type=int, default=300)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--devices", nargs="+", default=["cpu"], help="Subset of {cpu, cuda}")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = leave default; set 1-4 for an edge proxy)")
    parser.add_argument("--quantize", action="store_true", help="Also measure CPU dynamic INT8 latency")
    parser.add_argument("--report", type=str, default="", help="Optional JSON report path")
    # Legacy export options
    parser.add_argument("--checkpoint", type=str, default="data/models/finetuned_model.pth", help="Path to trained model")
    parser.add_argument("--out", type=str, default="data/models/edge", help="Output directory (legacy export)")
    args = parser.parse_args()

    if args.benchmark:
        benchmark_edge_latency(
            variants=args.variants,
            seq_len=args.seq_len,
            d_model=args.d_model,
            batch_sizes=tuple(args.batch_sizes),
            devices=tuple(args.devices),
            warmup=args.warmup,
            iters=args.iters,
            threads=args.threads,
            checkpoint=args.checkpoint if os.path.exists(args.checkpoint) else "",
            quantize=args.quantize,
            report=args.report,
        )
    else:
        export_edge_model(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
