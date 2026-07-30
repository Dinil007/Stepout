"""Benchmark trained classifier performance.

Measures:
- Average inference time
- FPS
- GPU Memory (if available)
- CPU Memory
- Model loading time
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classification.config import CLASS_NAMES, DatasetConfig
from app.classification.data import prepare_classification_datasets
from app.classification.models import create_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def benchmark(
    model_path: str = "models/classifier/best.pth",
    data_dir: str = "datasets/person_classifier/prepared",
    model_name: str = "efficientnet_b0",
    batch_size: int = 32,
    image_size: int = 224,
    num_warmup: int = 10,
    num_benchmark: int = 100,
    output_dir: str = "models/classifier",
) -> Dict:
    """Benchmark model performance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load model
    model = create_model(model_name, num_classes=4, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Measure model loading time
    load_start = time.time()
    # (Model already loaded above)
    load_time = time.time() - load_start

    # Load dataset
    config = DatasetConfig(
        prepared_dir=Path(data_dir),
        image_size=(image_size, image_size),
        batch_size=batch_size,
        augment=False,
    )

    _, _, test_dataset = prepare_classification_datasets(config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing overhead for benchmarking
    )

    # Get sample batch
    sample_batch = next(iter(test_loader))[0].to(device)

    # Warmup
    logger.info(f"Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(sample_batch)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark inference time
    logger.info(f"Benchmarking ({num_benchmark} iterations)...")
    inference_times = []

    with torch.no_grad():
        for i in range(num_benchmark):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(sample_batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.time()
            inference_times.append(end - start)

    # Calculate metrics
    inference_times = np.array(inference_times)
    avg_inference_time = float(np.mean(inference_times))
    std_inference_time = float(np.std(inference_times))
    fps = batch_size / avg_inference_time

    # Memory usage
    if device.type == "cuda":
        gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
        gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**2  # MB
    else:
        gpu_memory_allocated = 0.0
        gpu_memory_reserved = 0.0

    # CPU memory (approximate)
    import psutil
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024**2  # MB

    # Model size
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    num_params = sum(p.numel() for p in model.parameters())

    # Results
    results = {
        "model_name": model_name,
        "device": str(device),
        "batch_size": batch_size,
        "image_size": f"{image_size}x{image_size}",
        "model_loading_time_s": round(load_time, 4),
        "avg_inference_time_ms": round(avg_inference_time * 1000, 2),
        "std_inference_time_ms": round(std_inference_time * 1000, 2),
        "fps": round(fps, 2),
        "gpu_memory_allocated_mb": round(gpu_memory_allocated, 2),
        "gpu_memory_reserved_mb": round(gpu_memory_reserved, 2),
        "cpu_memory_mb": round(cpu_memory, 2),
        "model_size_mb": round(model_size_mb, 2),
        "num_parameters": num_params,
    }

    # Save results
    results_path = output_path / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"\nModel Loading Time: {load_time:.4f}s")
    print(f"Average Inference Time: {avg_inference_time*1000:.2f}ms (±{std_inference_time*1000:.2f}ms)")
    print(f"FPS: {fps:.2f}")
    print(f"\nGPU Memory Allocated: {gpu_memory_allocated:.2f} MB")
    print(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} MB")
    print(f"CPU Memory: {cpu_memory:.2f} MB")
    print(f"Model Size: {model_size_mb:.2f} MB")
    print(f"Number of Parameters: {num_params:,}")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/classifier/best.pth")
    parser.add_argument("--data-dir", default="datasets/person_classifier/prepared")
    parser.add_argument("--model-name", default="efficientnet_b0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="models/classifier")
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-benchmark", type=int, default=100)
    args = parser.parse_args()

    benchmark(
        args.model,
        args.data_dir,
        args.model_name,
        args.batch_size,
        args.image_size,
        args.num_warmup,
        args.num_benchmark,
        args.output_dir,
    )