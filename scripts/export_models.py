"""Export trained models to multiple formats.

Exports:
- PyTorch (.pt) - default
- ONNX (.onnx) - for deployment
- TorchScript (.torchscript) - for mobile/edge deployment
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classification.config import DatasetConfig
from app.classification.models import create_model


def export_model(
    checkpoint_path: str = "models/classifier/best.pth",
    output_dir: str = "models/classifier",
    model_name: str = "efficientnet_b0",
    num_classes: int = 4,
    image_size: int = 224,
) -> None:
    """Export model to multiple formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model
    config = DatasetConfig(model_name=model_name)
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model = model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Model: {model_name}, Classes: {num_classes}")

    # Export PyTorch (already saved during training)
    pt_path = output_path / "best_person_classifier.pt"
    torch.save(checkpoint, pt_path)
    print(f"Saved PyTorch model: {pt_path}")

    # Export ONNX
    onnx_path = output_path / "best_person_classifier.onnx"
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Saved ONNX model: {onnx_path}")

    # Export TorchScript
    scripted_path = output_path / "best_person_classifier.torchscript"
    scripted = torch.jit.script(model)
    scripted.save(scripted_path)
    print(f"Saved TorchScript model: {scripted_path}")

    print("\nExport complete!")
    print(f"All models saved to: {output_path}")


if __name__ == "__main__":
    export_model()