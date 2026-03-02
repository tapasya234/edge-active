import torch
import torch.onnx
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import json
import torchvision
import torch.nn as nn


class ModelExporter:
    """Export PyTorch model to ONNX for AI Hub"""

    def __init__(self, checkpoint_path, output_dir="./exports"):
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LPCVC input specifications
        self.batch_size = 1
        self.num_channels = 3
        self.num_frames = 8
        self.height = 112
        self.width = 112

    def load_model(self, num_classes):
        """Load trained PyTorch model"""

        # Create model architecture (same as training)
        model = torchvision.models.get_model("r2plus1d_18", weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=False
        )

        print(f"Checkpoint keys: {checkpoint.keys()}")

        # Handle both checkpoint formats
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
            model.load_state_dict(
                state_dict, strict=True
            )  # Add strict=True to catch errors
            print(f"Loaded {len(state_dict)} layers from checkpoint")
        else:
            model.load_state_dict(checkpoint, strict=True)

        model.eval()
        model = model.to(torch.float32)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")

        return model

    def export_to_onnx(self, model, onnx_path):
        """Export model to ONNX format"""
        print(f"\nExporting to ONNX: {onnx_path}")

        # Create dummy input matching LPCVC format: (B, C, T, H, W)
        dummy_input = torch.randn(
            self.batch_size,
            self.num_channels,
            self.num_frames,
            self.height,
            self.width,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        # Sanity check
        with torch.no_grad():
            output = model(dummy_input)
            print(f"  Sanity check output shape: {tuple(output.shape)}")

        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["video"],
            output_names=["logits"],
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=False,
        )
        print("ONNX export complete (dynamo=False)")

    def verify_onnx(self, onnx_path):
        """Verify ONNX model"""
        print("\nVerifying ONNX model...")

        # Load and check model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")

        # Print model info
        print(f"\nModel Information:")
        print(f"  Opset: {onnx_model.opset_import[0].version}")

        # Print inputs
        print("\n  Inputs:")
        for input_tensor in onnx_model.graph.input:
            dims = [
                d.dim_value if d.dim_value > 0 else d.dim_param
                for d in input_tensor.type.tensor_type.shape.dim
            ]
            print(f"    - {input_tensor.name}: {dims}")

        # Print outputs
        print("\n  Outputs:")
        for output_tensor in onnx_model.graph.output:
            dims = [
                d.dim_value if d.dim_value > 0 else d.dim_param
                for d in output_tensor.type.tensor_type.shape.dim
            ]
            print(f"    - {output_tensor.name}: {dims}")

    def test_inference(self, onnx_path):
        """Test ONNX inference"""
        print("\nTesting ONNX inference...")

        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)

        # Create test input
        test_input = np.random.randn(
            self.batch_size, self.num_channels, self.num_frames, self.height, self.width
        ).astype(np.float32)

        # Run inference
        outputs = session.run(None, {"video": test_input})

        print(f"  Inference successful")
        print(f"  Output shape: {outputs[0].shape}")
        print(f"  Predicted class: {np.argmax(outputs[0])}")

        return outputs

    def get_model_size(self, onnx_path):
        """Get model file size"""
        size_mb = Path(onnx_path).stat().st_size / (1024 * 1024)
        print(f"\nModel size: {size_mb:.2f} MB")
        return size_mb

    def export(self, num_classes, model_name="model"):
        """Complete export pipeline"""
        print("=" * 60)
        print("LPCVC Track 2 - Model Export to ONNX")
        print("=" * 60)

        # Load model
        model = self.load_model(num_classes)

        # Export to ONNX
        onnx_path = self.output_dir / f"{model_name}.onnx"
        self.export_to_onnx(model, onnx_path)

        # Verify
        self.verify_onnx(onnx_path)

        # Test inference
        self.test_inference(onnx_path)

        # Get size
        size_mb = self.get_model_size(onnx_path)

        # Save export metadata
        metadata = {
            "checkpoint_path": str(self.checkpoint_path),
            "onnx_path": str(onnx_path),
            "model_size_mb": size_mb,
            "input_shape": [
                self.batch_size,
                self.num_channels,
                self.num_frames,
                self.height,
                self.width,
            ],
            "num_classes": num_classes,
            "target_device": "Qualcomm Dragonwing IQ-9075 EVK",
        }

        metadata_path = self.output_dir / f"{model_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print("\n" + "=" * 60)
        print("Export Summary")
        print("=" * 60)
        print(f"✓ ONNX model: {onnx_path}")
        print(f"✓ Metadata: {metadata_path}")
        print(f"✓ Model size: {size_mb:.2f} MB")
        print("\nNext Steps:")
        print("1. Upload ONNX model to Qualcomm AI Hub")
        print("2. Compile for Dragonwing IQ-9075 platform")
        print("3. Profile inference time (<100ms required)")
        print("4. Submit via LPCVC submission form")
        print("=" * 60)

        return onnx_path


def main():
    """Main export script"""
    import argparse

    parser = argparse.ArgumentParser(description="Export LPCVC Track 2 model to ONNX")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to PyTorch checkpoint (.pth)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=92,
        help="Number of classes (default: 92 for QEVD)",
    )
    parser.add_argument(
        "--name", type=str, default="r2plus1d_qevd", help="Output model name"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./exports", help="Output directory"
    )

    args = parser.parse_args()

    # Export model
    exporter = ModelExporter(args.checkpoint, args.output_dir)
    exporter.export(args.num_classes, args.name)

    print("\n✓ Ready for AI Hub submission!")


if __name__ == "__main__":
    main()
