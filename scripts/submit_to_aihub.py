"""
Qualcomm AI Hub Submission Script for LPCVC Track 2

This script handles:
1. Model upload to AI Hub
2. Compilation for Dragonwing IQ-9075
3. Performance profiling
4. Sharing with LPCVC organizers
"""

import qai_hub
from pathlib import Path


class LPCVCSubmitter:
    """Handle AI Hub submission for LPCVC"""

    def __init__(self, onnx_path):
        self.onnx_path = Path(onnx_path)
        self.device_name = "Dragonwing IQ-9075 EVK"
        self.organizer_email = "lowpowervision@gmail.com"

        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    def upload_model(self):
        """
        Upload ONNX model to AI Hub
        """
        print(f"\nUploading model: {self.onnx_path}")

        model = qai_hub.upload_model(str(self.onnx_path))

        print(f" Model uploaded")
        print(f"  Model ID: {model.model_id}")

        return model

    def compile_model(self, model, quantize=False):
        """
        Compile model for target device
        """
        print(f"\nCompiling for {self.device_name}...")

        if quantize:
            options = "--target_runtime precompiled_qnn_onnx --quantize_full_type int8"
            print("  Using INT8 quantization for speed")
        else:
            options = "--target_runtime precompiled_qnn_onnx"
            print("  Using FP32 (no quantization)")

        compile_job = qai_hub.submit_compile_job(
            model=model, device=qai_hub.Device(self.device_name), options=options
        )

        print(f"  Compile Job ID: {compile_job.job_id}")
        print("  Waiting for compilation...")
        compile_job.wait()

        status = compile_job.get_status()
        if status.code == qai_hub.JobStatus.State.SUCCESS.name:
            print(" Compilation successful")
        else:
            print(f" !!!! Compilation failed: {status.code} - {status.message}")
            return None

        return compile_job

    def profile_model(self, compile_job):
        """
        Profile model performance on device
        """
        print("\nProfiling model performance...")

        profile_job = qai_hub.submit_profile_job(
            model=compile_job.get_target_model(),
            device=qai_hub.Device(self.device_name),
        )

        print(f"  Profile Job ID: {profile_job.job_id}")
        print("  Waiting for profiling...")
        profile_job.wait()

        # Get performance metrics
        profile_data = profile_job.download_profile()

        inference_time = (
            profile_data["execution_summary"]["estimated_inference_time"] / 1000
        )  # Convert µs to ms

        print(f" Profiling complete")
        print(f"  Inference Time: {inference_time:.2f} ms")

        if inference_time < 100:
            print(f"  PASSES LPCVC requirement (<100ms)")
        else:
            print(f" !!!! FAILS LPCVC requirement (>100ms)")
            print(f"  Consider:")
            print(f"    - Enabling INT8 quantization")
            print(f"    - Using a smaller model")
            print(f"    - Model pruning")

        return inference_time

    def share_with_organizers(self, compile_job):
        """
        Share compile job with LPCVC organizers
        """
        print(f"\nSharing with LPCVC organizers ({self.organizer_email})...")

        compile_job.modify_sharing(add_emails=[self.organizer_email])

        print(f" Compile job shared")
        print(f"  Job ID: {compile_job.job_id}")

    def submit(self, quantize=False):
        """
        Complete submission pipeline

        Args:
            quantize: Whether to use INT8 quantization

        Returns:
            compile_job_id: ID to use in LPCVC submission form
        """
        print("=" * 60)
        print("LPCVC Track 2 - AI Hub Submission")
        print("=" * 60)

        try:
            # Upload
            model = self.upload_model()

            # Compile
            compile_job = self.compile_model(model, quantize=quantize)
            if compile_job is None:
                print("\n Submission failed at compilation stage")
                return None

            # Profile
            inference_time = self.profile_model(compile_job)

            # Share with organizers
            self.share_with_organizers(compile_job)

            # Summary
            print("\n" + "=" * 60)
            print("Submission Summary")
            print("=" * 60)
            print(f" Model: {self.onnx_path}")
            print(f" Compile Job ID: {compile_job.job_id}")
            print(f" Inference Time: {inference_time:.2f} ms")
            print(f" Shared with: {self.organizer_email}")
            print("\n" + "=" * 60)
            print("Next Steps:")
            print("=" * 60)
            print("1. Go to LPCVC submission form")
            print("2. Enter the following Compile Job ID:")
            print(f"\n   {compile_job.job_id}\n")
            print("3. Fill in team information")
            print("4. Submit!")
            print("\nKeep this Compile Job ID for your records.")
            print("=" * 60)

            return compile_job.job_id

        except Exception as e:
            print(f"\n !!!!!! Submission failed: {e}")
            print("\nTroubleshooting:")
            print("1. Check your AI Hub credentials")
            print("2. Verify ONNX model is valid")
            print("3. Check AI Hub status page")
            return None

    def resume_job(self, job_id: str):
        """
        If the script fails for a reason but the model was successfully uploaded,
        resume job can be updated to process the remaining step.

        :param job_id: Description
        :type job_id: str
        """
        print("\nResuming job - ", job_id)
        compile_job = qai_hub.get_job(job_id)
        status = compile_job.get_status()

        if status.code == qai_hub.JobStatus.State.SUCCESS.name:
            print(" Compilation successful")
        else:
            print(f" !!!! Compilation failed: {status.message}")
            return None

        inference_time = self.profile_model(compile_job)
        self.share_with_organizers(compile_job)

        print("\n" + "=" * 60)
        print("Submission Summary")
        print("=" * 60)
        print(f" Model: {self.onnx_path}")
        print(f" Compile Job ID: {compile_job.job_id}")
        print(f" Inference Time: {inference_time:.2f} ms")
        print(f" Shared with: {self.organizer_email}")
        print("\n" + "=" * 60)


def main():
    """
    Main submission script
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Submit model to Qualcomm AI Hub for LPCVC"
    )
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX model")
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Use INT8 quantization (recommended for speed)",
    )

    parser.add_argument(
        "--resume-job",
        type=str,
        help="Resumes an existing job if it was stopped earlier than expected.",
    )

    args = parser.parse_args()

    print(args.resume_job)
    if args.resume_job:
        submitter = LPCVCSubmitter(args.model)
        submitter.resume_job(args.resume_job)
        print("\n Submission successful!")

        return

    # Submit model
    submitter = LPCVCSubmitter(args.model)
    compile_job_id = submitter.submit(quantize=args.quantize)

    if compile_job_id:
        print("\n Submission successful!")
    else:
        print("\n Submission failed - see errors above")


if __name__ == "__main__":
    main()
