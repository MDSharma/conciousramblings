import os
import torch
import subprocess
import re

def detect_mig_slices():
    """
    Detect MIG slices on the system.
    Returns a list of MIG identifiers (UUIDs if available, else index strings).
    """
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        mig_slices = []

        # Attempt UUID-style detection first (modern drivers)
        mig_slices = re.findall(r"(MIG-GPU-[\w-]+)", output)
        if mig_slices:
            print(f"[MIG] Detected UUID-based slices: {mig_slices}")
            return mig_slices

        # Fallback: short-format detection ("GPU 0: NVIDIA ... MIG 1g.5gb")
        mig_lines = [line for line in output.splitlines() if "MIG" in line]
        if mig_lines:
            mig_slices = [str(idx) for idx, _ in enumerate(mig_lines)]
            print(f"[MIG] Detected index-based slices: {mig_slices}")
            return mig_slices

        # No MIGs detected
        print("[MIG] No MIG slices found, using full GPU")
        return []

    except Exception as e:
        print(f"[MIG] Failed to detect MIG devices: {e}")
        return []

def assign_mig_to_user(user_index, mig_slices):
    """
    Assign a MIG slice to a user (round-robin).
    Sets CUDA_VISIBLE_DEVICE
