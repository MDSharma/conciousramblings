import os
import torch
import subprocess
import re

def detect_mig_slices():
    """Return a list of MIG instance UUIDs using `nvidia-smi -L`."""
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        mig_uuids = re.findall(r"(MIG-GPU-[\w-]+)", output)
        return mig_uuids
    except Exception as e:
        print(f"[MIG] Failed to detect MIG devices: {e}")
        return []

def assign_mig_to_user(user_index, mig_slices):
    """Assign a MIG slice based on the user index (round-robin)."""
    if not mig_slices:
        print("[MIG] No MIG slices detected, using full GPU")
        return None
    slice_to_use = mig_slices[user_index % len(mig_slices)]
    os.environ["CUDA_VISIBLE_DEVICES"] = slice_to_use
    print(f"[MIG] Assigned {slice_to_use} to user index {user_index}")
    return slice_to_use

# ----------------------------
# Example usage (for JupyterHub pre-spawn hook)
# ----------------------------

# Detect all MIG slices
mig_slices = detect_mig_slices()
print(f"[MIG] Available slices: {mig_slices}")

# Assign slice based on user UID (or any unique user index)
# For cloud-init testing, we can just use UID 0
user_index = os.getuid()
assigned_slice = assign_mig_to_user(user_index, mig_slices)

# PyTorch check
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available() and torch.cuda.device_count() > 0:
    for i in range(torch.cuda.device_count()):
        print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")

print("Done")
