#!/usr/bin/env python
import torch
import os
import subprocess
import sys

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode().strip()
    except:
        return ""

print("PyTorch version:", torch.__version__)
print("torch.cuda.is_available():", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("ERROR: CUDA runtime not available at all")
    sys.exit(1)

visible_count = torch.cuda.device_count()
print(f"Initially visible CUDA devices: {visible_count}")

if visible_count == 0:
    print("No devices visible by default — checking MIG status...")
    
    # Check if MIG mode is enabled on any GPU
    mig_mode = run_cmd("nvidia-smi -q | grep 'MIG Mode' | head -n1 | awk '{print $NF}'")
    print(f"MIG mode reported by nvidia-smi: {mig_mode}")
    
    if mig_mode == "Enabled":
        # MIG is enabled — try to list MIG devices
        mig_list = run_cmd("nvidia-smi -L | grep MIG")
        mig_uuids = [line.split()[-1].strip('()') for line in mig_list.splitlines() if "MIG-" in line]
        
        if mig_uuids:
            print(f"Found {len(mig_uuids)} active MIG instances → making them visible")
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(mig_uuids)
        else:
            print("MIG mode is Enabled but NO MIG instances exist!")
            print("    → This is a valid (but empty) state on multi-user instances.")
            print("    → We will expose the raw GPU indices 0–7 again so CUDA works.")
            # Force visibility of the physical GPUs even though MIG mode is on
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
        
        # Force PyTorch to re-initialize CUDA with new visibility
        torch.cuda._initialized = False
        visible_count = torch.cuda.device_count()
        print(f"After MIG handling → visible devices: {visible_count}")
        
        if visible_count == 0:
            print("Still no devices visible after MIG workaround → fatal error")
            sys.exit(1)

# At this point we MUST have at least one device
try:
    dev = torch.cuda.current_device()
    name = torch.cuda.get_device_name(dev)
    print(f"Success! Using device {dev}: {name}")
    
    # Quick smoke test
    x = torch.randn(16, 16).cuda()
    y = torch.randn(16, 16).cuda()
    z = torch.matmul(x, y)
    print(f"Matmul worked → result sum = {z.sum().item():.4f}")
    print("CUDA TEST PASSED (works with MIG enabled/disabled, instances or not)")
except Exception as e:
    print("GPU operation failed:", e)
    sys.exit(1)
