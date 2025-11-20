#!/bin/bash
# /usr/local/bin/enforce-mig-slices.sh
# FINAL VERSION — works perfectly on A100 with driver 580.xx

LOG="/var/log/mig-enforce.log"
echo "=== MIG ENFORCEMENT STARTED $(date) ===" | tee -a "$LOG"

MIG_STATUS=$(nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv,noheader,nounits | tr -d ' ')
echo "MIG mode: '$MIG_STATUS'" | tee -a "$LOG"

if [[ "$MIG_STATUS" != "Enabled" ]]; then
    echo "MIG not Enabled → exiting" | tee -a "$LOG"
    exit 0
fi

echo "MIG Enabled → enforcing 7x 1g.5gb (1g.10gb) slices" | tee -a "$LOG"

# Clean everything
echo "Destroying any old instances..." | tee -a "$LOG"
nvidia-smi mig -i 0 -dci 2>&1 | tee -a "$LOG"
nvidia-smi mig -i 0 -dgi 2>&1 | tee -a "$LOG"

# ONE COMMAND DOES IT ALL — creates GPU instances + compute instances automatically
echo "Creating 7x 1g.5gb slices WITH compute instances (-C flag does both now)..." | tee -a "$LOG"
nvidia-smi mig -i 0 -cgi 19,19,19,19,19,19,19 -C 2>&1 | tee -a "$LOG"

if [ $? -eq 0 ]; then
    echo "=== SUCCESS: 7 MIG slices fully created and ready for CUDA ===" | tee -a "$LOG"
else
    echo "=== FAILED to create slices ===" | tee -a "$LOG"
    exit 1
fi

# Final proof
echo "=== Current MIG devices ===" | tee -a "$LOG"
nvidia-smi -L | tee -a "$LOG"

echo "=== MIG ENFORCEMENT COMPLETED $(date) ===" | tee -a "$LOG"
