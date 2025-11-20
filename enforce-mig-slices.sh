#!/bin/bash
# /usr/local/bin/enforce-mig-slices.sh
# Idempotently enforce exactly 7 x 1g.5gb MIG instances on GPU 0
# Only runs if MIG mode is currently Enabled

LOG="/var/log/mig-enforce.log"
echo "$(date): enforce-mig-slices.sh started" >> "$LOG"

if ! nvidia-smi -i 0 -q | grep -q "MIG Mode.*: Enabled"; then
  echo "$(date): MIG mode is Disabled or Pending → nothing to do" >> "$LOG"
  exit 0
fi

echo "$(date): MIG mode is Enabled → enforcing 7x 1g.5gb slices on GPU 0" >> "$LOG"

# Clean any existing state (safe if nothing exists)
nvidia-smi mig -i 0 -dci >/dev/null 2>&1 || true
nvidia-smi mig -i 0 -dgi >/dev/null 2>&1 || true

# Create the seven 1g.5gb GPU instances (profile 19)
nvidia-smi mig -i 0 -cgi 19,19,19,19,19,19,19 -C >> "$LOG" 2>&1

# Create one compute instance on each GPU instance (required for CUDA visibility)
for gi in $(nvidia-smi mig -i 0 -L | grep "GPU instance" | awk '{print $3}'); do
  nvidia-smi mig -i 0 -gi "$gi" -cgi 0 -C >> "$LOG" 2>&1
done

echo "$(date): MIG slice enforcement completed successfully" >> "$LOG"
