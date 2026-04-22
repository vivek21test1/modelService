#!/bin/bash
PYTHON=/home/zeus/miniconda3/envs/cloudspace/bin/python
LOG=/teamspace/studios/this_studio/uvicorn.log
SERVICE=/teamspace/studios/this_studio/modelService

# Skip if already running
if pgrep -f "uvicorn app.main" > /dev/null 2>&1; then
    echo "[$(date)] uvicorn already running, skipping" >> "$LOG"
    exit 0
fi

cd "$SERVICE"
echo "[$(date)] Starting uvicorn..." >> "$LOG"
nohup "$PYTHON" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 >> "$LOG" 2>&1 &
echo "[$(date)] uvicorn PID: $!" >> "$LOG"
