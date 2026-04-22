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

# Fully detach from the shell session so studio.run() can return cleanly.
# stdin redirected from /dev/null so the process has no terminal attachment.
nohup "$PYTHON" -m uvicorn app.main:app \
    --host 0.0.0.0 --port 8000 \
    < /dev/null >> "$LOG" 2>&1 &

disown $!
echo "[$(date)] uvicorn PID: $!" >> "$LOG"
