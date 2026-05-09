#!/bin/bash
# Usage: ./pull_image.sh <filename>
# Example: ./pull_image.sh submission.tar.gz

FILE="${1:?Usage: ./pull_image.sh <filename>}"
VM="mks-admin@4.145.83.200"
REMOTE_DIR="/home/mks-admin/kddcup2026-data-agents-starter-kit"

scp "$VM:$REMOTE_DIR/$FILE" .

echo "Done: $FILE copied to $(pwd)/"
