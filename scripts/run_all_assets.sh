#!/bin/bash
# Script to run panosam on all images in the assets folder
# Usage: ./scripts/run_all_assets.sh [prompt] [preset]
# Example: ./scripts/run_all_assets.sh "window" "wideangle"

set -e

# Default parameters (matching the example command)
PROMPT="${1:-window}"
PRESET="${2:-wideangle}"

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ASSETS_DIR="$PROJECT_DIR/assets"

echo "==================================="
echo "PanoSAM Batch Processing"
echo "==================================="
echo "Prompt: $PROMPT"
echo "Preset: $PRESET"
echo "Assets directory: $ASSETS_DIR"
echo ""

# Count total images
TOTAL=$(find "$ASSETS_DIR" -maxdepth 1 -name "*.jpg" | wc -l | tr -d ' ')
echo "Found $TOTAL images to process"
echo ""

# Counter for progress
COUNT=0

# Process each image
for IMAGE in "$ASSETS_DIR"/*.jpg; do
    if [[ -f "$IMAGE" ]]; then
        COUNT=$((COUNT + 1))
        BASENAME=$(basename "$IMAGE")
        
        echo "[$COUNT/$TOTAL] Processing: $BASENAME"
        echo "-----------------------------------"
        
        uv run panosam --image "$IMAGE" --prompt "$PROMPT" --preset "$PRESET"
        
        echo ""
    fi
done

echo "==================================="
echo "Batch processing complete!"
echo "Processed $COUNT images"
echo "==================================="
