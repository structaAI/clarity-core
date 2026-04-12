#!/bin/bash

# Define your absolute path (Bash format for Git Bash)
TARGET_DIR="/d/Structa/claritycore/ai-engine/data"

# Create directory if it doesn't exist
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "📥 Starting COCO 2017 Full Download..."

# 1. DOWNLOAD & UNZIP ANNOTATIONS
if [ ! -d "annotations" ]; then
    echo "📜 Processing Annotations..."
    until curl -C - -L --fail --retry 999 http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations.zip; do
        echo "⚠️ Annotations download interrupted. Retrying in 5 seconds..."
        sleep 5
    done
    unzip -o annotations.zip
    rm annotations.zip
else
    echo "✅ Annotations already exist. Skipping."
fi

# 2. DOWNLOAD & UNZIP TRAIN IMAGES (19GB)
echo "🖼️ Processing Train2017 (118k images)..."
URL="http://images.cocodataset.org/zips/train2017.zip"
FILE="train2017.zip"

# The Loop: Keep trying until curl returns exit code 0 (success)
until curl -C - -L --fail --retry 999 --retry-delay 5 -o "$FILE" "$URL"; do
    echo "⚠️ Connection reset on Train2017. Resuming in 5 seconds..."
    sleep 5
done

if [ -f "$FILE" ]; then
    echo "📦 Unzipping Train2017... This will take some time."
    unzip -q -o "$FILE"
    rm "$FILE"
fi

# 3. DOWNLOAD & UNZIP VAL IMAGES (1GB)
echo "🖼️ Processing Val2017..."
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
VAL_FILE="val2017.zip"

until curl -C - -L --fail --retry 999 -o "$VAL_FILE" "$VAL_URL"; do
    echo "⚠️ Connection reset on Val2017. Resuming in 5 seconds..."
    sleep 5
done

if [ -f "$VAL_FILE" ]; then
    unzip -q -o "$VAL_FILE"
    rm "$VAL_FILE"
fi

echo "🚀 DONE! Dataset ready in $TARGET_DIR"