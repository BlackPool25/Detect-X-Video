#!/bin/bash
# Deploy multimodal deepfake detection to Modal

echo "🚀 Deploying Multimodal Deepfake Detection System to Modal"
echo "=========================================================="

# Check if modal CLI is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
fi

# Check Modal authentication
echo "🔑 Checking Modal authentication..."
if ! modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET" 2>/dev/null; then
    echo "⚠️  Please authenticate with Modal first:"
    echo "   modal token new"
    exit 1
fi

echo "✅ Modal authenticated"

# Deploy the main detection app
echo ""
echo "📦 Deploying detection app..."
cd "$(dirname "$0")"

modal deploy modal_services/deepfake_detector.py

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📝 Next steps:"
echo "1. Get your Modal app URL from the deployment output"
echo "2. Update MODAL_VIDEO_API_URL in your .env file"
echo "3. Test the API with: curl <MODAL_URL>/health"
echo ""
echo "🔗 Modal Dashboard: https://modal.com/apps"
