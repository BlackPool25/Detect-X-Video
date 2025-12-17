#!/bin/bash

# Modal Deployment Script for Optimized Deepfake Detection
# This script automates the deployment process

set -e  # Exit on error

echo "========================================"
echo "Modal Deployment - Deepfake Detection"
echo "========================================"
echo ""

# Check if modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
fi

# Check if modal token exists
if ! modal token show &> /dev/null; then
    echo "❌ Modal token not found. Please authenticate:"
    modal token new
fi

echo "✓ Modal CLI ready"
echo ""

# Check if weights exist
if [ ! -f "weights/SBI/FFc23.tar" ]; then
    echo "❌ Error: SBI weights not found at weights/SBI/FFc23.tar"
    echo "Please ensure the weights are downloaded."
    exit 1
fi

echo "✓ SBI weights found"
echo ""

# Deploy to Modal
echo "🚀 Deploying to Modal..."
echo ""

modal deploy modal_app_optimized.py

echo ""
echo "========================================"
echo "✓ Deployment Complete!"
echo "========================================"
echo ""
echo "Next Steps:"
echo ""
echo "1. Copy the deployment URLs from above"
echo ""
echo "2. Update your environment variables:"
echo "   Website: AI-Website/.env.local"
echo "   MODAL_VIDEO_API_URL=https://your-workspace--deepfake-detector-optimized-detect-video.modal.run"
echo ""
echo "   WhatsApp: Set environment variable or update whatsapp/modal_service.py"
echo "   export MODAL_VIDEO_API_URL=https://your-workspace--deepfake-detector-optimized-detect-video.modal.run"
echo ""
echo "3. Test the health endpoint:"
echo "   curl https://your-workspace--deepfake-detector-optimized-health.modal.run"
echo ""
echo "4. Test video detection:"
echo "   See MODAL_DEPLOYMENT_GUIDE.md for detailed testing instructions"
echo ""
echo "📊 View your deployment:"
echo "   https://modal.com/apps"
echo ""
