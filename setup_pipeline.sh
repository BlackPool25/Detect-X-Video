#!/bin/bash
# Quick Setup Script for 4-Layer Deepfake Detection Pipeline

set -e  # Exit on error

echo "========================================="
echo "4-Layer Deepfake Pipeline Setup"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing dependencies..."
pip install -r requirements_pipeline.txt

echo ""
echo "========================================="
echo "Checking Model Weights"
echo "========================================="
echo ""

# Check SBI weights
if [ -f "weights/SBI/FFc23.tar" ]; then
    echo "✓ SBI weights found: weights/SBI/FFc23.tar"
else
    echo "✗ SBI weights missing: weights/SBI/FFc23.tar"
    echo "  Download from: https://drive.google.com/file/d/1X0-NYT8KPursLZZdxduRQju6E52hauV0"
fi

# Check SyncNet weights
if [ -f "syncnet_python/data/syncnet_v2.model" ]; then
    echo "✓ SyncNet weights found: syncnet_python/data/syncnet_v2.model"
else
    echo "⚠ SyncNet weights missing. Downloading..."
    cd syncnet_python
    bash download_model.sh
    cd ..
    echo "✓ SyncNet weights downloaded"
fi

echo ""
echo "========================================="
echo "Checking Test Videos"
echo "========================================="
echo ""

# Count test videos
REAL_COUNT=$(find Test-Video/Real Test-Video/Celeb-real Test-Video/YouTube-real -name "*.mp4" -o -name "*.avi" 2>/dev/null | wc -l)
FAKE_COUNT=$(find Test-Video/Fake Test-Video/Celeb-synthesis -name "*.mp4" -o -name "*.avi" 2>/dev/null | wc -l)

echo "Real videos found: $REAL_COUNT"
echo "Fake videos found: $FAKE_COUNT"

if [ $REAL_COUNT -eq 0 ] && [ $FAKE_COUNT -eq 0 ]; then
    echo "⚠ No test videos found in Test-Video/"
    echo "  Please add videos to Test-Video/Real/ and Test-Video/Fake/"
else
    echo "✓ Test videos available"
fi

echo ""
echo "========================================="
echo "Environment Variables"
echo "========================================="
echo ""

# Check for .env file
if [ -f ".env" ]; then
    echo "✓ .env file exists"
else
    echo "⚠ Creating .env file with default Supabase credentials..."
    cat > .env << 'EOF'
# Supabase Configuration
SUPABASE_URL=https://cjkcwycnetdhumtqthuk.supabase.co
SUPABASE_KEY=sb_publishable_kYQsl9DIOWNzkcZNUojI1w_yIyL70XH
EOF
    echo "✓ .env file created"
fi

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Quick Start Commands:"
echo ""
echo "1. Test single video:"
echo "   python deepfake_pipeline.py --video Test-Video/Real/example.mp4"
echo ""
echo "2. Run test suite:"
echo "   python test_pipeline.py --test-dir Test-Video --max-videos 5"
echo ""
echo "3. Check Supabase integration:"
echo "   python supabase_logger.py"
echo ""
echo "See README_PIPELINE.md for detailed documentation."
echo ""
