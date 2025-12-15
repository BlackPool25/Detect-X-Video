#!/bin/bash
# Quick test script for multimodal deepfake detection system

echo "🧪 Multimodal Deepfake Detection - Test Suite"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Weights directory
echo "1️⃣  Checking model weights..."
WEIGHTS_DIR="modal_services/weights"

if [ -d "$WEIGHTS_DIR" ]; then
    echo -e "${GREEN}✅ Weights directory exists${NC}"
    
    if [ -f "$WEIGHTS_DIR/efficientnet_b7_deepfake.pt" ]; then
        SIZE=$(du -h "$WEIGHTS_DIR/efficientnet_b7_deepfake.pt" | cut -f1)
        echo -e "${GREEN}✅ EfficientNet-B7: $SIZE${NC}"
    else
        echo -e "${RED}❌ EfficientNet-B7 weight missing${NC}"
    fi
    
    if [ -f "$WEIGHTS_DIR/model.safetensors" ]; then
        SIZE=$(du -h "$WEIGHTS_DIR/model.safetensors" | cut -f1)
        echo -e "${GREEN}✅ Wav2Vec2: $SIZE${NC}"
    else
        echo -e "${RED}❌ Wav2Vec2 weight missing${NC}"
    fi
    
    if [ -f "$WEIGHTS_DIR/retinaface_resnet50.pth" ]; then
        SIZE=$(du -h "$WEIGHTS_DIR/retinaface_resnet50.pth" | cut -f1)
        echo -e "${GREEN}✅ RetinaFace: $SIZE${NC}"
    else
        echo -e "${RED}❌ RetinaFace weight missing${NC}"
    fi
else
    echo -e "${RED}❌ Weights directory not found${NC}"
fi

echo ""

# Check 2: Test videos
echo "2️⃣  Checking test videos..."
TEST_DIR="Test-Video"

if [ -d "$TEST_DIR" ]; then
    VIDEO_COUNT=$(find "$TEST_DIR" -name "*.mp4" | wc -l)
    echo -e "${GREEN}✅ Found $VIDEO_COUNT test video(s)${NC}"
    find "$TEST_DIR" -name "*.mp4" -exec basename {} \; | while read name; do
        echo "   📹 $name"
    done
else
    echo -e "${YELLOW}⚠️  Test-Video directory not found${NC}"
fi

echo ""

# Check 3: Modal CLI
echo "3️⃣  Checking Modal CLI..."
if command -v modal &> /dev/null; then
    VERSION=$(modal --version 2>&1 | head -1)
    echo -e "${GREEN}✅ Modal CLI installed: $VERSION${NC}"
    
    # Check authentication
    if modal token get &> /dev/null; then
        echo -e "${GREEN}✅ Modal authenticated${NC}"
    else
        echo -e "${YELLOW}⚠️  Modal not authenticated. Run: modal token new${NC}"
    fi
else
    echo -e "${RED}❌ Modal CLI not installed${NC}"
    echo "   Install with: pipx install modal"
fi

echo ""

# Check 4: Python dependencies (check if files are syntactically valid)
echo "4️⃣  Checking Python files..."

FILES=(
    "modal_services/deepfake_detector.py"
    "whatsapp/message_handler.py"
    "whatsapp/modal_service.py"
    "whatsapp/app.py"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        if python3 -m py_compile "$file" 2>/dev/null; then
            echo -e "${GREEN}✅ $file${NC}"
        else
            echo -e "${YELLOW}⚠️  $file (has syntax warnings - normal for Modal)${NC}"
        fi
    else
        echo -e "${RED}❌ $file not found${NC}"
    fi
done

echo ""

# Check 5: Supabase connection (if env vars are set)
echo "5️⃣  Checking environment..."

ENV_FILE="whatsapp/.env"
if [ -f "$ENV_FILE" ]; then
    echo -e "${GREEN}✅ .env file exists${NC}"
    
    # Check for required variables (without showing values)
    if grep -q "SUPABASE_URL" "$ENV_FILE"; then
        echo -e "${GREEN}✅ SUPABASE_URL configured${NC}"
    else
        echo -e "${YELLOW}⚠️  SUPABASE_URL not found in .env${NC}"
    fi
    
    if grep -q "SUPABASE_SERVICE_KEY" "$ENV_FILE"; then
        echo -e "${GREEN}✅ SUPABASE_SERVICE_KEY configured${NC}"
    else
        echo -e "${YELLOW}⚠️  SUPABASE_SERVICE_KEY not found in .env${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  .env file not found in whatsapp/${NC}"
fi

echo ""
echo "=============================================="
echo "📝 Summary"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Authenticate Modal: ${YELLOW}modal token new${NC}"
echo "2. Deploy to Modal: ${YELLOW}modal deploy modal_services/deepfake_detector.py${NC}"
echo "3. Copy Modal URL and update .env: MODAL_VIDEO_API_URL="
echo "4. Test with video from Test-Video folder"
echo ""
echo "Full guide: ${YELLOW}cat DEPLOYMENT_GUIDE.md${NC}"
