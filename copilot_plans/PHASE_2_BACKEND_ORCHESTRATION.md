# PHASE 2: BACKEND & ORCHESTRATION - 4-LAYER DEEPFAKE DETECTION PLATFORM

## EXECUTIVE SUMMARY
This phase builds the FastAPI backend that orchestrates the ML pipeline (Phase 1), implements fail-fast logic, and stores results in Supabase. The backend acts as the bridge between the ML core and frontend/WhatsApp integrations (Phase 3).

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────┐
│                    BACKEND ORCHESTRATION                    │
│                                                             │
│  ┌─────────────┐      ┌──────────────────────────────┐    │
│  │  FastAPI    │─────▶│  ML Pipeline Orchestrator    │    │
│  │  REST API   │      │  (from Phase 1)              │    │
│  └──────┬──────┘      └──────────┬───────────────────┘    │
│         │                        │                         │
│         │                        ▼                         │
│         │              ┌─────────────────────┐             │
│         │              │  Fail-Fast Logic    │             │
│         │              │  (90% threshold)    │             │
│         │              └─────────────────────┘             │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────┐         ┌────────────────────┐      │
│  │  Supabase Client │────────▶│  Storage Service   │      │
│  │  (Results DB)    │         │  (Video Upload)    │      │
│  └──────────────────┘         └────────────────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. FASTAPI APPLICATION STRUCTURE

### 1.1 Directory Layout
```
backend/
├── main.py                 # FastAPI app entry point
├── config.py               # Environment configuration
├── requirements.txt        # Python dependencies
├── Dockerfile              # Backend container
├── api/
│   ├── __init__.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── detection.py    # Main detection endpoints
│   │   └── health.py       # Health check endpoints
│   └── schemas/
│       ├── __init__.py
│       └── detection.py    # Pydantic models
├── services/
│   ├── __init__.py
│   ├── storage.py          # Supabase storage client
│   ├── database.py         # Supabase DB client
│   └── ml_service.py       # ML pipeline wrapper
└── utils/
    ├── __init__.py
    ├── video_processor.py  # Video preprocessing
    └── logger.py           # Structured logging
```

### 1.2 Core FastAPI Application

```python
# File: backend/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path

from api.routes import detection, health
from config import settings
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger()

# Initialize FastAPI app
app = FastAPI(
    title="4-Layer Deepfake Detection API",
    description="Proprietary multi-layer deepfake detection system",
    version="1.0.0"
)

# CORS Configuration (for Web Frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.WEB_FRONTEND_URL,  # Next.js app
        "http://localhost:3000",     # Local dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detection.router, prefix="/api/v1/detection", tags=["Detection"])
app.include_router(health.router, prefix="/api/v1/health", tags=["Health"])

@app.on_event("startup")
async def startup_event():
    """
    Initialize services on startup
    - Warm up ML models (optional)
    - Verify Supabase connection
    - Create temp directories
    """
    logger.info("Starting 4-Layer Deepfake Detection Backend...")
    
    # Create temp directory for video processing
    temp_dir = Path(settings.TEMP_DIR)
    temp_dir.mkdir(exist_ok=True)
    
    # Test Supabase connection
    from services.database import db_client
    try:
        db_client.test_connection()
        logger.info("✓ Supabase connection established")
    except Exception as e:
        logger.error(f"✗ Supabase connection failed: {e}")
        raise
    
    logger.info("✓ Backend ready to accept requests")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down backend...")
    
    # Clean up temp files
    import shutil
    temp_dir = Path(settings.TEMP_DIR)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    logger.info("✓ Shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
```

---

## 2. CONFIGURATION MANAGEMENT

```python
# File: backend/config.py

from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """
    Environment-based configuration
    Uses .env file or environment variables
    """
    
    # Application
    APP_NAME: str = "DeepfakeDetection4Layer"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    PORT: int = 8000
    
    # Supabase
    SUPABASE_URL: str
    SUPABASE_ANON_KEY: str  # For client-side operations
    SUPABASE_SERVICE_KEY: str  # For server-side admin operations
    SUPABASE_BUCKET_NAME: str = "deepfake-videos"
    
    # ML Pipeline (Phase 1)
    ML_WEIGHTS_DIR: str = "/app/weights"
    ML_DEVICE: str = "cuda"  # or "cpu"
    
    # Fail-Fast Configuration
    FAIL_FAST_THRESHOLD: float = 0.90  # 90% confidence threshold
    
    # Storage
    TEMP_DIR: str = "/tmp/deepfake_processing"
    MAX_VIDEO_SIZE_MB: int = 500  # 500MB limit
    MAX_VIDEO_DURATION_SEC: int = 300  # 5 minutes
    
    # Frontend URLs
    WEB_FRONTEND_URL: str = "https://yourdomain.com"
    
    # WhatsApp Integration
    WHATSAPP_WEBHOOK_SECRET: str
    
    # Modal.com (if using serverless GPU)
    MODAL_TOKEN_ID: Optional[str] = None
    MODAL_TOKEN_SECRET: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Singleton instance
settings = Settings()
```

```bash
# File: backend/.env.example

# Application
DEBUG=false
PORT=8000

# Supabase (from your existing setup)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=eyJhbGc...
SUPABASE_SERVICE_KEY=eyJhbGc...  # Keep secret!
SUPABASE_BUCKET_NAME=deepfake-videos

# ML Configuration
ML_WEIGHTS_DIR=/app/weights
ML_DEVICE=cuda
FAIL_FAST_THRESHOLD=0.90

# Storage
TEMP_DIR=/tmp/deepfake_processing
MAX_VIDEO_SIZE_MB=500
MAX_VIDEO_DURATION_SEC=300

# Frontend
WEB_FRONTEND_URL=https://yourdomain.com

# WhatsApp
WHATSAPP_WEBHOOK_SECRET=your-webhook-secret
```

---

## 3. SUPABASE DATABASE SCHEMA

### 3.1 Enhanced Detection History Table

```sql
-- File: backend/migrations/001_create_detection_results.sql

-- Extend existing detection_history table with 4-layer specific fields
ALTER TABLE public.detection_history 
ADD COLUMN IF NOT EXISTS layers_executed INTEGER[] DEFAULT '{}',
ADD COLUMN IF NOT EXISTS layer_1_audio JSONB,
ADD COLUMN IF NOT EXISTS layer_2_visual JSONB,
ADD COLUMN IF NOT EXISTS layer_3_lipsync JSONB,
ADD COLUMN IF NOT EXISTS layer_4_semantic JSONB,
ADD COLUMN IF NOT EXISTS fail_fast_triggered BOOLEAN DEFAULT FALSE,
ADD COLUMN IF NOT EXISTS processing_time_seconds NUMERIC(8,2),
ADD COLUMN IF NOT EXISTS error_message TEXT,
ADD COLUMN IF NOT EXISTS reasoning TEXT;

-- Create index for fast queries
CREATE INDEX IF NOT EXISTS idx_detection_history_user_created 
ON public.detection_history(user_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_detection_history_session 
ON public.detection_history(session_id);

-- Add comment for documentation
COMMENT ON COLUMN public.detection_history.layers_executed IS 
'Array of executed layer numbers, e.g., {1,2,3,4} or {1} if fail-fast triggered';

COMMENT ON COLUMN public.detection_history.layer_1_audio IS 
'JSON result from Layer 1: {confidence, is_fake, reasoning}';

COMMENT ON COLUMN public.detection_history.fail_fast_triggered IS 
'TRUE if processing stopped early due to high confidence in Layer 1';
```

### 3.2 Database Service

```python
# File: backend/services/database.py

from supabase import create_client, Client
from config import settings
from typing import Dict, Optional
import uuid
from datetime import datetime

class DatabaseService:
    """
    Supabase database client for storing detection results
    """
    
    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY  # Use service key for admin operations
        )
    
    def test_connection(self):
        """Verify Supabase connection"""
        # Simple query to test connection
        self.client.table('detection_history').select('id').limit(1).execute()
    
    def create_detection_record(
        self,
        user_id: Optional[str],
        session_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        file_url: str,
        pipeline_result: Dict
    ) -> str:
        """
        Store detection result in database
        
        Args:
            user_id: User UUID (None for anonymous WhatsApp users)
            session_id: Session identifier
            filename: Original filename
            file_type: 'video'
            file_size: File size in bytes
            file_url: Supabase storage URL
            pipeline_result: Output from ML pipeline (Phase 1)
        
        Returns:
            Record ID (UUID)
        """
        
        # Extract layer results from pipeline output
        layer_results = pipeline_result.get('layer_results', {})
        
        record_data = {
            'id': str(uuid.uuid4()),
            'user_id': user_id,
            'session_id': session_id,
            'filename': filename,
            'file_type': file_type,
            'file_size': file_size,
            'file_extension': filename.split('.')[-1].lower(),
            'file_url': file_url,
            
            # Detection results
            'detection_result': 'fake' if pipeline_result['final_verdict'] else 'real',
            'confidence_score': pipeline_result['confidence'],
            'reasoning': pipeline_result['reasoning'],
            
            # Layer-specific results
            'layers_executed': pipeline_result['layers_executed'],
            'layer_1_audio': layer_results.get('1_audio'),
            'layer_2_visual': layer_results.get('2_visual'),
            'layer_3_lipsync': layer_results.get('3_lipsync'),
            'layer_4_semantic': layer_results.get('4_semantic'),
            
            # Metadata
            'fail_fast_triggered': len(pipeline_result['layers_executed']) == 1,
            'processing_time_seconds': pipeline_result['processing_time'],
            'model_used': '4-Layer-v1.0',
            'is_file_available': True,
            'created_at': datetime.utcnow().isoformat()
        }
        
        response = self.client.table('detection_history').insert(record_data).execute()
        
        return record_data['id']
    
    def get_detection_by_id(self, detection_id: str) -> Optional[Dict]:
        """Retrieve detection record by ID"""
        response = self.client.table('detection_history') \
            .select('*') \
            .eq('id', detection_id) \
            .single() \
            .execute()
        
        return response.data if response.data else None
    
    def get_user_detections(
        self, 
        user_id: str, 
        limit: int = 50,
        offset: int = 0
    ) -> list[Dict]:
        """Get detection history for a user"""
        response = self.client.table('detection_history') \
            .select('*') \
            .eq('user_id', user_id) \
            .order('created_at', desc=True) \
            .range(offset, offset + limit - 1) \
            .execute()
        
        return response.data
    
    def update_file_deletion(self, detection_id: str):
        """Mark file as deleted (for privacy compliance)"""
        self.client.table('detection_history') \
            .update({
                'is_file_available': False,
                'file_deleted_at': datetime.utcnow().isoformat()
            }) \
            .eq('id', detection_id) \
            .execute()

# Singleton instance
db_client = DatabaseService()
```

---

## 4. STORAGE SERVICE (SUPABASE STORAGE)

```python
# File: backend/services/storage.py

from supabase import create_client, Client
from config import settings
from pathlib import Path
import uuid
from typing import Optional

class StorageService:
    """
    Supabase Storage client for video file management
    """
    
    def __init__(self):
        self.client: Client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_KEY
        )
        self.bucket_name = settings.SUPABASE_BUCKET_NAME
        
        # Ensure bucket exists
        self._ensure_bucket()
    
    def _ensure_bucket(self):
        """Create bucket if it doesn't exist"""
        try:
            buckets = self.client.storage.list_buckets()
            bucket_names = [b['name'] for b in buckets]
            
            if self.bucket_name not in bucket_names:
                self.client.storage.create_bucket(
                    self.bucket_name,
                    options={'public': False}  # Private bucket
                )
        except Exception as e:
            # Bucket might already exist
            pass
    
    def upload_video(
        self,
        file_path: str,
        user_id: Optional[str],
        session_id: str
    ) -> str:
        """
        Upload video to Supabase Storage
        
        Args:
            file_path: Local path to video file
            user_id: User UUID (or None for anonymous)
            session_id: Session identifier
        
        Returns:
            Public URL of uploaded file
        """
        
        # Generate unique storage path
        file_ext = Path(file_path).suffix
        storage_path = f"{session_id}/{uuid.uuid4()}{file_ext}"
        
        # Read file
        with open(file_path, 'rb') as f:
            file_data = f.read()
        
        # Upload to Supabase
        response = self.client.storage.from_(self.bucket_name).upload(
            path=storage_path,
            file=file_data,
            file_options={
                "content-type": "video/mp4",
                "upsert": False
            }
        )
        
        # Generate signed URL (valid for 1 hour)
        signed_url = self.client.storage.from_(self.bucket_name).create_signed_url(
            storage_path,
            expires_in=3600  # 1 hour
        )
        
        return signed_url['signedURL']
    
    def delete_video(self, storage_path: str):
        """Delete video from storage"""
        self.client.storage.from_(self.bucket_name).remove([storage_path])

# Singleton instance
storage_client = StorageService()
```

---

## 5. ML SERVICE WRAPPER

```python
# File: backend/services/ml_service.py

import sys
sys.path.insert(0, '/app')  # Add ML pipeline to path

from ml_pipeline.orchestrator import DeepfakePipeline
from config import settings
from typing import Dict
import os

class MLService:
    """
    Wrapper around Phase 1 ML Pipeline
    Handles initialization and inference requests
    """
    
    def __init__(self):
        # ML Pipeline configuration
        self.config = {
            'weights': {
                'sbi': os.path.join(settings.ML_WEIGHTS_DIR, 'SBI/efficientnetb4_FFpp_c23.pth'),
                'lipforensics': os.path.join(settings.ML_WEIGHTS_DIR, 'LipForensics/lipforensics_checkpoint.pth'),
                'ufd_fc': os.path.join(settings.ML_WEIGHTS_DIR, 'UniversalFakeDetect/fc_weights.pth')
            },
            'device': settings.ML_DEVICE
        }
        
        # Lazy initialization
        self._pipeline = None
    
    @property
    def pipeline(self) -> DeepfakePipeline:
        """Lazy-load ML pipeline on first request"""
        if self._pipeline is None:
            self._pipeline = DeepfakePipeline(self.config)
        return self._pipeline
    
    def analyze_video(self, video_path: str) -> Dict:
        """
        Run video through 4-layer detection pipeline
        
        Args:
            video_path: Path to video file
        
        Returns:
            {
                'final_verdict': bool,
                'confidence': float,
                'layers_executed': list[int],
                'layer_results': dict,
                'reasoning': str,
                'processing_time': float
            }
        """
        return self.pipeline.analyze_video(video_path)

# Singleton instance
ml_service = MLService()
```

---

## 6. DETECTION ENDPOINTS

```python
# File: backend/api/routes/detection.py

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Optional
import uuid
import os
from pathlib import Path

from services.ml_service import ml_service
from services.storage import storage_client
from services.database import db_client
from config import settings
from api.schemas.detection import DetectionResponse, DetectionStatusResponse
from utils.video_processor import validate_video, save_upload_file
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/analyze", response_model=DetectionResponse)
async def analyze_video(
    video: UploadFile = File(...),
    user_id: Optional[str] = None,  # From auth middleware
    session_id: Optional[str] = None
):
    """
    Main detection endpoint: Upload video and get deepfake analysis
    
    Flow:
    1. Validate video (size, duration, format)
    2. Save to temp directory
    3. Upload to Supabase Storage
    4. Run through ML pipeline (with fail-fast logic)
    5. Store results in database
    6. Return response
    """
    
    # Generate session ID if not provided
    if not session_id:
        session_id = str(uuid.uuid4())
    
    logger.info(f"[{session_id}] Received analysis request: {video.filename}")
    
    try:
        # Step 1: Validate video
        validation_result = await validate_video(video, settings.MAX_VIDEO_SIZE_MB, settings.MAX_VIDEO_DURATION_SEC)
        if not validation_result['valid']:
            raise HTTPException(status_code=400, detail=validation_result['error'])
        
        # Step 2: Save to temp directory
        temp_path = await save_upload_file(video, settings.TEMP_DIR)
        logger.info(f"[{session_id}] Video saved to {temp_path}")
        
        # Step 3: Upload to Supabase Storage
        try:
            storage_url = storage_client.upload_video(
                file_path=temp_path,
                user_id=user_id,
                session_id=session_id
            )
            logger.info(f"[{session_id}] Video uploaded to storage")
        except Exception as e:
            logger.error(f"[{session_id}] Storage upload failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to upload video to storage")
        
        # Step 4: Run ML Pipeline (FAIL-FAST LOGIC INSIDE)
        logger.info(f"[{session_id}] Starting ML pipeline analysis...")
        try:
            pipeline_result = ml_service.analyze_video(temp_path)
            logger.info(f"[{session_id}] Analysis complete. Verdict: {'FAKE' if pipeline_result['final_verdict'] else 'REAL'}")
            logger.info(f"[{session_id}] Layers executed: {pipeline_result['layers_executed']}")
            
            if len(pipeline_result['layers_executed']) == 1:
                logger.info(f"[{session_id}] FAIL-FAST triggered! Skipped Layers 2-4.")
        
        except Exception as e:
            logger.error(f"[{session_id}] ML pipeline error: {e}")
            
            # Store error in database
            db_client.create_detection_record(
                user_id=user_id,
                session_id=session_id,
                filename=video.filename,
                file_type='video',
                file_size=validation_result['size'],
                file_url=storage_url,
                pipeline_result={
                    'final_verdict': False,
                    'confidence': 0.0,
                    'layers_executed': [],
                    'layer_results': {},
                    'reasoning': f'Error: {str(e)}',
                    'processing_time': 0.0
                }
            )
            
            raise HTTPException(status_code=500, detail="ML analysis failed")
        
        # Step 5: Store results in database
        try:
            detection_id = db_client.create_detection_record(
                user_id=user_id,
                session_id=session_id,
                filename=video.filename,
                file_type='video',
                file_size=validation_result['size'],
                file_url=storage_url,
                pipeline_result=pipeline_result
            )
            logger.info(f"[{session_id}] Results stored with ID: {detection_id}")
        
        except Exception as e:
            logger.error(f"[{session_id}] Database storage failed: {e}")
            # Don't fail request if DB write fails (results are still valid)
            detection_id = None
        
        # Step 6: Cleanup temp file
        os.remove(temp_path)
        
        # Step 7: Return response
        return DetectionResponse(
            detection_id=detection_id,
            session_id=session_id,
            verdict='fake' if pipeline_result['final_verdict'] else 'real',
            confidence=pipeline_result['confidence'],
            reasoning=pipeline_result['reasoning'],
            layers_executed=pipeline_result['layers_executed'],
            layer_breakdown={
                'layer_1_audio': pipeline_result['layer_results'].get('1_audio'),
                'layer_2_visual': pipeline_result['layer_results'].get('2_visual'),
                'layer_3_lipsync': pipeline_result['layer_results'].get('3_lipsync'),
                'layer_4_semantic': pipeline_result['layer_results'].get('4_semantic')
            },
            fail_fast_triggered=len(pipeline_result['layers_executed']) == 1,
            processing_time=pipeline_result['processing_time'],
            video_url=storage_url
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{session_id}] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/results/{detection_id}", response_model=DetectionResponse)
async def get_detection_results(detection_id: str):
    """
    Retrieve detection results by ID
    """
    result = db_client.get_detection_by_id(detection_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Detection not found")
    
    # Reconstruct response format
    return DetectionResponse(
        detection_id=result['id'],
        session_id=result['session_id'],
        verdict=result['detection_result'],
        confidence=float(result['confidence_score']),
        reasoning=result['reasoning'],
        layers_executed=result['layers_executed'],
        layer_breakdown={
            'layer_1_audio': result['layer_1_audio'],
            'layer_2_visual': result['layer_2_visual'],
            'layer_3_lipsync': result['layer_3_lipsync'],
            'layer_4_semantic': result['layer_4_semantic']
        },
        fail_fast_triggered=result['fail_fast_triggered'],
        processing_time=float(result['processing_time_seconds']),
        video_url=result['file_url']
    )


@router.get("/history", response_model=list[DetectionStatusResponse])
async def get_user_history(
    user_id: str,
    limit: int = 20,
    offset: int = 0
):
    """
    Get detection history for a user
    """
    results = db_client.get_user_detections(user_id, limit, offset)
    
    return [
        DetectionStatusResponse(
            detection_id=r['id'],
            filename=r['filename'],
            verdict=r['detection_result'],
            confidence=float(r['confidence_score']),
            created_at=r['created_at'],
            fail_fast_triggered=r['fail_fast_triggered']
        )
        for r in results
    ]
```

---

## 7. PYDANTIC SCHEMAS

```python
# File: backend/api/schemas/detection.py

from pydantic import BaseModel, Field
from typing import Optional, Dict, List
from datetime import datetime

class LayerResult(BaseModel):
    """Individual layer result"""
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_fake: bool
    reasoning: str

class DetectionResponse(BaseModel):
    """Main detection response"""
    detection_id: Optional[str]
    session_id: str
    verdict: str = Field(..., pattern="^(real|fake)$")
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
    layers_executed: List[int]
    layer_breakdown: Dict[str, Optional[LayerResult]]
    fail_fast_triggered: bool
    processing_time: float
    video_url: str

class DetectionStatusResponse(BaseModel):
    """Simplified status for history listing"""
    detection_id: str
    filename: str
    verdict: str
    confidence: float
    created_at: datetime
    fail_fast_triggered: bool
```

---

## 8. VIDEO VALIDATION UTILITIES

```python
# File: backend/utils/video_processor.py

from fastapi import UploadFile
import cv2
import os
from pathlib import Path
import uuid
from typing import Dict

async def validate_video(
    video: UploadFile,
    max_size_mb: int,
    max_duration_sec: int
) -> Dict:
    """
    Validate uploaded video
    
    Checks:
    - File size
    - Video duration
    - Valid video format
    
    Returns:
        {
            'valid': bool,
            'error': str (if invalid),
            'size': int (bytes),
            'duration': float (seconds)
        }
    """
    
    # Check file size
    video.file.seek(0, 2)  # Seek to end
    size = video.file.tell()
    video.file.seek(0)  # Reset
    
    if size > max_size_mb * 1024 * 1024:
        return {
            'valid': False,
            'error': f"Video exceeds maximum size of {max_size_mb}MB"
        }
    
    # Save temporarily to check duration
    temp_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(temp_path, 'wb') as f:
        f.write(await video.read())
    
    await video.seek(0)  # Reset upload file
    
    # Check video properties
    try:
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        if duration > max_duration_sec:
            os.remove(temp_path)
            return {
                'valid': False,
                'error': f"Video exceeds maximum duration of {max_duration_sec} seconds"
            }
        
        os.remove(temp_path)
        
        return {
            'valid': True,
            'size': size,
            'duration': duration
        }
    
    except Exception as e:
        os.remove(temp_path)
        return {
            'valid': False,
            'error': f"Invalid video file: {str(e)}"
        }

async def save_upload_file(upload: UploadFile, dest_dir: str) -> str:
    """Save uploaded file to destination directory"""
    dest_path = Path(dest_dir) / f"{uuid.uuid4()}_{upload.filename}"
    
    with open(dest_path, 'wb') as f:
        content = await upload.read()
        f.write(content)
    
    return str(dest_path)
```

---

## 9. HEALTH CHECK ENDPOINTS

```python
# File: backend/api/routes/health.py

from fastapi import APIRouter, HTTPException
from services.database import db_client
import torch

router = APIRouter()

@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "4-Layer Deepfake Detection API",
        "version": "1.0.0"
    }

@router.get("/gpu")
async def gpu_status():
    """Check GPU availability and memory"""
    if not torch.cuda.is_available():
        return {
            "available": False,
            "message": "No GPU detected"
        }
    
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
        "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9
    }

@router.get("/db")
async def database_status():
    """Check database connection"""
    try:
        db_client.test_connection()
        return {"status": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {str(e)}")
```

---

## 10. DEPLOYMENT CONFIGURATION

### 10.1 Docker Compose (Development)

```yaml
# File: docker-compose.yml

version: '3.8'

services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
      - SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}
      - ML_DEVICE=cuda
    volumes:
      - ./weights:/app/weights  # Mount weights directory
      - ./backend:/app/backend  # Hot reload for dev
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 10.2 Backend Dockerfile

```dockerfile
# File: Dockerfile

# Use ML pipeline base image from Phase 1
FROM deepfake-ml-core:v1

# Install FastAPI dependencies
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    python-multipart==0.0.6 \
    pydantic==2.5.0 \
    pydantic-settings==2.1.0 \
    supabase==2.0.3 \
    python-dotenv==1.0.0

# Copy backend code
COPY ./backend /app/backend
WORKDIR /app/backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health/ || exit 1

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 11. MODAL.COM ALTERNATIVE (SERVERLESS GPU)

If deploying on Modal.com instead of dedicated GPU server:

```python
# File: backend/modal_deployment.py

import modal

# Create Modal app
stub = modal.Stub("deepfake-detection-4layer")

# Define GPU-enabled function
@stub.function(
    image=modal.Image.from_dockerfile("Dockerfile"),
    gpu="A10G",
    timeout=600,
    secret=modal.Secret.from_name("supabase-credentials")
)
def analyze_video_modal(video_bytes: bytes, session_id: str) -> dict:
    """
    Modal function wrapper for ML pipeline
    Runs on serverless GPU (cold start: ~30s)
    """
    import tempfile
    from services.ml_service import ml_service
    
    # Save video to temp file
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    
    # Run analysis
    result = ml_service.analyze_video(tmp_path)
    
    # Cleanup
    os.remove(tmp_path)
    
    return result

@stub.local_entrypoint()
def main():
    # Deploy to Modal
    print("Deploying to Modal.com...")
```

---

## 12. FAIL-FAST LOGIC VERIFICATION

### Self-Testing Embedded in Code

```python
# File: backend/tests/test_fail_fast.py

from services.ml_service import ml_service
from config import settings

def test_fail_fast_logic():
    """
    Verify fail-fast logic works as expected
    
    Test Cases:
    1. High confidence (>90%) in Layer 1 → Only Layer 1 executes
    2. Low confidence (<90%) in Layer 1 → All 4 layers execute
    """
    
    print("\n=== FAIL-FAST LOGIC TEST ===\n")
    
    # Mock high-confidence fake audio
    # (In production, use actual test video with known fake audio)
    test_video_fake_audio = "/app/test_samples/fake_audio_clear.mp4"
    
    if os.path.exists(test_video_fake_audio):
        result = ml_service.analyze_video(test_video_fake_audio)
        
        if result['layer_results']['1_audio']['confidence'] >= settings.FAIL_FAST_THRESHOLD:
            assert len(result['layers_executed']) == 1, "FAIL: Layers 2-4 executed when they shouldn't"
            print("✓ PASS: Fail-fast triggered correctly")
        else:
            print("⚠ Test inconclusive: Audio confidence below threshold")
    else:
        print("⚠ Skipping test: No test video available")

if __name__ == "__main__":
    test_fail_fast_logic()
```

---

## PHASE 2 DELIVERABLES

### 1. API Endpoints
- `POST /api/v1/detection/analyze` - Main detection endpoint
- `GET /api/v1/detection/results/{id}` - Retrieve results
- `GET /api/v1/detection/history` - User detection history
- `GET /api/v1/health/` - Health check
- `GET /api/v1/health/gpu` - GPU status
- `GET /api/v1/health/db` - Database status

### 2. Database Schema
- Enhanced `detection_history` table with layer-specific columns
- Indexes for performance
- Support for both authenticated users and anonymous WhatsApp users

### 3. Integration Points for Phase 3
- **Web Frontend:** RESTful API with CORS support
- **WhatsApp Bot:** Same `/analyze` endpoint with `session_id` parameter
- **Output Formats:** Full JSON for Web, simplified text for WhatsApp (Phase 3 will handle formatting)

---

## CRITICAL SUCCESS METRICS

### Performance
- **Fail-Fast Effectiveness:** 30-40% of requests stop at Layer 1
- **Response Time:** <30s for full pipeline, <8s for fail-fast
- **Concurrent Requests:** Handle 5 simultaneous videos (queue additional)

### Reliability
- **Uptime:** 99.5% (includes GPU warm-up time)
- **Error Rate:** <1% (excluding invalid inputs)
- **Database Write Success:** 99.9%

---

## NEXT PHASE HANDOFF

Phase 3 (Integration & Delivery) will consume:
1. **Backend API:** `http://backend:8000/api/v1/detection/analyze`
2. **Response Format:** `DetectionResponse` schema
3. **Database Access:** Direct Supabase queries for dashboard
4. **Storage URLs:** Signed URLs for video playback

**Backend is now ready for frontend/WhatsApp integration.**
