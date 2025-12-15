-- Migration: Add multimodal detection columns to detection_history table
-- Date: 2025-12-15
-- Description: Add detector_scores and model_metadata JSONB columns for storing individual detector results

-- Add detector_scores column for individual model scores
ALTER TABLE detection_history 
ADD COLUMN IF NOT EXISTS detector_scores JSONB DEFAULT '{}'::jsonb;

-- Add model_metadata column for processing information
ALTER TABLE detection_history 
ADD COLUMN IF NOT EXISTS model_metadata JSONB DEFAULT '{}'::jsonb;

-- Create GIN indexes for efficient JSONB queries
CREATE INDEX IF NOT EXISTS idx_detection_history_detector_scores 
ON detection_history USING GIN (detector_scores);

CREATE INDEX IF NOT EXISTS idx_detection_history_model_metadata 
ON detection_history USING GIN (model_metadata);

-- Add helpful comments
COMMENT ON COLUMN detection_history.detector_scores IS 
'Individual detector scores in format: {"visual_artifacts": 0.12, "temporal_consistency": 0.88, "audio_synthesis": 0.05, "face_quality": 0.95, "final_weighted_score": 0.89}';

COMMENT ON COLUMN detection_history.model_metadata IS 
'Model processing metadata in format: {"models_used": ["MTCNN", "EfficientNet-B7", "Wav2Vec2"], "processing_time_seconds": 12.5, "frames_analyzed": 60, "video_duration_seconds": 30}';
