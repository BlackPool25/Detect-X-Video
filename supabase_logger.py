"""
Supabase Integration for 4-Layer Deepfake Detection Pipeline
Stores detection results with layer-wise metadata
"""

import os
from typing import Optional, Dict, List
from datetime import datetime
from dataclasses import asdict
import json

from supabase import create_client, Client
from pipeline_production import PipelineResult, DetectionResult


class SupabaseDetectionLogger:
    """
    Logs detection results to Supabase database
    """
    
    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None
    ):
        """
        Initialize Supabase client
        
        Args:
            supabase_url: Supabase project URL (or set SUPABASE_URL env var)
            supabase_key: Supabase publishable key (or set SUPABASE_KEY env var)
        """
        self.url = supabase_url or os.getenv("SUPABASE_URL")
        self.key = supabase_key or os.getenv("SUPABASE_KEY")
        
        if not self.url or not self.key:
            raise ValueError(
                "Supabase credentials required. Set SUPABASE_URL and SUPABASE_KEY "
                "environment variables or pass them to constructor."
            )
        
        self.client: Client = create_client(self.url, self.key)
    
    def log_detection(
        self,
        result: PipelineResult,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        file_url: Optional[str] = None
    ) -> Dict:
        """
        Log detection result to Supabase
        
        Args:
            result: PipelineResult from pipeline
            user_id: Optional user UUID
            session_id: Optional session identifier
            file_url: Optional URL to stored video file
        
        Returns:
            Database insert response
        """
        # Prepare layer-wise results
        layer_details = []
        for layer_result in result.layer_results:
            layer_details.append({
                "layer_name": layer_result.layer_name,
                "is_fake": layer_result.is_fake,
                "confidence": layer_result.confidence,
                "processing_time": layer_result.processing_time,
                "details": layer_result.details
            })
        
        # Extract file metadata
        from pathlib import Path
        video_path = Path(result.video_path)
        
        file_size = video_path.stat().st_size if video_path.exists() else None
        file_extension = video_path.suffix.lstrip('.')
        
        # Prepare database record
        detection_record = {
            "user_id": user_id,
            "session_id": session_id,
            "filename": video_path.name,
            "file_type": "video",
            "file_size": file_size,
            "file_extension": file_extension,
            "file_url": file_url,
            "model_used": "4-Layer-Cascade-v1",
            "detection_result": result.final_verdict,
            "confidence_score": result.confidence,
            "metadata": {
                "stopped_at_layer": result.stopped_at_layer,
                "total_processing_time": result.total_time,
                "layer_results": layer_details,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Insert into detection_history table
        response = self.client.table("detection_history").insert(detection_record).execute()
        
        return response.data[0] if response.data else {}
    
    def get_detection_history(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """
        Retrieve detection history from database
        
        Args:
            user_id: Filter by user ID
            session_id: Filter by session ID
            limit: Maximum number of records to return
        
        Returns:
            List of detection records
        """
        query = self.client.table("detection_history").select("*")
        
        if user_id:
            query = query.eq("user_id", user_id)
        
        if session_id:
            query = query.eq("session_id", session_id)
        
        response = query.order("created_at", desc=True).limit(limit).execute()
        
        return response.data
    
    def get_statistics(
        self,
        user_id: Optional[str] = None,
        days: int = 30
    ) -> Dict:
        """
        Get detection statistics
        
        Args:
            user_id: Optional user filter
            days: Number of days to analyze
        
        Returns:
            Statistics dictionary
        """
        from datetime import timedelta
        
        # Get records from last N days
        cutoff_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        
        query = self.client.table("detection_history").select("*")
        query = query.gte("created_at", cutoff_date)
        
        if user_id:
            query = query.eq("user_id", user_id)
        
        response = query.execute()
        records = response.data
        
        # Calculate statistics
        total_detections = len(records)
        fake_count = sum(1 for r in records if r["detection_result"] == "FAKE")
        real_count = total_detections - fake_count
        
        avg_confidence = (
            sum(r["confidence_score"] for r in records) / total_detections
            if total_detections > 0 else 0
        )
        
        # Layer-wise statistics
        layer_stops = {}
        for record in records:
            metadata = record.get("metadata", {})
            stopped_at = metadata.get("stopped_at_layer", "Unknown")
            layer_stops[stopped_at] = layer_stops.get(stopped_at, 0) + 1
        
        return {
            "period_days": days,
            "total_detections": total_detections,
            "fake_count": fake_count,
            "real_count": real_count,
            "fake_percentage": (fake_count / total_detections * 100) if total_detections > 0 else 0,
            "average_confidence": avg_confidence,
            "layer_stop_distribution": layer_stops
        }


# Environment-based initialization helper
def get_logger() -> SupabaseDetectionLogger:
    """
    Get Supabase logger with default credentials from environment
    """
    return SupabaseDetectionLogger(
        supabase_url="https://cjkcwycnetdhumtqthuk.supabase.co",
        supabase_key=os.getenv("SUPABASE_KEY", "sb_publishable_kYQsl9DIOWNzkcZNUojI1w_yIyL70XH")
    )


# Example usage
if __name__ == "__main__":
    # Test connection
    logger = get_logger()
    
    # Get statistics
    stats = logger.get_statistics()
    print("\n=== Detection Statistics ===")
    print(json.dumps(stats, indent=2))
    
    # Get recent history
    history = logger.get_detection_history(limit=5)
    print(f"\n=== Recent Detections ({len(history)}) ===")
    for record in history:
        print(f"\n{record['filename']}:")
        print(f"  Result: {record['detection_result']}")
        print(f"  Confidence: {record['confidence_score']:.2%}")
        print(f"  Model: {record['model_used']}")
        print(f"  Created: {record['created_at']}")
