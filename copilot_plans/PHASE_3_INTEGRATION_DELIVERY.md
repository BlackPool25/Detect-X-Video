# PHASE 3: INTEGRATION & DELIVERY - WEB DASHBOARD & WHATSAPP ENDPOINTS

## EXECUTIVE SUMMARY
This phase connects the Backend API (Phase 2) to the existing Next.js Web Dashboard and WhatsApp Bot, implementing differentiated output formats: full layer-by-layer breakdown for Web users, and concise text summaries for WhatsApp users.

---

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────┐
│                    INTEGRATION LAYER                             │
│                                                                  │
│  ┌──────────────────┐              ┌─────────────────────────┐  │
│  │  Next.js Web     │              │   WhatsApp Bot          │  │
│  │  Dashboard       │              │   (Flask)               │  │
│  │                  │              │                         │  │
│  │  - Upload UI     │              │  - Media Handler        │  │
│  │  - Results View  │              │  - Text Formatter       │  │
│  │  - Layer Details │              │  - Status Checker       │  │
│  └────────┬─────────┘              └───────────┬─────────────┘  │
│           │                                    │                 │
│           │         Backend API (Phase 2)      │                 │
│           │         ┌─────────────────┐        │                 │
│           └────────▶│  FastAPI        │◀───────┘                 │
│                     │  /api/v1/       │                          │
│                     │  detection      │                          │
│                     └─────────────────┘                          │
│                              │                                   │
│                              ▼                                   │
│                     ┌─────────────────┐                          │
│                     │  Supabase       │                          │
│                     │  (Shared DB)    │                          │
│                     └─────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## PART A: WEB DASHBOARD INTEGRATION

### 1. NEXT.JS FRONTEND COMPONENTS

#### 1.1 Upload Component with 4-Layer Display

```tsx
// File: AI-Website/components/detection/VideoUploadDetector.tsx

'use client'

import React, { useState } from 'react'
import { Upload, Loader2, AlertCircle, CheckCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { useAuth } from '@/hooks/useAuth'

interface LayerResult {
  confidence: number
  is_fake: boolean
  reasoning: string
}

interface DetectionResult {
  detection_id: string
  session_id: string
  verdict: 'fake' | 'real'
  confidence: number
  reasoning: string
  layers_executed: number[]
  layer_breakdown: {
    layer_1_audio?: LayerResult
    layer_2_visual?: LayerResult
    layer_3_lipsync?: LayerResult
    layer_4_semantic?: LayerResult
  }
  fail_fast_triggered: boolean
  processing_time: number
  video_url: string
}

export function VideoUploadDetector() {
  const { user } = useAuth()
  const [file, setFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [progress, setProgress] = useState(0)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      
      // Validate file type
      if (!selectedFile.type.startsWith('video/')) {
        setError('Please upload a video file')
        return
      }
      
      // Validate file size (500MB limit)
      if (selectedFile.size > 500 * 1024 * 1024) {
        setError('Video file must be smaller than 500MB')
        return
      }
      
      setFile(selectedFile)
      setError(null)
      setResult(null)
    }
  }

  const handleAnalyze = async () => {
    if (!file) return

    setUploading(true)
    setAnalyzing(true)
    setError(null)
    setProgress(10)

    try {
      // Create FormData for multipart upload
      const formData = new FormData()
      formData.append('video', file)
      
      if (user) {
        formData.append('user_id', user.id)
      }

      setProgress(30)

      // Call Backend API (Phase 2)
      const response = await fetch('/api/detection/analyze', {
        method: 'POST',
        body: formData,
        headers: {
          // Don't set Content-Type - browser will set it with boundary
        }
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Analysis failed')
      }

      setProgress(100)
      const data: DetectionResult = await response.json()
      
      setResult(data)
      setAnalyzing(false)
      setUploading(false)

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
      setAnalyzing(false)
      setUploading(false)
      setProgress(0)
    }
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle>4-Layer Deepfake Detection</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-col items-center justify-center border-2 border-dashed rounded-lg p-12">
            <Upload className="w-12 h-12 text-gray-400 mb-4" />
            <input
              type="file"
              accept="video/*"
              onChange={handleFileSelect}
              className="hidden"
              id="video-upload"
            />
            <label htmlFor="video-upload">
              <Button variant="outline" disabled={analyzing}>
                {file ? 'Change Video' : 'Select Video'}
              </Button>
            </label>
            {file && (
              <p className="mt-4 text-sm text-gray-600">
                Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
              </p>
            )}
          </div>

          {file && !result && (
            <Button
              onClick={handleAnalyze}
              disabled={analyzing}
              className="w-full"
            >
              {analyzing ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Analyzing...
                </>
              ) : (
                'Analyze Video'
              )}
            </Button>
          )}

          {analyzing && (
            <div className="space-y-2">
              <Progress value={progress} />
              <p className="text-sm text-gray-600 text-center">
                {progress < 30 ? 'Uploading video...' : 'Running AI analysis...'}
              </p>
            </div>
          )}

          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Results Section - Full Layer Breakdown */}
      {result && (
        <LayerBreakdownDisplay result={result} />
      )}
    </div>
  )
}
```

#### 1.2 Layer-by-Layer Breakdown Component

```tsx
// File: AI-Website/components/detection/LayerBreakdownDisplay.tsx

'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { 
  CheckCircle, 
  XCircle, 
  Music, 
  Eye, 
  Mic, 
  Sparkles,
  Clock,
  Zap
} from 'lucide-react'

interface LayerBreakdownProps {
  result: DetectionResult
}

export function LayerBreakdownDisplay({ result }: LayerBreakdownProps) {
  const layerIcons = {
    1: Music,
    2: Eye,
    3: Mic,
    4: Sparkles
  }

  const layerNames = {
    1: 'Audio Analysis',
    2: 'Visual Artifacts',
    3: 'Lip-Sync Detection',
    4: 'Semantic Analysis'
  }

  const getLayerData = (layerNum: number) => {
    const key = `layer_${layerNum}_${layerNames[layerNum].split(' ')[0].toLowerCase()}`
    return result.layer_breakdown[key]
  }

  return (
    <div className="space-y-6">
      {/* Main Verdict Card */}
      <Card className={`border-2 ${
        result.verdict === 'fake' 
          ? 'border-red-500 bg-red-50' 
          : 'border-green-500 bg-green-50'
      }`}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              {result.verdict === 'fake' ? (
                <>
                  <XCircle className="w-6 h-6 text-red-600" />
                  <span className="text-red-600">DEEPFAKE DETECTED</span>
                </>
              ) : (
                <>
                  <CheckCircle className="w-6 h-6 text-green-600" />
                  <span className="text-green-600">AUTHENTIC VIDEO</span>
                </>
              )}
            </CardTitle>
            <Badge variant={result.verdict === 'fake' ? 'destructive' : 'default'}>
              {(result.confidence * 100).toFixed(1)}% Confidence
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-gray-700">{result.reasoning}</p>
          
          <div className="mt-4 flex gap-4 text-sm text-gray-600">
            <div className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              {result.processing_time.toFixed(1)}s processing time
            </div>
            {result.fail_fast_triggered && (
              <div className="flex items-center gap-1 text-orange-600">
                <Zap className="w-4 h-4" />
                Fast-tracked (high confidence in Layer 1)
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Layer-by-Layer Breakdown */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {result.layers_executed.map((layerNum) => {
          const Icon = layerIcons[layerNum]
          const layerData = getLayerData(layerNum)
          
          if (!layerData) return null

          return (
            <Card key={layerNum} className="border-gray-200">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Icon className="w-5 h-5 text-blue-600" />
                    <CardTitle className="text-base">
                      Layer {layerNum}: {layerNames[layerNum]}
                    </CardTitle>
                  </div>
                  <Badge variant={layerData.is_fake ? 'destructive' : 'default'}>
                    {layerData.is_fake ? 'FAKE' : 'REAL'}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Confidence:</span>
                  <span className="font-semibold">
                    {(layerData.confidence * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${
                      layerData.is_fake ? 'bg-red-500' : 'bg-green-500'
                    }`}
                    style={{ width: `${layerData.confidence * 100}%` }}
                  />
                </div>
                <p className="text-sm text-gray-700 mt-2">
                  {layerData.reasoning}
                </p>
              </CardContent>
            </Card>
          )
        })}

        {/* Skipped Layers (if fail-fast triggered) */}
        {result.fail_fast_triggered && [2, 3, 4].map((layerNum) => (
          <Card key={`skipped-${layerNum}`} className="border-gray-200 opacity-50">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-2">
                {React.createElement(layerIcons[layerNum], { className: 'w-5 h-5 text-gray-400' })}
                <CardTitle className="text-base text-gray-500">
                  Layer {layerNum}: {layerNames[layerNum]}
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <Alert>
                <AlertDescription>
                  Skipped due to high confidence in Layer 1
                </AlertDescription>
              </Alert>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Technical Details */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Detection Details</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="grid grid-cols-2 gap-2">
            <div>
              <span className="text-gray-600">Detection ID:</span>
              <p className="font-mono text-xs">{result.detection_id}</p>
            </div>
            <div>
              <span className="text-gray-600">Session ID:</span>
              <p className="font-mono text-xs">{result.session_id}</p>
            </div>
            <div>
              <span className="text-gray-600">Layers Executed:</span>
              <p>{result.layers_executed.join(', ')}</p>
            </div>
            <div>
              <span className="text-gray-600">Model Version:</span>
              <p>4-Layer-v1.0</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
```

#### 1.3 API Route for Next.js

```typescript
// File: AI-Website/app/api/detection/analyze/route.ts

import { NextRequest, NextResponse } from 'next/server'

const BACKEND_API_URL = process.env.BACKEND_API_URL || 'http://localhost:8000'

export async function POST(req: NextRequest) {
  try {
    // Get FormData from request
    const formData = await req.formData()
    
    // Forward to Backend API (Phase 2)
    const response = await fetch(`${BACKEND_API_URL}/api/v1/detection/analyze`, {
      method: 'POST',
      body: formData,
      // Headers will be automatically set by fetch for FormData
    })

    if (!response.ok) {
      const error = await response.json()
      return NextResponse.json(
        { error: error.detail || 'Analysis failed' },
        { status: response.status }
      )
    }

    const result = await response.json()
    
    return NextResponse.json(result)

  } catch (error) {
    console.error('Detection API error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export const config = {
  api: {
    bodyParser: false, // Important for file uploads
  },
}
```

#### 1.4 Environment Configuration

```bash
# File: AI-Website/.env.local

# Backend API
NEXT_PUBLIC_BACKEND_API_URL=http://localhost:8000
BACKEND_API_URL=http://backend:8000  # Internal Docker network

# Existing Supabase config...
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

---

## PART B: WHATSAPP BOT INTEGRATION

### 2. WHATSAPP MESSAGE HANDLER MODIFICATIONS

#### 2.1 Enhanced Video Processing

```python
# File: whatsapp/message_handler.py (MODIFICATIONS)

import os
import requests
from config import BACKEND_API_URL

def handle_media_message(message, from_number):
    """
    Handle media messages (images, videos, documents)
    Enhanced with 4-Layer detection for videos
    """
    media_type = None
    media_id = None
    
    # Determine media type
    if 'image' in message:
        media_type = 'image'
        media_id = message['image']['id']
    elif 'video' in message:
        media_type = 'video'
        media_id = message['video']['id']
    elif 'document' in message:
        media_type = 'document'
        media_id = message['document']['id']
    
    if not media_id:
        return "Sorry, I couldn't process that media file."
    
    # Download media from WhatsApp
    from whatsapp_service import download_whatsapp_media
    file_content, mime_type, media_data = download_whatsapp_media(media_id)
    
    if not file_content:
        return "Sorry, I couldn't download the media file."
    
    # Get file metadata
    filename = media_data.get('file_name', 'unnamed')
    file_size = len(file_content)
    
    # Upload to Supabase
    from storage_service import (
        upload_to_supabase,
        store_detection_history,
        determine_file_type_and_bucket
    )
    
    file_type, bucket_name = determine_file_type_and_bucket(filename, mime_type)
    
    try:
        # Upload file
        file_url = upload_to_supabase(
            file_content,
            filename,
            bucket_name,
            mime_type
        )
        
        # Store initial detection record
        detection_record = store_detection_history(
            user_id=None,  # Anonymous WhatsApp user
            session_id=from_number,  # Use phone number as session
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            file_url=file_url,
            detection_result='pending',  # Will be updated by callback
            confidence_score=0.0
        )
        
        # ==== ENHANCED: Trigger 4-Layer Detection for Videos ====
        if file_type == 'video':
            try:
                response = trigger_video_detection(
                    video_url=file_url,
                    detection_id=detection_record['id'],
                    from_number=from_number
                )
                
                if response['status'] == 'processing':
                    return (
                        f"✅ Video uploaded successfully!\n\n"
                        f"🎬 Running advanced 4-layer deepfake detection...\n"
                        f"📊 This may take 15-30 seconds.\n\n"
                        f"You'll receive the results shortly."
                    )
                else:
                    raise Exception("Failed to start detection")
            
            except Exception as e:
                print(f"⚠️ Video detection failed: {e}")
                return (
                    f"✅ Video uploaded, but detection failed.\n"
                    f"File: {filename}\n"
                    f"Size: {file_size / 1024 / 1024:.2f} MB"
                )
        
        # For non-video files, return simple confirmation
        return format_media_response(filename, file_type, file_size, file_url)
    
    except Exception as e:
        print(f"Error handling media: {e}")
        return "Sorry, I couldn't process your file."


def trigger_video_detection(video_url: str, detection_id: str, from_number: str) -> dict:
    """
    Trigger 4-layer video detection via Backend API
    
    Args:
        video_url: Supabase storage URL
        detection_id: Database record ID
        from_number: WhatsApp user number
    
    Returns:
        {'status': 'processing', 'task_id': '...'}
    """
    backend_url = os.getenv('BACKEND_API_URL', 'http://backend:8000')
    callback_url = os.getenv('WHATSAPP_CALLBACK_URL')  # e.g., https://your-ngrok.io/api/detection_callback
    
    # Note: We need to make the video URL publicly accessible for Backend
    # Supabase storage URLs should already be public or use signed URLs
    
    response = requests.post(
        f"{backend_url}/api/v1/detection/analyze",
        files={'video': ('video.mp4', requests.get(video_url).content)},
        data={
            'session_id': from_number,
            'user_id': None  # Anonymous
        },
        timeout=10  # Quick timeout - don't wait for processing
    )
    
    response.raise_for_status()
    return response.json()
```

#### 2.2 Detection Callback Handler

```python
# File: whatsapp/app.py (ADD NEW ENDPOINT)

@app.route("/api/detection_callback", methods=["POST"])
def detection_callback():
    """
    Receive detection results from Backend API (Phase 2)
    Update database and send WhatsApp notification
    """
    try:
        data = request.get_json()
        
        detection_id = data.get('detection_id')
        session_id = data.get('session_id')  # WhatsApp phone number
        result = data.get('result')
        
        if not detection_id or not result:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Update database with final results
        from storage_service import get_supabase_client
        supabase = get_supabase_client()
        
        supabase.table('detection_history').update({
            'detection_result': result['verdict'],
            'confidence_score': result['confidence'],
            'reasoning': result['reasoning'],
            'layers_executed': result['layers_executed'],
            'layer_1_audio': result['layer_breakdown'].get('layer_1_audio'),
            'layer_2_visual': result['layer_breakdown'].get('layer_2_visual'),
            'layer_3_lipsync': result['layer_breakdown'].get('layer_3_lipsync'),
            'layer_4_semantic': result['layer_breakdown'].get('layer_4_semantic'),
            'fail_fast_triggered': result['fail_fast_triggered'],
            'processing_time_seconds': result['processing_time']
        }).eq('id', detection_id).execute()
        
        # Send WhatsApp notification with SIMPLIFIED text format
        from whatsapp_service import send_whatsapp_message
        
        whatsapp_message = format_whatsapp_detection_result(result)
        send_whatsapp_message(session_id, whatsapp_message)
        
        return jsonify({'success': True}), 200
    
    except Exception as e:
        print(f"Callback error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def format_whatsapp_detection_result(result: dict) -> str:
    """
    Format detection result for WhatsApp (TEXT SUMMARY)
    
    WhatsApp Constraint: Text-only, no HTML/rich formatting
    Goal: Concise, actionable information
    """
    verdict = result['verdict'].upper()
    confidence = result['confidence'] * 100
    
    # Verdict emoji
    emoji = "🚨" if result['verdict'] == 'fake' else "✅"
    
    # Build message
    message_parts = [
        f"{emoji} *DETECTION COMPLETE*",
        "",
        f"*Verdict:* {verdict}",
        f"*Confidence:* {confidence:.1f}%",
        "",
        f"📝 *Reasoning:*",
        result['reasoning'],
        ""
    ]
    
    # Add layer summary (simplified)
    if result['fail_fast_triggered']:
        message_parts.extend([
            "⚡ *Analysis:* Fast-tracked (high confidence in audio layer)",
            f"🔍 Layers analyzed: {len(result['layers_executed'])}/4"
        ])
    else:
        message_parts.extend([
            f"🔍 *Layers analyzed:* All 4 layers",
            ""
        ])
        
        # Show which layers detected fake (if any)
        fake_indicators = []
        if result['layer_breakdown'].get('layer_1_audio', {}).get('is_fake'):
            fake_indicators.append("Audio")
        if result['layer_breakdown'].get('layer_2_visual', {}).get('is_fake'):
            fake_indicators.append("Visual")
        if result['layer_breakdown'].get('layer_3_lipsync', {}).get('is_fake'):
            fake_indicators.append("Lip-Sync")
        if result['layer_breakdown'].get('layer_4_semantic', {}).get('is_fake'):
            fake_indicators.append("AI Signatures")
        
        if fake_indicators:
            message_parts.append(f"🔴 *Detected in:* {', '.join(fake_indicators)}")
    
    message_parts.extend([
        "",
        f"⏱️ Processed in {result['processing_time']:.1f}s"
    ])
    
    return "\n".join(message_parts)
```

#### 2.3 WhatsApp Configuration Updates

```python
# File: whatsapp/config.py (ADD)

# Backend API Configuration
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")

# Callback URL (must be publicly accessible)
# For local dev, use ngrok: https://abc123.ngrok.io/api/detection_callback
WHATSAPP_CALLBACK_URL = os.getenv("WHATSAPP_CALLBACK_URL")
```

```bash
# File: whatsapp/.env (ADD)

# Backend API
BACKEND_API_URL=http://backend:8000
WHATSAPP_CALLBACK_URL=https://your-ngrok-url.ngrok.io/api/detection_callback
```

---

## PART C: DEPLOYMENT & TESTING

### 3. DOCKER COMPOSE INTEGRATION

```yaml
# File: docker-compose.yml (FULL STACK)

version: '3.8'

services:
  # Backend API (Phase 2)
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
      - SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}
      - ML_DEVICE=cuda
      - DEBUG=false
    volumes:
      - ./weights:/app/weights
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    networks:
      - deepfake-net

  # WhatsApp Bot
  whatsapp:
    build:
      context: ./whatsapp
      dockerfile: Dockerfile
    ports:
      - "5000:5000"
    environment:
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_KEY=${SUPABASE_ANON_KEY}
      - WHATSAPP_ACCESS_TOKEN=${WHATSAPP_ACCESS_TOKEN}
      - WHATSAPP_PHONE_NUMBER_ID=${WHATSAPP_PHONE_NUMBER_ID}
      - BACKEND_API_URL=http://backend:8000
      - WHATSAPP_CALLBACK_URL=${WHATSAPP_CALLBACK_URL}
    depends_on:
      - backend
    networks:
      - deepfake-net

  # Next.js Web App
  webapp:
    build:
      context: ./AI-Website
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_SUPABASE_URL=${SUPABASE_URL}
      - NEXT_PUBLIC_SUPABASE_ANON_KEY=${SUPABASE_ANON_KEY}
      - BACKEND_API_URL=http://backend:8000
    depends_on:
      - backend
    networks:
      - deepfake-net

networks:
  deepfake-net:
    driver: bridge
```

### 4. TESTING CHECKLIST

#### 4.1 Web Dashboard Tests

```typescript
// File: AI-Website/__tests__/detection.test.ts

import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { VideoUploadDetector } from '@/components/detection/VideoUploadDetector'

describe('VideoUploadDetector', () => {
  test('uploads video and displays results', async () => {
    render(<VideoUploadDetector />)
    
    // Select file
    const fileInput = screen.getByLabelText(/select video/i)
    const file = new File(['dummy'], 'test.mp4', { type: 'video/mp4' })
    fireEvent.change(fileInput, { target: { files: [file] } })
    
    // Click analyze
    const analyzeBtn = screen.getByText(/analyze video/i)
    fireEvent.click(analyzeBtn)
    
    // Wait for results
    await waitFor(() => {
      expect(screen.getByText(/DEEPFAKE DETECTED|AUTHENTIC VIDEO/i)).toBeInTheDocument()
    })
    
    // Check layer breakdown is displayed
    expect(screen.getByText(/Layer 1: Audio Analysis/i)).toBeInTheDocument()
  })
  
  test('displays fail-fast indicator', async () => {
    // Mock API response with fail_fast_triggered: true
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({
          verdict: 'fake',
          fail_fast_triggered: true,
          layers_executed: [1]
        })
      })
    )
    
    render(<VideoUploadDetector />)
    // ... trigger analysis ...
    
    await waitFor(() => {
      expect(screen.getByText(/Fast-tracked/i)).toBeInTheDocument()
      expect(screen.getByText(/Skipped due to high confidence/i)).toBeInTheDocument()
    })
  })
})
```

#### 4.2 WhatsApp Bot Tests

```python
# File: whatsapp/tests/test_detection_callback.py

import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_detection_callback_updates_database(client, mocker):
    """Test that callback properly updates database"""
    
    # Mock Supabase update
    mock_supabase = mocker.patch('storage_service.get_supabase_client')
    
    # Mock WhatsApp message sending
    mock_send = mocker.patch('whatsapp_service.send_whatsapp_message')
    
    # Send callback
    response = client.post('/api/detection_callback', json={
        'detection_id': 'test-id-123',
        'session_id': '+1234567890',
        'result': {
            'verdict': 'fake',
            'confidence': 0.87,
            'reasoning': 'Test reasoning',
            'layers_executed': [1, 2, 3, 4],
            'layer_breakdown': {},
            'fail_fast_triggered': False,
            'processing_time': 25.3
        }
    })
    
    assert response.status_code == 200
    assert mock_supabase.called
    assert mock_send.called

def test_whatsapp_message_formatting():
    """Test WhatsApp message format is concise"""
    
    result = {
        'verdict': 'fake',
        'confidence': 0.92,
        'reasoning': 'Strong AI signatures detected',
        'fail_fast_triggered': True,
        'layers_executed': [1],
        'processing_time': 8.5,
        'layer_breakdown': {}
    }
    
    message = format_whatsapp_detection_result(result)
    
    # Check message is text-only
    assert '<' not in message  # No HTML tags
    assert 'FAKE' in message
    assert '92.0%' in message
    assert 'Fast-tracked' in message
    assert len(message) < 500  # Concise
```

---

## PART D: OUTPUT FORMAT COMPARISON

### Web Dashboard Output (Rich UI)
```
┌────────────────────────────────────────────────┐
│ 🚨 DEEPFAKE DETECTED                          │
│ Confidence: 87.3%                             │
│                                               │
│ Reasoning: FAKE detected (87% confidence).    │
│ Indicators: Audio, Visual                     │
│                                               │
│ ⏱️ 18.3s processing • ⚡ Fast-tracked         │
└────────────────────────────────────────────────┘

Layer Breakdown:
┌──────────────────────┬──────────────────────┐
│ 🎵 Layer 1: Audio    │ 👁️ Layer 2: Visual   │
│ FAKE • 94.2%         │ FAKE • 78.5%         │
│ Synthetic voice      │ Manipulation         │
│ patterns detected    │ signatures found     │
├──────────────────────┼──────────────────────┤
│ 🎤 Layer 3: Lip-Sync │ ✨ Layer 4: Semantic │
│ SKIPPED              │ SKIPPED              │
│ (High confidence in  │ (High confidence in  │
│ Layer 1)             │ Layer 1)             │
└──────────────────────┴──────────────────────┘

Detection Details:
• Detection ID: 550e8400-e29b-41d4...
• Layers Executed: 1, 2
• Model: 4-Layer-v1.0
```

### WhatsApp Output (Text Summary)
```
🚨 *DETECTION COMPLETE*

*Verdict:* FAKE
*Confidence:* 87.3%

📝 *Reasoning:*
FAKE detected (87% confidence). Indicators: Audio, Visual

⚡ *Analysis:* Fast-tracked (high confidence in audio layer)
🔍 Layers analyzed: 2/4
🔴 *Detected in:* Audio, Visual

⏱️ Processed in 18.3s
```

**Key Differences:**
- **Web:** Full layer-by-layer breakdown with progress bars, icons, cards
- **WhatsApp:** Concise text with emojis for visual hierarchy, <500 characters
- **Web:** Technical details (detection ID, session ID)
- **WhatsApp:** Actionable summary only

---

## PART E: USER FLOW DIAGRAMS

### Web Dashboard Flow
```
User → Upload Video → Progress Bar (10% → 30% → 100%)
                               ↓
                    Backend API Processing
                               ↓
                    Results Display:
                    - Main Verdict Card
                    - 4 Layer Cards (or "Skipped" cards)
                    - Technical Details
                               ↓
                    User can save/share/re-analyze
```

### WhatsApp Flow
```
User → Send Video to Bot → "✅ Video uploaded! Running detection..."
                                         ↓
                              Backend API Processing
                                         ↓
                              Callback to WhatsApp Bot
                                         ↓
                              Database Update
                                         ↓
                              Send Text Summary:
                              "🚨 DETECTION COMPLETE
                               Verdict: FAKE (87%)
                               ..."
                                         ↓
                              User receives notification
```

---

## CRITICAL SUCCESS METRICS

### Web Dashboard
- **Load Time:** <2s for results page
- **Upload Success Rate:** >99% (excluding network errors)
- **UI Responsiveness:** Layer cards render in <200ms
- **Mobile Compatibility:** Works on devices >375px width

### WhatsApp Bot
- **Message Delivery:** <5s after callback received
- **Message Length:** <500 characters (WhatsApp recommended)
- **Uptime:** 99.5% (ngrok tunneling can introduce downtime)
- **Callback Success:** >95% (network issues may cause retries)

---

## DEPLOYMENT STEPS

### 1. Web Dashboard Deployment

```bash
# Build Next.js app
cd AI-Website
npm run build

# Deploy to Vercel
vercel --prod

# Or Docker
docker build -t deepfake-webapp .
docker run -p 3000:3000 deepfake-webapp
```

### 2. WhatsApp Bot Deployment

```bash
# Use ngrok for local dev
ngrok http 5000

# Set webhook URL in Meta Developer Console:
# https://abc123.ngrok.io/webhook

# Production: Deploy to cloud (AWS/GCP/Azure)
# Ensure WHATSAPP_CALLBACK_URL is publicly accessible
```

### 3. Full Stack Launch

```bash
# Start all services
docker-compose up -d

# Check health
curl http://localhost:8000/api/v1/health
curl http://localhost:5000/health
curl http://localhost:3000/api/health
```

---

## TROUBLESHOOTING

### Issue: WhatsApp doesn't receive results

**Check:**
1. `WHATSAPP_CALLBACK_URL` is publicly accessible
2. Backend API can reach WhatsApp bot (test with `curl`)
3. Database record has correct `session_id` (phone number)

**Fix:**
```bash
# Test callback manually
curl -X POST https://your-ngrok.io/api/detection_callback \
  -H "Content-Type: application/json" \
  -d '{
    "detection_id": "test-123",
    "session_id": "+1234567890",
    "result": {...}
  }'
```

### Issue: Web Dashboard shows "Analysis failed"

**Check:**
1. Backend API is running (`docker ps`)
2. Video file is under 500MB
3. CORS is enabled for frontend domain

**Fix:**
```python
# In backend/main.py, verify CORS config
allow_origins=[
    "http://localhost:3000",  # Add your domain
    "https://yourdomain.com"
]
```

---

## NEXT STEPS (POST-DEPLOYMENT)

1. **Analytics:** Track detection requests, fail-fast rate, average processing time
2. **User Feedback:** Add thumbs up/down for result accuracy
3. **Batch Processing:** Allow Web users to upload multiple videos
4. **WhatsApp Rich Messages:** Explore WhatsApp Business API templates for better formatting
5. **Real-time Status:** WebSocket updates for Web dashboard instead of polling

---

## CONCLUSION

Phase 3 delivers:
- ✅ **Web Dashboard:** Rich, interactive UI with full layer-by-layer breakdown
- ✅ **WhatsApp Bot:** Concise text summaries optimized for mobile messaging
- ✅ **Unified Backend:** Single API serves both interfaces
- ✅ **Differentiated Output:** Format adapts to platform constraints
- ✅ **Production-Ready:** Docker deployment, health checks, error handling

**All 3 phases are now complete and integrated.**
