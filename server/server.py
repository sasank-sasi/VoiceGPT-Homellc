from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import asyncio
import uvicorn
import base64
from voice_bot import AdvancedVoiceBot
import tempfile
import os
import wave
import numpy as np
from typing import Optional

app = FastAPI(title="Voice Bot API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bot instance
bot: Optional[AdvancedVoiceBot] = None

class BotStatus(BaseModel):
    status: str
    message: str

@app.post("/start")
async def start_bot():
    """Initialize and start the voice bot"""
    global bot
    try:
        if bot is None:
            bot = AdvancedVoiceBot()
            return {"status": "success", "message": "Voice bot initialized successfully"}
        return {"status": "warning", "message": "Voice bot is already running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
async def stop_bot():
    """Stop and cleanup the voice bot"""
    global bot
    try:
        if bot:
            bot.audio.terminate()
            bot = None
            return {"status": "success", "message": "Voice bot stopped successfully"}
        return {"status": "warning", "message": "Voice bot is not running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/conversation")
async def websocket_endpoint(websocket: WebSocket):
    """Handle real-time voice conversation"""
    global bot
    
    if not bot:
        await websocket.close(code=1000, reason="Voice bot not initialized")
        return
        
    await websocket.accept()
    
    try:
        while True:
            # Start recording when client sends "start_recording"
            command = await websocket.receive_text()
            if command == "start_recording":
                # Record audio
                audio_file = await bot.record_audio(duration=5)
                
                if audio_file:
                    # Transcribe audio
                    transcription = await bot.transcribe_audio(audio_file)
                    await websocket.send_json({
                        "type": "transcription",
                        "text": transcription
                    })
                    
                    # Get AI response
                    if transcription:
                        response = await bot.get_ai_response(transcription)
                        await websocket.send_json({
                            "type": "ai_response",
                            "text": response
                        })
                        
                        # Convert response to speech
                        await bot.speak(response)
                        await websocket.send_json({
                            "type": "audio_complete",
                            "message": "Audio playback completed"
                        })
                
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()

@app.get("/conversation-history")
async def get_conversation_history():
    """Get the current conversation history"""
    global bot
    try:
        if bot:
            return JSONResponse(content=bot.conversation_history)
        raise HTTPException(status_code=404, detail="Voice bot not initialized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global bot
    return {
        "status": "healthy",
        "bot_status": "running" if bot else "stopped"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )