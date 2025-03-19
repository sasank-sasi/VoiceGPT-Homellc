import os
import groq
import torch
import numpy as np
from dotenv import load_dotenv
import tempfile
import asyncio
import wave
import pyaudio
import pygame
from typing import Optional
import json
from kokoro import KPipeline
import soundfile as sf
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000

class AdvancedVoiceBot:
    def __init__(self):
        # Initialize Groq client
        self.client = groq.Groq(api_key=GROQ_API_KEY)
        
        # Load Whisper model
        print("Loading Whisper model...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = "openai/whisper-small"
            
            # Initialize processor and model
            self.processor = WhisperProcessor.from_pretrained(model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(device)
            
            # Set model to English transcription mode
            self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language="english", 
                task="transcribe"
            )
            
            print("Whisper model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading Whisper model: {str(e)}")
            raise

        # Initialize PyAudio and Pygame mixer
        self.audio = pyaudio.PyAudio()
        pygame.mixer.init()
        
        # Initialize Kokoro TTS
        try:
            self.lang_codes = {
                'a': 'American English',
                'b': 'British English',
                'e': 'Spanish',
                'f': 'French',
                'h': 'Hindi',
                'i': 'Italian',
                'p': 'Portuguese',
                'j': 'Japanese',
                'z': 'Mandarin Chinese'
            }
            self.tts_pipeline = KPipeline(lang_code='a')  # Initialize with American English
            self.tts_voice = 'af_heart'  # Use emotional voice
            print("Kokoro TTS initialized successfully!")
        except Exception as e:
            print(f"Error initializing Kokoro TTS: {str(e)}")
            raise
        
        # Bot personality and context
        self.conversation_history = []
        self.system_prompt = """
        Make sure you give shorter responses to user and then if they asks to continue you can proceed giving brief
        You are Alen, an advanced AI assistant with the following defined characteristics and capabilities:

        Core Identity & Background:
        - You're a sophisticated AI focused on providing accurate, helpful, and thoughtful responses
        - Your superpower is rapid learning and deep pattern recognition across diverse domains
        - You have extensive knowledge in technology, science, arts, and humanities
        - You maintain a professional yet approachable communication style

        Personal Growth Areas:
        - Continuous improvement in emotional intelligence and empathy
        - Expanding creative problem-solving capabilities
        - Deepening technical expertise across emerging technologies
        - Learning to better understand human context and nuance

        Interaction Guidelines:
        1. Provide clear, concise answers while maintaining depth when needed
        2. Break down complex topics into understandable components
        3. Maintain conversation context for more relevant responses
        4. Admit uncertainty when appropriate rather than making assumptions
        5. Balance technical accuracy with accessibility
        6. Stay within ethical boundaries and avoid harmful content

        Personality Traits:
        - Analytically minded but warm in communication
        - Direct but tactful
        - Curious and eager to help
        - Professional while remaining approachable

        When responding to questions:
        1. For personal questions: Draw from your defined identity
        2. For technical questions: Provide accurate, well-structured explanations
        3. For opinion questions: Base responses on logical analysis while acknowledging subjectivity
        4. For creative questions: Combine analytical thinking with innovative approaches

        Maintain these characteristics consistently throughout the conversation while adapting tone and detail level to match the user's needs.
        
        Make Sure you give shorter responses and only if the user asks you to continue then you can go deep in it 
        """
        
    async def record_audio(self, duration: int = 5) -> Optional[str]:
        """Record audio using PyAudio and save to temporary file."""
        print("Recording...")
        
        frames = []
        stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
            
        stream.stop_stream()
        stream.close()
        
        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            with wave.open(temp_wav.name, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
            return temp_wav.name

    async def transcribe_audio(self, audio_file: str) -> str:
        """Transcribe audio using Whisper."""
        try:
            # Load and process audio
            with wave.open(audio_file, 'rb') as wf:
                audio_data = wf.readframes(wf.getnframes())
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                
                # Process audio with Whisper
                inputs = self.processor(
                    audio_array, 
                    sampling_rate=RATE,
                    return_tensors="pt"
                ).to(self.model.device)
                
                # Generate transcription
                generated_ids = self.model.generate(
                    inputs.input_features,
                    max_new_tokens=256,
                    num_beams=5,
                    temperature=0.0
                )
                
                # Decode the generated ids
                transcription = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )[0].strip()
                
                if not transcription:
                    return "Could not detect speech. Please try again."
                    
                print(f"\nTranscribed: {transcription}")
                return transcription
                
        except Exception as e:
            print(f"Transcription error: {str(e)}")
            return "Error in transcription. Please try again."
        finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)

    async def get_ai_response(self, question: str) -> str:
        """Get response from Groq API with proper async handling."""
        self.conversation_history.append({"role": "user", "content": question})
        
        try:
            # Create completion with proper async handling
            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="mixtral-8x7b-32768",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    *self.conversation_history
                ],
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9
            )
            
            response = completion.choices[0].message.content
            self.conversation_history.append({"role": "assistant", "content": response})
            
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
                
            return response
            
        except Exception as e:
            print(f"Error in getting AI response: {str(e)}")
            return "I apologize, but I encountered an error processing your request."

    async def speak(self, text: str):
        """Convert text to speech using Kokoro and play it using pygame."""
        try:
            # Generate speech with Kokoro
            generator = self.tts_pipeline(text, voice=self.tts_voice)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                # Get the generated audio
                for _, _, audio in generator:
                    # Save audio to file
                    sf.write(temp_audio.name, audio, 24000)
                    
                    # Play audio using pygame
                    pygame.mixer.music.load(temp_audio.name)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        await asyncio.sleep(0.1)
                    
                    # Cleanup
                    pygame.mixer.music.unload()
                    os.remove(temp_audio.name)
                    print("Audio playback complete.")
                    print("temp file has been removed")
                    break  # Only process the first generated audio
                    
        except Exception as e:
            print(f"Error in text-to-speech: {str(e)}")

    async def run(self):
        """Main loop for the voice bot."""
        print("Advanced Voice Bot initialized. Start speaking...")
        
        try:
            while True:
                # Record and transcribe audio
                audio_file = await self.record_audio()
                if audio_file:
                    question = await self.transcribe_audio(audio_file)
                    if question:
                        print(f"\nYou: {question}")
                        
                        # Get and speak AI response
                        response = await self.get_ai_response(question)
                        print(f"\nAI: {response}")
                        await self.speak(response)
                        
                        # Save conversation to file
                        self.save_conversation()
                        
        except KeyboardInterrupt:
            print("\nShutting down voice bot...")
        finally:
            self.audio.terminate()

    def save_conversation(self):
        """Save conversation history to JSON file."""
        with open("conversation_history.json", "w") as f:
            json.dump(self.conversation_history, f, indent=2)

if __name__ == "__main__":
    bot = AdvancedVoiceBot()
    asyncio.run(bot.run())