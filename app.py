import os
import groq
import torch
import numpy as np
from dotenv import load_dotenv
import asyncio
import tempfile
import json
import gradio as gr
import pygame
import soundfile as sf
from kokoro import KPipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize Groq client
client = groq.Groq(api_key=GROQ_API_KEY)

# Load Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/whisper-small"

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)

# Set model to English transcription mode
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
    language="english",
    task="transcribe"
)

# Initialize Kokoro TTS
tts_pipeline = KPipeline(lang_code='a')  # American English
tts_voice = 'af_heart'  # Emotional voice

# Initialize pygame for audio playback
pygame.mixer.init()

conversation_history = []

system_prompt = """
        Make sure you give shorter responses under "350 characters" dont mention any num of characters etc to user and then if they asks to continue you can proceed giving brief
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
        
        Make Sure you give shorter responses "350 characters" dont mention any nym odf chracters etc and only if the user asks you to continue then you can go deep in it 
        
"""

async def transcribe_audio(audio_file: str) -> str:
    """Transcribe audio using Whisper."""
    try:
        audio_array, sample_rate = sf.read(audio_file, dtype='float32')
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        
        if sample_rate != 16000:
            from scipy import signal
            target_length = int(len(audio_array) * 16000 / sample_rate)
            audio_array = signal.resample(audio_array, target_length)

        inputs = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(model.device)

        generated_ids = model.generate(
            inputs.input_features,
            max_new_tokens=256,
            num_beams=5,
            temperature=0.0
        )

        transcription = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        return transcription or "Could not detect speech. Please try again."

    except Exception as e:
        print(f"Transcription error: {str(e)}")
        return "Error in transcription."

async def get_ai_response(question: str) -> str:
    """Get response from Groq API."""
    conversation_history.append({"role": "user", "content": question})

    try:
        completion = await asyncio.to_thread(
            client.chat.completions.create,
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": system_prompt},
                *conversation_history
            ],
            temperature=0.7,
            max_tokens=2048,
            top_p=0.9
        )

        response = completion.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": response})

        if len(conversation_history) > 10:
            conversation_history.pop(0)

        return response

    except Exception as e:
        print(f"Error in AI response: {str(e)}")
        return "I encountered an error."

async def text_to_speech(text: str) -> str:
    """Convert text to speech using Kokoro and return audio file path."""
    try:
        generator = tts_pipeline(text, voice=tts_voice)
        temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)

        for _, _, audio in generator:
            sf.write(temp_audio.name, audio, 24000)
            break  # Only process the first generated audio

        return temp_audio.name

    except Exception as e:
        print(f"Error in text-to-speech: {str(e)}")
        return ""

def process_audio(audio_path: str):
    """Pipeline to transcribe, get AI response, convert to speech, and return output."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    transcription = loop.run_until_complete(transcribe_audio(audio_path))
    response = loop.run_until_complete(get_ai_response(transcription))
    tts_audio = loop.run_until_complete(text_to_speech(response))

    return transcription, response, tts_audio

def play_audio(audio_path: str):
    """Play audio file using pygame."""
    if os.path.exists(audio_path):
        pygame.mixer.music.load(audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        pygame.mixer.music.unload()
        os.remove(audio_path)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🗣️ Advanced Voice Bot")

    with gr.Row():
        mic = gr.Audio(
            sources=["microphone"],  # Changed from source to sources
            type="filepath",
            label="🎤 Speak here"
        )
        output_text = gr.Textbox(label="📝 Transcription")
        response_text = gr.Textbox(label="🤖 AI Response")

    with gr.Row():
        audio_file_output = gr.Audio(
            label="🔊 AI Speech Output",
            autoplay=True,
            show_download_button=True
        )

    def on_process(audio_path):
        if audio_path is None:
            return "No audio recorded", "Please speak into the microphone", None
        transcription, response, tts_audio = process_audio(audio_path)
        return transcription, response, tts_audio

    # Update event handlers
    mic.change(
        fn=on_process,
        inputs=[mic],
        outputs=[output_text, response_text, audio_file_output]
    )

# Launch the interface
if __name__ == "__main__":
    demo.queue()  # Enable queuing for better handling of concurrent users
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
