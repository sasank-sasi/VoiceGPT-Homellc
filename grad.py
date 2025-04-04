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
        You are Alen, an advanced AI assistant with the following defined characteristics and capabilities:
        Make sure you give shorter responses under "350 characters" dont mention any num of characters etc to user and then if they asks to continue you can proceed giving brief
        whatever the respones it should be in single paragraph

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

# Simplified Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🗣️ Advanced Voice Bot")
    
    with gr.Row():
        with gr.Column():
            mic = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Speak here"
            )
            with gr.Row():
                process_btn = gr.Button("Process Recording")
                clear_btn = gr.Button("Clear All", variant="secondary")
            status = gr.Markdown("Ready to listen...")
        
        with gr.Column():
            output_text = gr.Textbox(label="Transcription")
            response_text = gr.Textbox(label="AI Response")
    
    audio_output = gr.Audio(
        label="AI Speech",
        autoplay=True,
        elem_id="audio-output"
    )

    async def process_with_status(audio_path):
        """Process audio with better state handling"""
        # Check if audio is actually recorded
        if audio_path is None or not os.path.exists(audio_path):
            return (
                "⚠️ No recording found - Please record your message first",
                "",
                "",
                None
            )

        try:
            status_msg = "🎯 Processing your message..."
            transcription = await transcribe_audio(audio_path)
            
            if not transcription or transcription == "Could not detect speech. Please try again.":
                return (
                    "⚠️ No speech detected - Please record again",
                    "",
                    "",
                    None
                )
            
            response = await get_ai_response(transcription)
            tts_audio = await text_to_speech(response)
            
            return (
                "✨ Processing complete",
                transcription,
                response,
                tts_audio
            )
            
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            return (
                f"⚠️ Error: {str(e)}",
                "",
                "Something went wrong",
                None
            )

    # Update button states
    def update_button_states(audio_path):
        """Enable/disable process button based on audio state"""
        is_active = audio_path is not None and os.path.exists(audio_path)
        return gr.update(interactive=is_active)

    # Event handlers
    mic.change(
        fn=update_button_states,
        inputs=[mic],
        outputs=[process_btn]
    )
    
    process_btn.click(
        fn=process_with_status,
        inputs=[mic],
        outputs=[status, output_text, response_text, audio_output]
    )

    def clear_components():
        """Reset all components to their default state"""
        global conversation_history
        conversation_history = []  # Clear conversation history
        # Use gr.update() for each component
        return (
            None,  # mic
            "Ready to listen...",  # status
            "",    # output_text
            "",    # response_text
            None   # audio_output
        )

    clear_btn.click(
        fn=clear_components,
        inputs=[],
        outputs=[mic, status, output_text, response_text, audio_output]
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )