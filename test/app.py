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

# Add custom CSS for better UI
custom_css = """
.container { max-width: 900px; margin: auto; }
.output-box { margin-top: 15px; }
.audio-player { margin-top: 10px; }
audio::-webkit-media-controls-play-button,
audio::-webkit-media-controls-panel { 
    background-color: #f0f0f0; 
}
"""

# Gradio Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🗣️ Advanced Voice Bot")
    
    with gr.Row():
        with gr.Column(scale=1):
            mic = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Speak here",
                interactive=True
            )
            process_btn = gr.Button("🚀 Process Recording", variant="primary")
            status = gr.Markdown("💡 Ready to listen...")
            
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="📝 Transcription",
                lines=2
            )
            response_text = gr.Textbox(
                label="🤖 AI Response",
                lines=3
            )
            
    with gr.Row():
        audio_output = gr.Audio(
            label="🔊 AI Speech",
            autoplay=True,
            show_download_button=False,
            elem_id="audio-output",
            container=True,
            streaming=False,
            elem_classes=["audio-player"]
        )

    async def process_with_status(audio_path):
        """Process audio and yield status updates with correct output format."""
        if audio_path is None:
            return (
                "⚠️ No audio detected",  # status
                "",                      # output_text
                "Please speak first",    # response_text
                None                     # audio_output
            )

        try:
            # Initial status
            status = "🎯 Processing your message..."
            
            # Process audio
            transcription, response, tts_audio = process_audio(audio_path)
            
            # Update transcript
            status = "✨ Got your message!"
            
            if tts_audio and os.path.exists(tts_audio):
                # Update response and trigger audio
                status = "🔊 Speaking..."
                
                # Small delay for audio loading
                await asyncio.sleep(0.5)
                
                try:
                    return (
                        status,
                        transcription,
                        response,
                        tts_audio
                    )
                finally:
                    # Clean up temp file after sending
                    if os.path.exists(tts_audio):
                        os.remove(tts_audio)
            else:
                return (
                    "⚠️ Failed to generate speech",
                    transcription,
                    response,
                    None
                )
                
        except Exception as e:
            print(f"Error in processing: {str(e)}")
            return (
                f"⚠️ Error: {str(e)}",
                "Error processing audio",
                "Something went wrong",
                None
            )

    # Update the CSS to include button styling
    demo.load(css="""
        .primary-btn {
            background: #2196F3;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            margin: 10px 0;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .primary-btn:hover {
            background: #1976D2;
            transform: translateY(-2px);
        }
        .primary-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
    """)

    # Update event handlers
    process_btn.click(
        fn=process_with_status,
        inputs=[mic],
        outputs=[status, output_text, response_text, audio_output],
        show_progress=True
    )

    # Remove automatic processing on recording stop
    mic.stop_recording()

    # Add JavaScript for autoplay using the correct method
    demo.load(js="""
        function autoplaySetup() {
            let lastAudio = null;
            
            function attemptAutoplay(audio) {
                if (!audio) return;
                
                const playPromise = audio.play();
                if (playPromise !== undefined) {
                    playPromise.catch(error => {
                        console.log("Autoplay prevented:", error);
                        const container = audio.parentElement;
                        if (!container.querySelector('.manual-play')) {
                            const button = document.createElement('button');
                            button.className = 'manual-play';
                            button.innerHTML = '▶️ Play Response';
                            button.style.cssText = 'margin: 5px; padding: 5px 10px;';
                            button.onclick = () => audio.play();
                            container.insertBefore(button, audio);
                        }
                    });
                }
            }

            // Watch for new audio elements
            new MutationObserver((mutations) => {
                mutations.forEach(mutation => {
                    mutation.addedNodes.forEach(node => {
                        if (node.tagName === 'AUDIO' && node.closest('#audio-output')) {
                            lastAudio = node;
                            attemptAutoplay(node);
                        }
                    });
                });
            }).observe(document, { childList: true, subtree: true });

            // Handle initial audio element
            const initialAudio = document.querySelector('#audio-output audio');
            if (initialAudio) attemptAutoplay(initialAudio);
        }

        // Initialize after DOM loads
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', autoplaySetup);
        } else {
            autoplaySetup();
        }
    """)

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )