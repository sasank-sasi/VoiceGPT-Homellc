import soundfile as sf
from kokoro import KPipeline
import torch
import os
import tempfile
from typing import Optional

class KokoroTTS:
    def __init__(self, lang_code: str = 'a', voice: str = 'af_heart'):
        """
        Initialize Kokoro TTS pipeline.
        
        Args:
            lang_code (str): Language code ('a' for American English, 'b' for British English)
            voice (str): Voice ID to use
        """
        # Validate language code
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
        
        if lang_code not in self.lang_codes:
            raise ValueError(f"Invalid language code. Must be one of: {list(self.lang_codes.keys())}")
            
        self.pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        print(f"Kokoro TTS initialized successfully with {self.lang_codes[lang_code]}!")

    def generate_speech(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate speech from text and save to file.
        
        Args:
            text (str): Text to convert to speech
            output_path (str, optional): Path to save audio file. If None, creates temp file
            
        Returns:
            str: Path to the generated audio file
        """
        try:
            # Create generator from pipeline
            generator = self.pipeline(text, voice=self.voice)
            
            # Get the first (and usually only) generated audio
            for _, _, audio in generator:
                if output_path is None:
                    # Create temporary file with .wav extension
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        output_path = temp_file.name
                
                # Save audio to file
                sf.write(output_path, audio, 24000)
                print(f"Audio saved to: {output_path}")
                return output_path
                
        except Exception as e:
            print(f"Error generating speech: {str(e)}")
            return None

def main():
    try:
        # Initialize TTS with American English
        tts = KokoroTTS(lang_code='a')  # 'a' for American English
        
        # Example text
        text = "Hello! This is a test of the Kokoro text to speech system."
        
        # Generate speech
        output_file = tts.generate_speech(text)
        
        if output_file:
            print(f"Generated audio file at: {output_file}")
            
            os.system(f"afplay {output_file}")
            
            try:
                os.remove(output_file)
                print("Temp file cleaned up.")
            except Exception as e:
                print(f"Error cleaning up file: {str(e)}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())