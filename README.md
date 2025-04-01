# Home LLC Voice Assistant

## Overview
The Home LLC Voice Assistant is an advanced AI-powered voice assistant designed to enable seamless human-computer interaction using speech-to-text (STT), natural language processing (NLP), and text-to-speech (TTS) technologies. It integrates multiple AI models and services to enable real-time voice conversations, making it ideal for home automation, customer support, and accessibility applications.

## Technology Stack

### Models & Services Used

#### Speech-to-Text (STT)
- **OpenAI Whisper (small model)**
  - High-accuracy transcription with support for multiple languages

#### Language Model
- **Groq API with Mixtral-8x7b-32768**
  - Natural language understanding and response generation
  - Context Window: 32k tokens (supports long conversations)

#### Text-to-Speech (TTS)
- **Kokoro TTS Pipeline (Default provider)**
- **Google Cloud Text-to-Speech (Optional, high-quality neural voices)**
  - Voice: "af_heart" (emotional voice)
  - Language: American English

## Process Flow
1. User speaks into the microphone.
2. Whisper transcribes audio to text.
3. Groq Mixtral processes the transcribed text and generates a response.
4. Kokoro TTS converts the generated response into speech.
5. The system plays the synthesized voice output to the user.

---

## Why Use Gradio Instead of Streamlit?

### Challenges with Streamlit:
- Not optimized for real-time audio processing – Streamlit is designed for web-based data visualization and batch processing, making it inefficient for streaming audio interactions.
- Latency issues – Streamlit requires full refreshes for UI updates, which disrupts real-time conversations.
- Lack of native support for audio – It doesn’t have built-in support for live audio input and playback, requiring additional workarounds.

### Advantages of Gradio for This Project:
- Built-in support for real-time audio input & output
- Lower latency with async processing
- Simplified UI with minimal code
- Easier integration with Hugging Face Spaces for deployment
- Supports rapid debugging without restarting the server

---

## Groq LLM Selection Analysis for Home LLC Voice Assistant

### Why Groq?

#### Performance Benefits
- Ultra-low latency (sub-100ms response times)
- Optimized inference hardware
- Consistent performance under load
- Better streaming capability for real-time conversations

#### Key Features of Groq's Mixtral-8x7b-32768

```python
# Example latency comparison
async def compare_response_times():
    start = time.time()
    response = await client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.7
    )
    return time.time() - start  # Typically < 100ms
```

---

## Alternative Models Available on Groq

### 1. Qwen-72B
**Pros:**
- Larger parameter count (72B)
- Strong multilingual capabilities
- Open source and customizable

**Cons:**
- Higher latency than Mixtral
- Larger memory requirements

### 2. DeepSeek-67B
**Pros:**
- Excellent in code generation
- Strong analytical capabilities
- Open source

**Cons:**
- Less optimized for conversational tasks
- Higher token cost per response

---

## Recommended Model Selection Strategy

```python
from enum import Enum
from typing import Dict, Any

class GroqModel(Enum):
    MIXTRAL = "mixtral-8x7b-32768"
    QWEN = "qwen-72b"
    DEEPSEEK = "deepseek-67b"

class ModelSelector:
    def __init__(self):
        self.model_configs: Dict[str, Any] = {
            GroqModel.MIXTRAL.value: {
                "max_tokens": 32768,
                "temperature": 0.7,
                "top_p": 0.9,
                "use_case": "conversation"
            },
            GroqModel.QWEN.value: {
                "max_tokens": 8192,
                "temperature": 0.8,
                "top_p": 0.9,
                "use_case": "multilingual"
            },
            GroqModel.DEEPSEEK.value: {
                "max_tokens": 4096,
                "temperature": 0.6,
                "top_p": 0.9,
                "use_case": "analytical"
            }
        }
    
    def get_optimal_model(self, task_type: str) -> str:
        """Select optimal model based on task requirements"""
        if task_type == "conversation":
            return GroqModel.MIXTRAL.value
        elif task_type == "multilingual":
            return GroqModel.QWEN.value
        elif task_type == "analytical":
            return GroqModel.DEEPSEEK.value
        return GroqModel.MIXTRAL.value  # Default to Mixtral
```

---

## Why We Chose Mixtral-8x7b-32768

### 1. **Optimal Size-Performance Ratio**
- 8x7B architecture provides a good balance
- Efficient resource utilization
- Lower cost per token than larger models

### 2. **Conversation Optimizations**
- Better context handling
- More natural dialogue flow
- Consistent personality maintenance

### 3. **Integration Benefits**
- Better streaming support
- Reliable API performance
- Strong documentation and support

### 4. **Cost Effectiveness**
- Lower compute requirements
- Efficient token usage
- Better pricing for high-volume usage

---

## Future Considerations

1. **Add model switching capability:**
   - Allow dynamic model selection based on task
   - Implement fallback options
   - Monitor performance metrics

2. **Evaluate new models as they become available:**
   - Track performance benchmarks
   - Test multilingual capabilities
   - Compare cost-effectiveness

3. **Consider fine-tuning options:**
   - Custom personality development
   - Domain-specific knowledge
   - Response optimization

---

## Running the Application
To run the Home LLC Voice Assistant using Gradio:

```bash
pip install requirements.txt
python grad.py
```

---

## License
This project is licensed under the MIT License.

---

## Contributors
- **Sai Sasank Bezawada** – Lead Developer

For inquiries, contact: [your email or GitHub link]
