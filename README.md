# Text-to-Speech Voice Generation System Using Parler-TTS

## Neural Speech Synthesis Pipeline with Voice Conditioning

This project implements a neural text-to-speech (TTS) system using Meta’s Parler-TTS model to generate natural-sounding speech audio from text input. The system allows controlled voice generation by conditioning the model on a descriptive voice prompt, enabling customizable speech synthesis with different speaking styles.

The pipeline demonstrates practical application of generative AI in speech synthesis using Hugging Face Transformers and PyTorch.

---

## Project Overview

Traditional text-to-speech systems generate robotic or uniform speech outputs. Modern neural TTS models like Parler-TTS enable expressive and controllable speech generation.

This project builds a lightweight inference pipeline that:

* Converts text input into natural speech audio
* Allows voice style conditioning using descriptive prompts
* Generates high-quality WAV audio files
* Supports GPU acceleration for faster inference

It demonstrates how generative AI models can be used for real-time speech synthesis applications.

---

## Engineering Objectives

* Load and deploy a pretrained neural TTS model
* Tokenize text and voice conditioning prompts
* Generate speech waveform using transformer-based architecture
* Convert generated audio tensors into playable WAV files
* Enable reproducible audio generation pipeline

---

## Key Features

* Neural text-to-speech generation using Parler-TTS
* Voice conditioning via natural language description
* High-quality waveform generation
* GPU acceleration support (CUDA-enabled inference)
* Audio export in WAV format
* Direct playback using IPython audio interface
* Lightweight inference pipeline using pretrained models

---

## System Architecture

```text id="tts_flow"
Input Text Prompt
        │
        ▼
Voice Description Prompt
        │
        ▼
Tokenization (Text + Style Conditioning)
        │
        ▼
Parler-TTS Transformer Model
        │
        ▼
Audio Waveform Generation
        │
        ▼
WAV File Output
        │
        ▼
Audio Playback
```

---

## Technology Stack

### Machine Learning

* PyTorch
* Hugging Face Transformers
* Parler-TTS model

### Audio Processing

* SoundFile (audio writing)
* NumPy (tensor handling)

### Runtime Environment

* CUDA (GPU acceleration support)
* CPU fallback execution

---

## Model Details

### Parler-TTS Mini

* Model: `parler-tts/parler-tts-mini-v1`
* Type: Neural text-to-speech transformer
* Capability: Voice-conditioned speech synthesis
* Input: Text + voice description
* Output: High-fidelity speech waveform

---

## Implementation Steps

### 1. Install Dependencies

```python id="tts_install"
!pip install git+https://github.com/huggingface/parler-tts.git transformers soundfile -q
```

---

### 2. Import Libraries

```python id="tts_import"
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
```

---

### 3. Device Configuration

```python id="tts_device"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
```

---

### 4. Load Model and Tokenizer

```python id="tts_model"
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
)
```

---

### 5. Input Configuration

#### Text Prompt

```python id="tts_text"
prompt = "My name is Fahad Ansari, I'm from Darbhanga"
```

#### Voice Description

```python id="tts_voice"
description = "Jon's voice is monotone yet slightly slow in delivery, with a very close recording that almost has no background noise."
```

---

### 6. Tokenization

```python id="tts_tokenize"
input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
```

---

### 7. Speech Generation

```python id="tts_generate"
generation = model.generate(
    input_ids=input_ids,
    prompt_input_ids=prompt_input_ids
)
```

---

### 8. Convert to Audio File

```python id="tts_audio"
audio_arr = generation.cpu().numpy().squeeze()

sf.write(
    "parler_tts_output.wav",
    audio_arr,
    model.config.sampling_rate
)
```

---

### 9. Audio Playback (Notebook)

```python id="tts_play"
import IPython
IPython.display.Audio("parler_tts_output.wav")
```

---

## Example Output

### Input Text

```text id="tts_ex1"
My name is Fahad Ansari, I'm from Darbhanga
```

### Voice Style

```text id="tts_ex2"
Monotone, slightly slow delivery, close recording, low background noise
```

### Output

* Generated speech audio file: `parler_tts_output.wav`
* Human-like synthesized speech with controlled voice characteristics

---

## Technical Highlights

### 1. Voice Conditioning

The model uses natural language descriptions to control speech style, tone, and delivery.

### 2. End-to-End Neural Generation

Direct conversion from text to waveform without traditional phoneme pipelines.

### 3. GPU Acceleration

Supports CUDA-based inference for faster audio generation.

### 4. Lightweight Deployment

Runs inference using pretrained model without additional training.

---

## Applications

* AI voice assistants
* Text-to-speech systems
* Accessibility tools for visually impaired users
* Voice-based chatbots
* Content narration systems
* Educational tools for language learning
* Automated audio content generation

---

## Limitations

* Requires significant compute for real-time usage on CPU
* Output quality depends on prompt clarity
* Limited control over fine-grained prosody adjustments
* Model size may impact deployment on low-resource devices

---

## Future Enhancements

* Real-time streaming speech synthesis
* Web-based TTS interface (Streamlit / React)
* Voice cloning capabilities
* Multi-language speech synthesis
* API deployment using FastAPI
* Audio post-processing (noise reduction, normalization)
* Batch text-to-audio generation pipeline

---

## Reproducibility

This project ensures reproducibility through:

* Pretrained open-source model usage
* Fixed inference pipeline
* Deterministic tokenization workflow
* Standard audio output format (WAV)

---

## Intended Use

This project is intended for:

* Software engineering portfolio demonstration
* AI/ML speech synthesis experimentation
* Educational use in generative AI systems
* Prototyping voice-based applications

---

## Author

**Abid Bin Azhar**
Software Engineer 

---

## Citation

If you use this project, please cite:

**Azhar AB.** Neural Text-to-Speech System using Parler-TTS. GitHub, 2026.
