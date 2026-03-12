"""
TADA Text-to-Speech Demo Application

A Gradio-based UI for the TADA (Text-Acoustic Dual Alignment) speech synthesis model.
"""

import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import torch
import torchaudio

from tada.modules.encoder import Encoder
from tada.modules.tada import InferenceOptions, TadaForCausalLM

# Supported languages for multilingual generation
SUPPORTED_LANGUAGES = {
    "English": None,  # Default (no language parameter needed)
    "Arabic": "ar",
    "Chinese": "ch",
    "German": "de",
    "Spanish": "es",
    "French": "fr",
    "Italian": "it",
    "Japanese": "ja",
    "Polish": "pl",
    "Portuguese": "pt",
}

# Model options
MODEL_OPTIONS = {
    "TADA-1B (Faster)": "HumeAI/tada-1b",
    "TADA-3B-ML (Higher Quality, Multilingual)": "HumeAI/tada-3b-ml",
}

# Get the samples directory path
SAMPLES_DIR = Path(__file__).parent / "tada" / "samples"


def load_sample_transcripts() -> dict:
    """Load sample audio transcripts from the JSON file."""
    transcript_path = SAMPLES_DIR / "prompt_transcripts.json"
    if transcript_path.exists():
        with open(transcript_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_sample_audio_files() -> list[str]:
    """Get list of available sample audio files."""
    if SAMPLES_DIR.exists():
        return [
            str(SAMPLES_DIR / f)
            for f in os.listdir(SAMPLES_DIR)
            if f.endswith(".wav") and not f.endswith("_decoded.wav") and not f.endswith("_generated.wav")
        ]
    return []


# Global model cache
_model_cache: dict = {}


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_models(model_name: str, language: str) -> tuple:
    """Load the encoder and model, using cache when possible."""
    global _model_cache

    device = get_device()
    lang_code = SUPPORTED_LANGUAGES.get(language)

    # Create cache key
    cache_key = f"{model_name}_{lang_code}"

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # Load encoder with appropriate language
    if lang_code:
        encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder", language=lang_code).to(device)
    else:
        encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(device)

    # Load model
    model_path = MODEL_OPTIONS[model_name]
    model = TadaForCausalLM.from_pretrained(model_path).to(device)

    _model_cache[cache_key] = (encoder, model)
    return encoder, model


def generate_speech(
    text_to_synthesize: str,
    reference_audio: tuple | str | None,
    reference_transcript: str,
    language: str,
    model_name: str,
    temperature: float,
    cfg_scale: float,
    num_flow_steps: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple[tuple[int, Any] | None, str]:
    """
    Generate speech from text using the TADA model.

    Args:
        text_to_synthesize: The text to convert to speech
        reference_audio: Reference audio for voice cloning (tuple of (sample_rate, audio_data) or file path)
        reference_transcript: Transcript of the reference audio (required for non-English)
        language: Target language for synthesis
        model_name: Which TADA model to use
        temperature: Sampling temperature for text generation
        cfg_scale: Classifier-free guidance scale for acoustics
        num_flow_steps: Number of flow matching steps

    Returns:
        Tuple of (audio_output, status_message)
    """
    if not text_to_synthesize.strip():
        return None, "⚠️ Please enter some text to synthesize."

    if reference_audio is None:
        return None, "⚠️ Please upload a reference audio file or select a sample."

    try:
        progress(0.1, desc="Loading models...")
        device = get_device()
        encoder, model = load_models(model_name, language)

        progress(0.3, desc="Processing reference audio...")

        # Handle different audio input formats
        if isinstance(reference_audio, tuple):
            # Gradio audio component format: (sample_rate, audio_data)
            sample_rate, audio_data = reference_audio
            if isinstance(audio_data, list):
                audio_data = torch.tensor(audio_data)
            else:
                audio_data = torch.from_numpy(audio_data)
            # Ensure correct shape (channels, samples)
            if audio_data.dim() == 1:
                audio_data = audio_data.unsqueeze(0)
            elif audio_data.dim() == 2 and audio_data.shape[1] < audio_data.shape[0]:
                # Shape is (samples, channels), transpose to (channels, samples)
                audio_data = audio_data.T
            audio = audio_data.float().to(device)
        elif isinstance(reference_audio, str):
            # File path
            audio, sample_rate = torchaudio.load(reference_audio)
            audio = audio.to(device)
        else:
            return None, "⚠️ Invalid audio format."

        progress(0.5, desc="Encoding reference audio...")

        # Prepare prompt text (use transcript for non-English or if provided)
        prompt_text = reference_transcript.strip() if reference_transcript.strip() else None
        lang_code = SUPPORTED_LANGUAGES.get(language)

        if lang_code and not prompt_text:
            return None, f"⚠️ For {language}, please provide the transcript of your reference audio."

        # Create prompt from reference audio
        if prompt_text:
            prompt = encoder(audio, text=[prompt_text], sample_rate=sample_rate)
        else:
            prompt = encoder(audio, text=None, sample_rate=sample_rate)

        progress(0.7, desc="Generating speech...")

        # Configure inference options
        inference_options = InferenceOptions(
            text_temperature=temperature,
            acoustic_cfg_scale=cfg_scale,
            num_flow_matching_steps=num_flow_steps,
        )

        # Generate speech
        output = model.generate(
            prompt=prompt,
            text=text_to_synthesize,
            inference_options=inference_options,
        )

        progress(0.9, desc="Processing output...")

        # Get the generated audio
        if output.audio and output.audio[0] is not None:
            generated_audio = output.audio[0].cpu()
            # TADA outputs at 24kHz
            output_sample_rate = 24000
            return (output_sample_rate, generated_audio.numpy()), "✅ Speech generated successfully!"
        else:
            return None, "❌ Failed to generate audio. Please try with different parameters."

    except Exception as e:
        error_msg = str(e)
        if "CUDA out of memory" in error_msg:
            return None, "❌ GPU out of memory. Try using a shorter reference audio or reducing parameters."
        return None, f"❌ Error: {error_msg}"


def load_sample_audio(sample_name: str) -> tuple[str | None, str]:
    """Load a sample audio file and its transcript."""
    if not sample_name or sample_name == "None":
        return None, ""

    transcripts = load_sample_transcripts()
    sample_path = str(SAMPLES_DIR / sample_name)

    if os.path.exists(sample_path):
        transcript = transcripts.get(sample_name, "")
        return sample_path, transcript

    return None, ""


def create_demo() -> gr.Blocks:
    """Create the Gradio demo interface."""
    # Get available sample files
    sample_files = get_sample_audio_files()
    sample_names = ["None"] + [os.path.basename(f) for f in sample_files]

    with gr.Blocks(title="TADA Text-to-Speech") as demo:
        gr.Markdown(
            """
            # 🎙️ TADA: Text-to-Speech Demo

            **TADA** (Text-Acoustic Dual Alignment) is a generative framework for high-fidelity speech synthesis.
            Upload a reference audio to clone the voice, then enter text to synthesize!

            [![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2602.23068)
            [![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue)](https://huggingface.co/spaces/HumeAI/tada)
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Reference Voice")

                # Sample audio selector
                sample_dropdown = gr.Dropdown(
                    choices=sample_names,
                    value="None",
                    label="Use Sample Audio",
                    info="Select a sample audio file or upload your own below",
                )

                # Reference audio input
                reference_audio = gr.Audio(
                    label="Reference Audio",
                    type="numpy",
                    sources=["upload", "microphone"],
                )

                # Reference transcript
                reference_transcript = gr.Textbox(
                    label="Reference Audio Transcript",
                    placeholder="Enter the transcript of your reference audio (required for non-English)",
                    lines=3,
                    info="Required for non-English languages to ensure proper alignment",
                )

                # Language selector
                language = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value="English",
                    label="Language",
                    info="Select the language for synthesis",
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📝 Text to Synthesize")

                # Text input
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=5,
                    max_lines=10,
                )

                # Model selector
                model_selector = gr.Dropdown(
                    choices=list(MODEL_OPTIONS.keys()),
                    value="TADA-3B-ML (Higher Quality, Multilingual)",
                    label="Model",
                    info="Choose between faster (1B) or higher quality (3B) models",
                )

                # Advanced settings
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.6,
                        step=0.1,
                        label="Temperature",
                        info="Higher values = more varied output",
                    )

                    cfg_scale = gr.Slider(
                        minimum=1.0,
                        maximum=3.0,
                        value=1.6,
                        step=0.1,
                        label="CFG Scale",
                        info="Classifier-free guidance scale for acoustics",
                    )

                    num_flow_steps = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5,
                        label="Flow Matching Steps",
                        info="More steps = higher quality but slower",
                    )

                # Generate button
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")

        # Output section
        gr.Markdown("### 🔊 Generated Audio")
        with gr.Row():
            with gr.Column():
                output_audio = gr.Audio(
                    label="Generated Speech",
                    type="numpy",
                    autoplay=False,
                )
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                )

        # Example inputs
        gr.Markdown("### 📚 Examples")
        gr.Examples(
            examples=[
                ["Please call Stella. Ask her to bring these things with her from the store.", "English"],
                ["The quick brown fox jumps over the lazy dog.", "English"],
                ["今日はとても良い天気ですね。散歩に行きましょう。", "Japanese"],
                ["Hallo, wie geht es Ihnen heute? Ich hoffe, Sie haben einen schönen Tag.", "German"],
                ["Hola, ¿cómo estás? Espero que tengas un buen día.", "Spanish"],
            ],
            inputs=[text_input, language],
            label="Click on an example to load it",
        )

        # Event handlers
        def on_sample_select(sample_name):
            if sample_name and sample_name != "None":
                audio_path, transcript = load_sample_audio(sample_name)
                return audio_path, transcript
            return None, ""

        sample_dropdown.change(
            fn=on_sample_select,
            inputs=[sample_dropdown],
            outputs=[reference_audio, reference_transcript],
        )

        generate_btn.click(
            fn=generate_speech,
            inputs=[
                text_input,
                reference_audio,
                reference_transcript,
                language,
                model_selector,
                temperature,
                cfg_scale,
                num_flow_steps,
            ],
            outputs=[output_audio, status_output],
        )

        # Footer
        gr.Markdown(
            """
            ---
            <div class="footer">
            Built with ❤️ using <a href="https://github.com/HumeAI/tada">TADA</a> by
            <a href="https://hume.ai">Hume AI</a>
            </div>
            """,
            elem_classes=["footer"],
        )

    return demo


# Main entry point
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ),
    )
