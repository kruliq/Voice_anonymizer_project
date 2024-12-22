"""
Voice Anonymization System
-------------------------
This module implements a voice anonymization system using speech-to-text
and text-to-speech models. It preserves the content of speech while
modifying speaker characteristics.

Main components:
- Audio loading and preprocessing
- Speech transcription
- Text-to-speech synthesis
- Voice characteristic modification
"""

import argparse
import os
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, VitsModel, AutoTokenizer
import torch
from scipy.io.wavfile import write as wav_write
from datasets import load_dataset

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Voice anonymization system')
    parser.add_argument('-i', '--input', type=str, required=True,
                       help='Input audio file path')
    parser.add_argument('-o', '--output', type=str, required=True,
                       help='Output audio file path')
    return parser.parse_args()

def validate_paths(input_path: str, output_path: str) -> tuple[bool, str]:
    """Validate input and output file paths."""
    if not os.path.exists(input_path):
        return False, f"Input file {input_path} does not exist"
    
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError:
            return False, f"Cannot create output directory {output_dir}"
    
    return True, ""

def load_audio(file_path: str) -> tuple[torch.Tensor, int]:
    """
    Load an audio file and return the waveform and sample rate.
    """
    try:
        # Resolve absolute path
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Audio file not found: {abs_path}")
            
        # Verify file is readable
        if not os.access(abs_path, os.R_OK):
            raise PermissionError(f"Cannot read audio file: {abs_path}")
            
        # Load audio using absolute path
        waveform, sample_rate = torchaudio.load(abs_path)
        
        if waveform.shape[0] == 0 or sample_rate == 0:
            raise ValueError("Invalid audio file: empty or corrupt")
            
        return waveform, sample_rate
        
    except Exception as e:
        print(f"Error loading audio file {file_path}: {str(e)}")
        raise

def transcribe_audio(waveform: torch.Tensor, 
                    sample_rate: int, 
                    model: Wav2Vec2ForCTC, 
                    processor: Wav2Vec2Processor, 
                    device: torch.device) -> str:
    """
    Transcribe audio waveform to text using Wav2Vec2 model.

    Args:
        waveform: Audio waveform tensor of shape (channels, samples)
        sample_rate: Sampling rate of the audio in Hz
        model: Pre-trained Wav2Vec2 model for transcription
        processor: Wav2Vec2 processor for audio preprocessing
        device: Computation device (CPU/GPU)

    Returns:
        str: Transcribed text from audio

    Example:
        >>> text = transcribe_audio(waveform, 16000, model, processor, device)
        >>> print(f"Transcription: {text}")
    """
    if waveform is None or sample_rate is None:
        return "Error in loading audio file."

    # Resample the audio to 16kHz if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Process the audio
    input_values = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").input_values.to(device)
    
    # Perform inference
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Decode the logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

def get_speech_duration(waveform: torch.Tensor, 
                       sample_rate: int) -> float:
    """
    Calculate duration of speech in seconds.
    
    Args:
        waveform: Audio waveform tensor of shape (channels, samples)
        sample_rate: Sampling rate of the audio in Hz
    
    Returns:
        float: Duration of the speech in seconds

    Example:
        >>> duration = get_speech_duration(waveform, sample_rate)
        >>> print(f"Duration: {duration:.2f}s")
    """
    return waveform.shape[-1] / sample_rate

def normalize_length(waveform: torch.Tensor, 
                    target_length: int, 
                    sample_rate: int) -> torch.Tensor:
    """
    Adjust waveform length without changing pitch using resampling.
    
    Args:
        waveform: Audio waveform tensor to adjust
        target_length: Desired length in samples
        sample_rate: Sampling rate of the audio in Hz
    
    Returns:
        torch.Tensor: Length-normalized waveform

    Example:
        >>> normalized_waveform = normalize_length(waveform, target_length, sample_rate)
        >>> print(f"New length: {normalized_waveform.shape[-1]}")
    """
    current_length = waveform.shape[-1]
    if current_length == target_length:
        return waveform
        
    # Calculate new sampling rate for time stretching
    stretch_factor = target_length / current_length
    new_sample_rate = int(sample_rate * stretch_factor)
    
    # Apply resampling for time stretching
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate,
        new_freq=new_sample_rate
    )
    return resampler(waveform)

def synthesize_speech(text: str, 
                     model: VitsModel, 
                     tokenizer: AutoTokenizer, 
                     output_path: str, 
                     device: torch.device, 
                     target_duration: float = None) -> None:
    """
    Synthesize speech from text with optional duration matching.
    
    Args:
        text: Input text to synthesize
        model: Pre-trained TTS model
        tokenizer: Text tokenizer for the model
        output_path: Path to save the output audio file
        device: Computation device (CPU/GPU)
        target_duration: Optional target duration in seconds

    Example:
        >>> synthesize_speech("Hello world", model, tokenizer, "output.wav", device, 5.0)
    """
    # Prepare input text
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate speech
    with torch.no_grad():
        output = model(**inputs).waveform
        audio = output.squeeze().cpu()
    
    # Match duration if specified
    if target_duration is not None:
        target_samples = int(target_duration * model.config.sampling_rate)
        audio = normalize_length(
            audio, 
            target_samples,
            model.config.sampling_rate
        )
    
    # Save the synthesized speech
    wav_write(
        output_path, 
        rate=model.config.sampling_rate,
        data=(audio * 32767).numpy().astype('int16')
    )

def main():
    """
    Main function to run the voice anonymization system.
    
    Steps:
        1. Load audio file
        2. Extract speaker embeddings
        3. Transcribe audio to text
        4. Synthesize speech with modified voice characteristics
        5. Save the output audio file

    Example:
        >>> main()
    """
    args = parse_arguments()
    
    # Validate paths
    valid, error_msg = validate_paths(args.input, args.output)
    if not valid:
        print(f"Error: {error_msg}")
        return 1
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    stt_model_name = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
    stt_model = Wav2Vec2ForCTC.from_pretrained(stt_model_name).to(device)
    stt_processor = Wav2Vec2Processor.from_pretrained(stt_model_name)
    
    # Load the embeddings dataset
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

    # Load the TTS model and tokenizer
    tts_model_name = "facebook/mms-tts-pol"
    tts_model = VitsModel.from_pretrained(tts_model_name).to(device)
    tts_tokenizer = AutoTokenizer.from_pretrained(tts_model_name)
    
    try:
        print(f"Attempting to load audio from: {args.input}")
        waveform, sample_rate = load_audio(args.input)
        if waveform is None or sample_rate is None:
            raise ValueError("Failed to load audio file")
            
        # Get the original duration of the audio
        original_duration = waveform.shape[-1] / sample_rate
        # Transcribe the audio
        transcription = transcribe_audio(waveform, sample_rate, stt_model, stt_processor, device)
        print(f"Transcription: {transcription}")
        
        # Get input speech duration
        input_duration = get_speech_duration(waveform, sample_rate)
        print(f"Input duration: {input_duration:.2f}s")
        
        # Synthesize with matching duration
        synthesize_speech(
            transcription,
            tts_model, 
            tts_tokenizer,
            args.output,
            device,
            target_duration=input_duration
        )
        print(f"Anonymized speech saved to {args.output}")
        return 0
    except Exception as e:
        print(f"Error in main: {str(e)}")
        return 1

if __name__ == "__main__":
    exit(main())