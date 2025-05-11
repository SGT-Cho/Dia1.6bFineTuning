#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference script for Korean TTS using fine-tuned Dia model
- Generates speech from Korean text input
- Converts generated mel spectrograms to waveforms using Griffin-Lim
"""

import os
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
from transformers import WhisperTokenizer, AutoConfig, AutoModelForSpeechSeq2Seq
from jamo import h2j, j2hcj
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add Dia package directory to path
sys.path.append(os.path.abspath('/workspace/dia/Dia'))
# Import Dia
from dia.model import Dia
from dia.config import DiaConfig

def hangul_to_jamo(text):
    """Convert Hangul text to decomposed Jamo representation"""
    return j2hcj(h2j(text))

def griffin_lim(spectrogram, n_iter=50, hop_length=256, win_length=1024, n_fft=1024):
    """
    Convert spectrogram to waveform using Griffin-Lim algorithm
    """
    # Convert from dB to power
    spectrogram = librosa.db_to_power(spectrogram)
    
    # Perform Griffin-Lim
    angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
    complex_spec = spectrogram.astype(np.complex128) * angles
    signal = librosa.istft(complex_spec, hop_length=hop_length, win_length=win_length)
    
    for _ in range(n_iter):
        _, phase = librosa.magphase(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
        complex_spec = spectrogram.astype(np.complex128) * np.exp(1j * np.angle(phase))
        signal = librosa.istft(complex_spec, hop_length=hop_length, win_length=win_length)
    
    return signal

def load_model_and_tokenizer(model_dir, tokenizer_dir=None):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from {model_dir}")
    
    # Default tokenizer directory if not specified
    if tokenizer_dir is None:
        # First try with the default workspace tokenizer directory
        workspace_tokenizer = "/workspace/dia/tokenizer"
        if os.path.exists(workspace_tokenizer):
            tokenizer_dir = workspace_tokenizer
        else:
            # Otherwise check if there's a tokenizer directory next to the model
            model_parent_dir = os.path.dirname(model_dir)
            possible_tokenizer = os.path.join(model_parent_dir, "tokenizer")
            if os.path.exists(possible_tokenizer):
                tokenizer_dir = possible_tokenizer
    
    # Load the tokenizer
    if tokenizer_dir and os.path.exists(tokenizer_dir):
        print(f"Loading tokenizer from {tokenizer_dir}")
        tokenizer = WhisperTokenizer.from_pretrained(
            tokenizer_dir,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>"
        )
    else:
        print(f"Tokenizer not found. Please specify a valid tokenizer directory.")
        raise FileNotFoundError(f"No tokenizer found at {tokenizer_dir}")
    
    # Check if we're loading a Dia native model or a transformers model
    is_pth_file = model_dir.endswith('.pth')
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if is_pth_file:
        print("Loading native Dia model from .pth file")
        # For Dia models, the device handling is built-in to the model initialization
        # Dia automatically uses CUDA if available
        dia = Dia.from_pretrained("nari-labs/Dia-1.6B", load_dac=True)
        model = dia
        # Dia models don't have eval() method
    else:
        # Load model configuration for transformers model
        config = AutoConfig.from_pretrained(model_dir)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, config=config)
        # Only move transformers models to the device and set to eval mode
        model = model.to(device)
        model.eval()
    
    return model, tokenizer, device

def generate_speech(model, tokenizer, text, device, output_path=None, plot=False):
    """
    Generate speech from Korean text input
    
    Args:
        model: Fine-tuned Dia model
        tokenizer: Korean Jamo tokenizer
        text: Korean text input
        device: Torch device
        output_path: Path to save the generated audio
        plot: Whether to plot the mel spectrogram
    
    Returns:
        Generated waveform
    """
    print(f"Generating speech for text: {text}")
    
    # Convert text to jamo
    jamo_text = hangul_to_jamo(text)
    print(f"Jamo representation: {jamo_text}")
    
    # Debug: print tokenized output to check if tokenizer is working properly
    tokenized = tokenizer(jamo_text, return_tensors="pt")
    print(f"Tokenized input length: {len(tokenized.input_ids[0])}")
    print(f"First few tokens: {tokenized.input_ids[0][:10].tolist()}")
    
    # Check what type of model we're using
    if isinstance(model, Dia):
        # Use Dia native inference
        print("Using Dia native inference")
        
        # 1) mel-spectrogram 생성
        mel = model.generate(
            text=[jamo_text],
            max_tokens=4096,        # 충분히 긴 시퀀스 생성
            temperature=0.8,
            top_p=0.95,
            cfg_scale=3.0,
            verbose=True
        )
        
        # 2) mel 형태 확인
        mel_array = np.array(mel)
        print(f">>> generated mel shape: {mel_array.shape}")
        
        # 3) mel → waveform 변환
        if len(mel_array.shape) == 2:  # 2D array인 경우: 실제 멜-스펙트로그램
            print("Converting mel-spectrogram to waveform using Griffin-Lim...")
            waveform = griffin_lim(
                spectrogram=mel_array,
                n_iter=60,
                hop_length=256,
                win_length=1024,
                n_fft=1024
            )
            print(f">>> Generated waveform shape: {waveform.shape}")
        else:  # 이미 waveform인 경우
            print("Output is already a waveform, not converting...")
            if isinstance(mel, np.ndarray):
                waveform = mel
            elif isinstance(mel, list) and len(mel) > 0:
                waveform = mel[0]
            else:
                print(f"Unexpected mel type: {type(mel)}")
                waveform = np.array(mel)
            
            # 긴 오디오에서 실제 발화 부분만 찾기 (처음부터 0.02 이상의 진폭이 있는 마지막 지점까지)
            abs_waveform = np.abs(waveform)
            significant_indices = np.where(abs_waveform > 0.02)[0]
            if len(significant_indices) > 0:
                end_idx = significant_indices[-1] + int(44100 * 0.5)  # 마지막 소리 이후 0.5초 더 유지
                end_idx = min(end_idx, len(waveform))  # 배열 범위 초과 방지
                trimmed_waveform = waveform[:end_idx]
                print(f">>> Trimmed waveform length: {len(trimmed_waveform)} samples ({len(trimmed_waveform)/44100:.2f} seconds)")
                waveform = trimmed_waveform
            else:
                print(">>> No significant audio found, using full output")
            
        sample_rate = 44100  # Dia native output sample rate
    else:
        # Use transformers pipeline
        # Tokenize
        input_ids = tokenizer(jamo_text, return_tensors="pt").input_ids.to(device)
        
        # Generate mel spectrogram with improved parameters
        with torch.no_grad():
            # Check transformers version to use the appropriate generation parameters
            import transformers
            transformers_version = transformers.__version__
            print(f"Transformers version: {transformers_version}")
            
            major, minor = map(int, transformers_version.split('.')[:2])
            if major >= 4 and minor >= 37:
                # For Transformers v4.37+
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=800,  # Generate more tokens
                    return_dict_in_generate=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    decoder_start_token_id=tokenizer.bos_token_id,  # Explicitly set BOS token
                )
            else:
                # For older Transformers versions
                outputs = model.generate(
                    input_ids,
                    max_length=1024,  # Generate longer sequence
                    early_stopping=True,  # Stop at EOS
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    decoder_start_token_id=tokenizer.bos_token_id,  # Explicitly set BOS token
                )
        
        # Debug: print shape
        print(">>> Generated mel shape:", outputs.shape if hasattr(outputs, 'shape') else "Not available (complex output format)")
        
        # Convert model outputs to mel spectrogram
        if hasattr(outputs, 'sequences'):
            # Handle return_dict_in_generate=True format
            mel_spectrogram = outputs.sequences[0].cpu().numpy()
        else:
            # Handle standard output format
            mel_spectrogram = outputs[0].cpu().numpy()
        
        # Apply Griffin-Lim to convert spectrogram to audio (with more iterations)
        waveform = griffin_lim(mel_spectrogram, n_iter=60, hop_length=256, win_length=1024, n_fft=1024)
        sample_rate = 22050
    
    # Plot if requested
    if plot:
        plt.figure(figsize=(10, 4))
        # Create spectrogram from waveform for plotting
        S = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(
            S_dB, 
            x_axis='time', 
            y_axis='mel', 
            sr=sample_rate
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Mel spectrogram for: {text}")
        plt.tight_layout()
        
        # Save plot if output path is provided
        if output_path:
            plot_path = f"{os.path.splitext(output_path)[0]}.png"
            plt.savefig(plot_path)
            print(f"Mel spectrogram saved to {plot_path}")
        else:
            plt.show()
    
    # Save audio if output path is provided
    if output_path:
        sf.write(output_path, waveform, sample_rate)
        print(f"Audio saved to {output_path}")
    
    return waveform

def main():
    parser = argparse.ArgumentParser(description="Generate speech from Korean text using fine-tuned Dia model")
    parser.add_argument("--model-dir", type=str, required=True, help="Path to the fine-tuned model directory or .pth file")
    parser.add_argument("--tokenizer-dir", type=str, default=None, help="Path to the tokenizer directory")
    parser.add_argument("--text", type=str, default="안녕하세요", help="Korean text to synthesize")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")
    parser.add_argument("--plot", action="store_true", help="Plot the mel spectrogram")
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer(args.model_dir, args.tokenizer_dir)
    
    # Generate speech
    generate_speech(model, tokenizer, args.text, device, args.output, args.plot)

if __name__ == "__main__":
    main()