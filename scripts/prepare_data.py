#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data preparation script for Korean TTS fine-tuning with Dia-1.6B
- Downloads and processes KSS and Zeroth-Korean datasets
- Converts Korean text to Jamo representation
"""

import os
import pandas as pd
import argparse
from jamo import h2j, j2hcj
from datasets import load_dataset
import librosa
import soundfile as sf
import shutil
from tqdm import tqdm

def hangul_to_jamo(text):
    """Convert Hangul text to decomposed Jamo representation"""
    return j2hcj(h2j(text))

def download_kss():
    """Download KSS dataset (you'll need to implement actual download logic)"""
    print("Note: KSS dataset requires manual download from https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset")
    print("Please download and extract it to data/kss/wavs/ manually")
    
    # Create metadata.csv placeholder if it doesn't exist
    if not os.path.exists("data/kss/metadata.csv"):
        with open("data/kss/metadata.csv", "w", encoding="utf-8") as f:
            f.write("filename|text\n")
        print("Created empty metadata.csv. Please fill it with KSS data.")

def download_zeroth():
    """Download Zeroth-Korean dataset from Hugging Face"""
    print("Downloading Zeroth-Korean dataset...")
    dataset = load_dataset("Bingsu/zeroth-korean")
    
    # Create directory if it doesn't exist
    os.makedirs("data/zeroth/wavs", exist_ok=True)
    
    # Create metadata file
    with open("data/zeroth/metadata.csv", "w", encoding="utf-8") as f:
        f.write("filename|text\n")
        
        # Process each sample
        for idx, sample in enumerate(tqdm(dataset["train"])):
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            filename = f"zeroth_{idx:06d}.wav"
            
            # Resample to 22050 Hz if needed
            if sr != 22050:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=22050)
                sr = 22050
                
            # Save audio file
            sf.write(f"data/zeroth/wavs/{filename}", audio, sr)
            
            # Add to metadata
            f.write(f"{filename}|{sample['text']}\n")
    
    print(f"Processed {len(dataset['train'])} Zeroth-Korean samples")

def prepare_jamo_metadata():
    """Convert text in metadata files to Jamo representation"""
    # Process KSS
    if os.path.exists("data/kss/metadata.csv"):
        try:
            df = pd.read_csv("data/kss/metadata.csv", sep="|", names=["filename", "text"])
            df["jamo"] = df.text.apply(hangul_to_jamo)
            df.to_csv("data/kss/metadata_jamo.csv", sep="|", index=False)
            print(f"Processed {len(df)} KSS samples")
        except Exception as e:
            print(f"Error processing KSS metadata: {e}")
    
    # Process Zeroth
    if os.path.exists("data/zeroth/metadata.csv"):
        try:
            df = pd.read_csv("data/zeroth/metadata.csv", sep="|", names=["filename", "text"])
            df["jamo"] = df.text.apply(hangul_to_jamo)
            df.to_csv("data/zeroth/metadata_jamo.csv", sep="|", index=False)
            print(f"Processed {len(df)} Zeroth samples")
        except Exception as e:
            print(f"Error processing Zeroth metadata: {e}")

def export_jamo_corpus():
    """Export all Jamo text for SentencePiece training"""
    jamo_texts = []
    
    # Collect from KSS
    if os.path.exists("data/kss/metadata_jamo.csv"):
        df = pd.read_csv("data/kss/metadata_jamo.csv", sep="|")
        jamo_texts.extend(df["jamo"].tolist())
    
    # Collect from Zeroth
    if os.path.exists("data/zeroth/metadata_jamo.csv"):
        df = pd.read_csv("data/zeroth/metadata_jamo.csv", sep="|")
        jamo_texts.extend(df["jamo"].tolist())
    
    # Write to file
    with open("data/all_jamo.txt", "w", encoding="utf-8") as f:
        for text in jamo_texts:
            f.write(f"{text}\n")
    
    print(f"Exported {len(jamo_texts)} Jamo texts for SentencePiece training")

def main():
    parser = argparse.ArgumentParser(description="Prepare Korean TTS datasets")
    parser.add_argument("--kss", action="store_true", help="Download KSS dataset")
    parser.add_argument("--zeroth", action="store_true", help="Download Zeroth dataset")
    parser.add_argument("--all", action="store_true", help="Prepare all datasets")
    parser.add_argument("--jamo", action="store_true", help="Convert metadata to Jamo")
    parser.add_argument("--export", action="store_true", help="Export Jamo corpus")
    args = parser.parse_args()
    
    if args.all or args.kss:
        download_kss()
    
    if args.all or args.zeroth:
        download_zeroth()
    
    if args.all or args.jamo:
        prepare_jamo_metadata()
    
    if args.all or args.export:
        export_jamo_corpus()

if __name__ == "__main__":
    main()