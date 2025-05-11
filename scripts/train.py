#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main training script for fine-tuning Dia-1.6B for Korean TTS
- Loads Korean datasets and preprocesses them for training
- Sets up Dia model with Korean tokenizer
- Trains the model with the appropriate parameters
"""

import os
import argparse
import logging
import json
import torch
import librosa
import numpy as np
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Audio
from transformers import (
    WhisperTokenizer,
    AutoConfig, 
    AutoModelForSpeechSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from jamo import h2j, j2hcj
import soundfile as sf

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/training.log")
    ]
)
logger = logging.getLogger(__name__)

class KoreanTTSDataset:
    """Class to handle Korean TTS dataset loading and preprocessing"""
    
    def __init__(self, tokenizer, data_dirs, metadata_files, sample_rate=22050, mel_n_fft=1024):
        """
        Initialize the dataset handler
        
        Args:
            tokenizer: Tokenizer for text processing
            data_dirs: List of directories containing wav files
            metadata_files: List of metadata files with jamo text
            sample_rate: Target audio sample rate
            mel_n_fft: Size of FFT for mel spectrogram
        """
        self.tokenizer = tokenizer
        self.data_dirs = data_dirs
        self.metadata_files = metadata_files
        self.sample_rate = sample_rate
        self.mel_n_fft = mel_n_fft
        self.mel_hop_length = mel_n_fft // 4  # Standard hop length
        self.mel_dim = 80  # Standard mel dimensions
    
    def load_datasets(self):
        """Load all datasets and concatenate them"""
        all_datasets = []
        
        # Process each metadata file
        for data_dir, metadata_file in zip(self.data_dirs, self.metadata_files):
            if not os.path.exists(metadata_file):
                logger.warning(f"Metadata file {metadata_file} not found. Skipping.")
                continue
            
            # Load the dataset from CSV
            logger.info(f"Loading dataset from {metadata_file}")
            dataset = load_dataset(
                "csv", 
                data_files=metadata_file,
                delimiter="|",
                column_names=["filename", "text", "jamo"]
            )["train"]
            
            # Set the audio column
            dataset = dataset.cast_column(
                "filename", 
                Audio(sampling_rate=self.sample_rate)
            )
            
            # Add the data directory as a column for path resolution
            dataset = dataset.add_column("data_dir", [data_dir] * len(dataset))
            
            all_datasets.append(dataset)
        
        # Concatenate all datasets
        if not all_datasets:
            raise ValueError("No valid datasets found. Please run prepare_data.py first.")
        
        return concatenate_datasets(all_datasets) if len(all_datasets) > 1 else all_datasets[0]
    
    def preprocess_function(self, example):
        """
        Preprocess a single example for training
        
        This function:
        - Loads the audio file
        - Computes the mel spectrogram
        - Tokenizes the jamo text
        """
        # Full path to the audio file
        audio_path = os.path.join(example["data_dir"], "wavs", example["filename"])
        
        # Load the audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_fft=self.mel_n_fft,
            hop_length=self.mel_hop_length,
            n_mels=self.mel_dim
        )
        
        # Convert to log scale and normalize
        log_mel = librosa.power_to_db(mel, ref=np.max)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-5)
        
        # Tokenize the jamo text
        input_ids = self.tokenizer(
            example["jamo"],
            return_tensors="pt"
        ).input_ids[0]
        
        return {
            "input_ids": input_ids,
            "mel_spectrogram": torch.tensor(log_mel, dtype=torch.float),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long)
        }

    def prepare_dataset(self, dataset, batch_size=8):
        """Prepare the dataset for training"""
        # Apply preprocessing
        logger.info("Preprocessing dataset...")
        processed_dataset = dataset.map(
            self.preprocess_function,
            remove_columns=["filename", "text", "jamo", "data_dir", "audio"],
            batched=False,
            num_proc=4
        )
        
        # Set the format for PyTorch
        processed_dataset = processed_dataset.with_format("torch")
        
        return processed_dataset

def load_model_and_tokenizer(args):
    """Load the Dia model and tokenizer"""
    logger.info(f"Loading tokenizer from {args.tokenizer_dir}")
    
    # Load the tokenizer
    if os.path.exists(args.tokenizer_dir):
        tokenizer = WhisperTokenizer.from_pretrained(
            args.tokenizer_dir,
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>"
        )
    else:
        logger.warning(f"Tokenizer directory {args.tokenizer_dir} not found. Using default tokenizer.")
        tokenizer = WhisperTokenizer.from_pretrained(
            "nari-labs/Dia-1.6B",
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>"
        )
    
    # Load model configuration
    logger.info(f"Loading model configuration from {args.model_name}")
    config = AutoConfig.from_pretrained(args.model_name)
    
    # Adjust the config for Korean TTS if needed
    # Example: changing mel spectrogram dimensions and sampling rate
    config.mel_dim = 80
    config.sampling_rate = 22050
    config.mel_hop_length = 256  # hop length for mel spectrogram
    
    # Load the model
    logger.info(f"Loading model from {args.model_name}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.model_name,
        config=config
    )
    
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Dia-1.6B for Korean TTS")
    
    # Model and tokenizer arguments
    parser.add_argument("--model-name", type=str, default="nari-labs/Dia-1.6B", 
                        help="Model name or path")
    parser.add_argument("--tokenizer-dir", type=str, default="tokenizer", 
                        help="Directory containing the tokenizer files")
    
    # Dataset arguments
    parser.add_argument("--kss", action="store_true", help="Use KSS dataset")
    parser.add_argument("--zeroth", action="store_true", help="Use Zeroth dataset")
    parser.add_argument("--all-datasets", action="store_true", help="Use all available datasets")
    
    # Training arguments
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, 
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--output-dir", type=str, default="checkpoints", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--resume", type=str, default=None, 
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--fp16", action="store_true", help="Enable mixed precision training")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Prepare data directories and metadata files
    data_dirs = []
    metadata_files = []
    
    if args.all_datasets or args.kss:
        data_dirs.append("data/kss")
        metadata_files.append("data/kss/metadata_jamo.csv")
    
    if args.all_datasets or args.zeroth:
        data_dirs.append("data/zeroth")
        metadata_files.append("data/zeroth/metadata_jamo.csv")
    
    if not data_dirs:
        parser.error("At least one dataset must be specified (--kss, --zeroth, or --all-datasets)")
    
    # Create dataset handler and load datasets
    dataset_handler = KoreanTTSDataset(tokenizer, data_dirs, metadata_files)
    raw_dataset = dataset_handler.load_datasets()
    processed_dataset = dataset_handler.prepare_dataset(raw_dataset, args.batch_size)
    
    logger.info(f"Loaded {len(processed_dataset)} samples for training")
    
    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        logging_dir="logs",
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=5,
        fp16=args.fp16,
        report_to="tensorboard",
        remove_unused_columns=False,  # Important for custom datasets
    )
    
    # Initialize the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )
    
    # Start training
    logger.info("Starting training...")
    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()
    
    # Save the final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    logger.info(f"Model saved to {os.path.join(args.output_dir, 'final')}")

if __name__ == "__main__":
    main()