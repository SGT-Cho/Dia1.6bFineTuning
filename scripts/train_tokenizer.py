#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for training SentencePiece tokenizer on Korean jamo text
- Trains a SentencePiece model on the prepared jamo corpus
- Integrates with Whisper tokenizer for TTS fine-tuning
"""

import os
import argparse
import sentencepiece as spm
from transformers import WhisperTokenizer
import json

def train_sentencepiece(input_file, model_prefix, vocab_size=8000, character_coverage=1.0):
    """Train a SentencePiece tokenizer on jamo text"""
    print(f"Training SentencePiece tokenizer with vocab_size={vocab_size}...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found. Run prepare_data.py with --export flag first.")
    
    # Train the model
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type="unigram",
        normalization_rule_name="identity",  # No normalization for jamo
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
    )
    
    print(f"SentencePiece model saved to {model_prefix}.model and {model_prefix}.vocab")
    return f"{model_prefix}.model"

def adapt_whisper_tokenizer(spm_model_path, output_dir):
    """Adapt Whisper tokenizer with Korean jamo SentencePiece model"""
    print(f"Adapting Whisper tokenizer with SentencePiece model {spm_model_path}...")
    
    try:
        # Whisper 토크나이저 로드
        print("Loading Whisper tokenizer from 'openai/whisper-small'...")
        tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-small")
        
        # SentencePiece 모델을 직접 사용하는 방식으로 수정
        print(f"Copying SentencePiece model to output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # SentencePiece 모델 복사
        import shutil
        target_spm_path = os.path.join(output_dir, os.path.basename(spm_model_path))
        shutil.copy(spm_model_path, target_spm_path)
        
        # vocab.json 파일과 필요한 토크나이저 파일 저장
        tokenizer.save_pretrained(output_dir)
        
        # 토크나이저 설정 저장
        vocab_config = {
            "name": "Korean Jamo Whisper Tokenizer",
            "spm_model_file": os.path.basename(spm_model_path),
            "pad_token": "<pad>",
            "unk_token": "<unk>",
            "bos_token": "<s>",
            "eos_token": "</s>"
        }
        
        with open(os.path.join(output_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(vocab_config, f, indent=2)
        
        print(f"Adapted tokenizer and SentencePiece model saved to {output_dir}")
        
        # SentencePiece를 Jamo 처리에 사용하기 위한 클래스 생성
        from sentencepiece import SentencePieceProcessor
        spm_model = SentencePieceProcessor()
        spm_model.Load(spm_model_path)
        
        # 원래 토크나이저에 SentencePiece 전처리 추가
        orig_encode = tokenizer.encode
        
        def encode_with_sp(text, *args, **kwargs):
            # Jamo 문자열을 SentencePiece로 토크나이징
            sp_tokens = spm_model.EncodeAsPieces(text)
            sp_tokens_str = " ".join(sp_tokens)
            # 기존 토크나이저로 처리
            return orig_encode(sp_tokens_str, *args, **kwargs)
            
        # 커스텀 메서드 추가
        tokenizer.encode_with_sp = encode_with_sp
        
        return tokenizer
    
    except Exception as e:
        print(f"Error adapting tokenizer: {e}")
        raise

def test_tokenizer(tokenizer, test_texts):
    """Test the tokenizer on some sample Korean texts"""
    print("\nTesting tokenizer with sample Korean texts:")
    
    for text in test_texts:
        from jamo import h2j, j2hcj
        jamo_text = j2hcj(h2j(text))
        tokens = tokenizer(jamo_text, return_tensors="pt").input_ids[0]
        decoded = tokenizer.decode(tokens)
        
        print(f"\nOriginal: {text}")
        print(f"Jamo: {jamo_text}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded}")
        print(f"Token count: {len(tokens)}")

def main():
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer for Korean TTS")
    parser.add_argument("--input", type=str, default="data/all_jamo.txt", help="Input jamo corpus file")
    parser.add_argument("--model-prefix", type=str, default="korean_jamo", help="Model prefix for SentencePiece")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Vocabulary size")
    parser.add_argument("--output-dir", type=str, default="tokenizer", help="Output directory for the tokenizer")
    parser.add_argument("--test", action="store_true", help="Test the tokenizer after training")
    args = parser.parse_args()
    
    # Train SentencePiece model
    spm_model_path = train_sentencepiece(
        args.input, 
        args.model_prefix, 
        args.vocab_size
    )
    
    # Adapt Whisper tokenizer
    tokenizer = adapt_whisper_tokenizer(spm_model_path, args.output_dir)
    
    # Test the tokenizer if requested
    if args.test:
        test_texts = [
            "안녕하세요",
            "한국어 음성합성을 위한 모델입니다.",
            "디아 모델을 파인튜닝하고 있어요."
        ]
        test_tokenizer(tokenizer, test_texts)

if __name__ == "__main__":
    main()