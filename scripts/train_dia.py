#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dia 모델을 한국어 TTS를 위해 파인튜닝하는 스크립트
- Dia의 네이티브 구현을 사용하여 한국어 음성 데이터셋으로 파인튜닝
- 한국어 자모(Jamo) 처리 및 통합
"""

import os
import argparse
import logging
import json
import torch
import numpy as np
import sys
from pathlib import Path
from datasets import load_dataset, concatenate_datasets, Audio
from jamo import h2j, j2hcj
import soundfile as sf
import time
import torchaudio
import librosa

# Dia 모델 임포트를 위한 경로 추가
sys.path.append(os.path.abspath('/workspace/dia/Dia'))

from dia.config import DiaConfig
from dia.model import Dia, ComputeDtype
from dia.layers import DiaModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
# 8-bit 최적화를 위한 bitsandbytes 임포트
from bitsandbytes.optim import AdamW8bit

# 로깅 설정
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
    """한국어 TTS 데이터셋 처리를 위한 클래스"""
    
    def __init__(self, dia_model, data_dirs, metadata_files, sample_rate=22050, mel_n_fft=1024):
        """
        데이터셋 핸들러 초기화
        
        Args:
            dia_model: Dia 모델 인스턴스
            data_dirs: 오디오 파일이 포함된 디렉토리 리스트
            metadata_files: 자모 텍스트가 포함된 메타데이터 파일 리스트
            sample_rate: 대상 오디오 샘플 레이트
            mel_n_fft: 멜 스펙트로그램을 위한 FFT 크기
        """
        self.dia_model = dia_model
        self.config = dia_model.config
        self.data_dirs = data_dirs
        self.metadata_files = metadata_files
        self.sample_rate = sample_rate
        self.mel_n_fft = mel_n_fft
        self.mel_hop_length = mel_n_fft // 4
        self.mel_dim = 80
        self.device = dia_model.device
    
    def load_datasets(self):
        """모든 데이터셋을 로드하고 연결"""
        all_datasets = []
        
        # 각 메타데이터 파일 처리
        for data_dir, metadata_file in zip(self.data_dirs, self.metadata_files):
            if not os.path.exists(metadata_file):
                logger.warning(f"메타데이터 파일 {metadata_file}을 찾을 수 없습니다. 건너뜁니다.")
                continue
            
            # CSV에서 데이터셋 로드 - Audio 기능 사용하지 않고 직접 로드
            logger.info(f"{metadata_file}에서 데이터셋 로드 중")
            import pandas as pd
            df = pd.read_csv(metadata_file, delimiter='|', names=["filename", "text", "jamo"])
            
            # 데이터 디렉토리 정보 추가
            df["data_dir"] = data_dir
            df["full_path"] = df.apply(lambda row: os.path.join(data_dir, "wavs", row["filename"]), axis=1)
            
            # 존재하는 파일만 필터링
            before_count = len(df)
            df = df[df["full_path"].apply(os.path.exists)]
            after_count = len(df)
            
            if before_count > after_count:
                logger.warning(f"{before_count - after_count}개의 오디오 파일을 찾을 수 없습니다.")
            
            logger.info(f"{after_count}개의 유효한 오디오 파일이 있습니다.")
            
            # Dictionary 형태로 데이터셋 생성
            dataset = {
                "filename": df["filename"].tolist(),
                "text": df["text"].tolist(),
                "jamo": df["jamo"].tolist(),
                "data_dir": df["data_dir"].tolist(),
                "full_path": df["full_path"].tolist(),
            }
            all_datasets.append(dataset)
        
        # 모든 데이터셋 연결
        if not all_datasets:
            raise ValueError("유효한 데이터셋을 찾을 수 없습니다. prepare_data.py를 먼저 실행하세요.")
        
        if len(all_datasets) > 1:
            # 여러 데이터셋 연결
            combined_dataset = {}
            for key in all_datasets[0].keys():
                combined_dataset[key] = []
                for dataset in all_datasets:
                    combined_dataset[key].extend(dataset[key])
            return combined_dataset
        else:
            return all_datasets[0]
    
    def _encode_text_to_bytes(self, text):
        """텍스트를 Dia 모델의 바이트 표현으로 변환"""
        return self.dia_model._encode_text(text)
    
    def _encode_audio(self, audio_path):
        """오디오 파일을 DAC 코드북 인덱스로 인코딩"""
        try:
            audio, sr = torchaudio.load(audio_path, channels_first=True)
            if sr != 44100:  # Dia 모델의 기본 샘플 레이트
                audio = torchaudio.functional.resample(audio, sr, 44100)
            
            # Dia 모델의 _encode 메서드를 사용하여 오디오 인코딩
            audio = audio.to(self.device)
            encoded_audio = self.dia_model._encode(audio)
            return encoded_audio
        except Exception as e:
            logger.error(f"오디오 파일 {audio_path} 인코딩 중 오류: {e}")
            return None
    
    def save_processed_dataset(self, processed_dataset, output_path):
        """처리된 데이터셋을 파일로 저장"""
        logger.info(f"처리된 데이터셋 저장 중: {output_path}")
        
        # 처리된 데이터 저장
        save_data = {
            "text_tokens": [t.cpu().tolist() for t in processed_dataset["text_tokens"]],
            "audio_tokens": [a.cpu().tolist() for a in processed_dataset["audio_tokens"]],
            "original_indices": processed_dataset["original_indices"]
        }
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 저장
        with open(output_path, 'wb') as f:
            torch.save(save_data, f)
        
        logger.info(f"데이터셋이 성공적으로 저장됨: {output_path}")
    
    def load_processed_dataset(self, input_path):
        """저장된 전처리 데이터셋 로드"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"처리된 데이터셋 파일을 찾을 수 없음: {input_path}")
        
        logger.info(f"처리된 데이터셋 로드 중: {input_path}")
        
        # 저장된 데이터 로드
        with open(input_path, 'rb') as f:
            load_data = torch.load(f)
        
        # 텐서로 변환
        processed_dataset = {
            "text_tokens": [torch.tensor(t, dtype=torch.long) for t in load_data["text_tokens"]],
            "audio_tokens": [torch.tensor(a, dtype=torch.long) for a in load_data["audio_tokens"]],
            "original_indices": load_data["original_indices"]
        }
        
        # 커스텀 데이터셋 클래스 생성
        from torch.utils.data import Dataset
        
        class DiaDataset(Dataset):
            def __init__(self, processed_data):
                self.text_tokens = processed_data["text_tokens"]
                self.audio_tokens = processed_data["audio_tokens"]
            
            def __len__(self):
                return len(self.text_tokens)
            
            def __getitem__(self, idx):
                return {
                    "text_tokens": self.text_tokens[idx],
                    "audio_tokens": self.audio_tokens[idx]
                }
        
        return DiaDataset(processed_dataset)
    
    def prepare_dataset(self, dataset, batch_size=8, max_samples=None, save_path=None, load_path=None):
        """데이터셋을 학습용으로 준비
        
        Args:
            dataset: 원본 데이터셋
            batch_size: 처리 배치 크기
            max_samples: 처리할 최대 샘플 수 (None이면 전체)
            save_path: 처리된 데이터셋을 저장할 경로 (None이면 저장 안 함)
            load_path: 저장된 처리 데이터셋을 로드할 경로 (None이면 로드 안 함)
        """
        # 이미 전처리된 데이터셋이 있으면 로드
        if load_path and os.path.exists(load_path):
            return self.load_processed_dataset(load_path)
            
        logger.info("데이터셋 전처리 중...")
        
        # 데이터셋을 일괄 처리하고 오디오 및 텍스트 토큰화
        text_tokens = []
        audio_tokens = []
        valid_indices = []
        
        # 배치 단위로 처리하여 메모리 효율 증가
        total_samples = len(dataset["full_path"])
        if max_samples is not None:
            total_samples = min(total_samples, max_samples)
            
        batch_size = min(batch_size, 32)  # 한 번에 처리할 최대 샘플 수
        
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_indices = list(range(i, batch_end))
            
            batch_text_tokens = []
            batch_audio_tokens = []
            batch_valid_indices = []
            
            for j in batch_indices:
                # 자모 텍스트 인코딩
                jamo_text = dataset["jamo"][j]
                try:
                    encoded_text = self._encode_text_to_bytes(jamo_text)
                    
                    # 오디오 파일 로드 및 인코딩
                    audio_path = dataset["full_path"][j]
                    encoded_audio = self._encode_audio(audio_path)
                    
                    if encoded_audio is not None:
                        batch_text_tokens.append(encoded_text)
                        batch_audio_tokens.append(encoded_audio)
                        batch_valid_indices.append(j)
                    else:
                        logger.warning(f"오디오 파일을 인코딩할 수 없음: {audio_path}")
                except Exception as e:
                    logger.error(f"샘플 처리 오류 (인덱스 {j}): {e}")
            
            text_tokens.extend(batch_text_tokens)
            audio_tokens.extend(batch_audio_tokens)
            valid_indices.extend(batch_valid_indices)
            
            # 진행 상황 로깅
            logger.info(f"처리 중: {batch_end}/{total_samples} 샘플 ({len(valid_indices)}/{batch_end}개 유효)")
        
        # 유효한 샘플로 구성된 최종 데이터셋 생성
        processed_dataset = {
            "text_tokens": text_tokens,
            "audio_tokens": audio_tokens,
            "original_indices": valid_indices,
        }
        
        # 처리된 데이터셋 저장
        if save_path:
            self.save_processed_dataset(processed_dataset, save_path)
        
        # 커스텀 데이터셋 클래스 생성
        from torch.utils.data import Dataset
        
        class DiaDataset(Dataset):
            def __init__(self, processed_data):
                self.text_tokens = processed_data["text_tokens"]
                self.audio_tokens = processed_data["audio_tokens"]
            
            def __len__(self):
                return len(self.text_tokens)
            
            def __getitem__(self, idx):
                return {
                    "text_tokens": self.text_tokens[idx],
                    "audio_tokens": self.audio_tokens[idx]
                }
        
        return DiaDataset(processed_dataset)

class DiaTrainer:
    """Dia 모델 파인튜닝을 위한 훈련 클래스"""
    
    def __init__(
        self,
        dia_model,
        dataset,
        output_dir="checkpoints",
        learning_rate=1e-4,
        batch_size=8,
        gradient_accumulation_steps=4,
        epochs=50,
        fp16=False,
    ):
        """
        트레이너 초기화
        
        Args:
            dia_model: 파인튜닝할 Dia 모델 인스턴스
            dataset: 전처리된 데이터셋 
            output_dir: 모델 체크포인트를 저장할 디렉토리
            learning_rate: 학습률
            batch_size: 배치 크기
            gradient_accumulation_steps: 경사 누적 단계
            epochs: 학습 에폭 수 
            fp16: 혼합 정밀도 학습 활성화 여부
        """
        self.dia_model = dia_model
        self.device = dia_model.device
        self.dataset = dataset
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.fp16 = fp16
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 옵티마이저 설정 - 인코더와 디코더 모두 학습
        params = list(dia_model.model.encoder.parameters()) + list(dia_model.model.decoder.parameters())
        
        # 8-bit 옵티마이저를 사용하여 메모리 사용량 절감
        self.optimizer = AdamW8bit(params, lr=learning_rate)
        
        # 스케줄러 및 스캐폴더 설정 - 최신 API 사용
        self.scaler = torch.amp.GradScaler('cuda') if fp16 else None
        
        # 데이터로더 설정
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """데이터로더를 위한 콜레이트 함수"""
        text_tokens = [item["text_tokens"] for item in batch]
        audio_tokens = [item["audio_tokens"] for item in batch]
        
        # 패딩 및 배치 준비
        max_text_len = min(max(len(t) for t in text_tokens), 
                          self.dia_model.config.data.text_length)
        max_audio_len = min(max(len(a) for a in audio_tokens),
                          self.dia_model.config.data.audio_length)
        
        # 텍스트 토큰 패딩
        text_pad_value = self.dia_model.config.data.text_pad_value
        padded_text = torch.full(
            (len(batch), max_text_len), 
            fill_value=text_pad_value,
            dtype=torch.long
        )
        
        # 오디오 토큰 패딩
        audio_pad_value = self.dia_model.config.data.audio_pad_value
        audio_channels = self.dia_model.config.data.channels
        padded_audio = torch.full(
            (len(batch), max_audio_len, audio_channels),
            fill_value=audio_pad_value,
            dtype=torch.long
        )
        
        # 배치 채우기 (잘라내기 필요한 경우 포함)
        for i in range(len(batch)):
            # 텍스트 토큰 처리 - 길이 제한
            text_len = min(len(text_tokens[i]), max_text_len)
            padded_text[i, :text_len] = text_tokens[i][:text_len]
            
            # 오디오 토큰 처리 - 길이 제한
            audio_len = min(len(audio_tokens[i]), max_audio_len)
            padded_audio[i, :audio_len, :] = audio_tokens[i][:audio_len]
        
        return {
            "text_tokens": padded_text.to(self.device),
            "audio_tokens": padded_audio.to(self.device)
        }
    
    def _training_step(self, batch):
        """단일 학습 단계 수행"""
        text_tokens = batch["text_tokens"]  # (B, T)
        audio_tokens = batch["audio_tokens"]  # (B, T', C)
        
        # 먼저 Dia 모델을 학습 모드로 전환
        self.dia_model.model.train()
        
        # 혼합 정밀도 연산 설정 (최신 API 사용)
        autocast_context = torch.amp.autocast(device_type='cuda', dtype=torch.float16) if self.fp16 else torch.no_grad()
        
        with autocast_context:
            batch_size = text_tokens.size(0)
            
            # 인코더 상태 생성 대신 커스텀 위치 인코딩 및 마스크 생성
            from dia.state import EncoderInferenceState, create_attn_mask
            
            # 모델이 기대하는 길이로 위치 텐서 생성 (1024로 고정)
            model_text_len = self.dia_model.config.data.text_length
            real_text_len = text_tokens.size(1)
            
            # 일관된 길이의 위치 텐서 생성
            positions = torch.arange(real_text_len, dtype=torch.float32, device=self.device).unsqueeze(0)
            positions = positions.expand(batch_size * 2, -1)  # (B*2, T)
            
            # 패딩 마스크 생성 (실제 텍스트와 패딩 구분)
            padding_mask = (text_tokens != self.dia_model.config.data.text_pad_value).to(self.device)
            padding_mask = padding_mask.repeat_interleave(2, dim=0)  # (B*2, T)
            
            # 어텐션 마스크 생성
            attn_mask = create_attn_mask(padding_mask, padding_mask, self.device, is_causal=False)
            
            # 수동 엔코더 상태 생성
            enc_state = EncoderInferenceState(
                max_seq_len=real_text_len,
                device=self.device,
                positions=positions,
                padding_mask=padding_mask,
                attn_mask=attn_mask
            )
            
            # 인코더 출력 계산
            encoder_outputs = self.dia_model.model.encoder(text_tokens.repeat_interleave(2, dim=0), enc_state)
            
            # 디코더 상태 초기화 및 크로스 어텐션 캐시 계산
            from dia.state import DecoderInferenceState
            dec_cross_attn_cache = self.dia_model.model.decoder.precompute_cross_attn_cache(
                encoder_outputs, enc_state.positions, enc_state.padding_mask)
            
            dec_state = DecoderInferenceState.new(
                self.dia_model.config, enc_state, encoder_outputs, dec_cross_attn_cache,
                self.dia_model.model.encoder.compute_dtype
            )
            
            # 디코더에 오디오 토큰 입력하여 로짓 계산 (자기회귀 방식 아닌 병렬 교사 강제)
            logits = self.dia_model.model.decoder(audio_tokens.repeat_interleave(2, dim=0), dec_state)  # (B*2, T', C, V)
            logits = logits[::2]  # 반복된 배치 중 첫 번째 그룹만 사용 (B, T', C, V)
            
            # 손실 계산: 다음 토큰 예측
            shift_logits = logits[:, :-1].contiguous()  # (B, T'-1, C, V)
            shift_labels = audio_tokens[:, 1:].contiguous()  # (B, T'-1, C)
            
            # 채널별 손실 계산 및 평균화
            loss = 0
            audio_channels = self.dia_model.config.data.channels
            
            for c in range(audio_channels):
                channel_logits = shift_logits[:, :, c]  # (B, T'-1, V)
                channel_labels = shift_labels[:, :, c]  # (B, T'-1)
                channel_loss = torch.nn.functional.cross_entropy(
                    channel_logits.reshape(-1, channel_logits.size(-1)),
                    channel_labels.reshape(-1),
                    ignore_index=self.dia_model.config.data.audio_pad_value
                )
                loss += channel_loss
            
            loss = loss / audio_channels
        
        # 손실 스케일링 및 역전파
        if self.fp16:
            self.scaler.scale(loss / self.gradient_accumulation_steps).backward()
        else:
            (loss / self.gradient_accumulation_steps).backward()
        
        return loss.item()
    
    def train(self):
        """전체 학습 과정 수행"""
        self.dia_model.model.train()
        
        total_steps = len(self.dataloader) * self.epochs
        global_step = 0
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            for step, batch in enumerate(self.dataloader):
                step_loss = self._training_step(batch)
                epoch_loss += step_loss
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # 옵티마이저 스텝 수행
                    if self.fp16:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad()
                    global_step += 1
                
                # 로깅
                if (step + 1) % 10 == 0:
                    logger.info(
                        f"에폭: {epoch+1}/{self.epochs}, "
                        f"스텝: {step+1}/{len(self.dataloader)}, "
                        f"손실: {step_loss:.4f}, "
                        f"전체 진행: {global_step}/{total_steps}"
                    )
            
            # 에폭 종료 시 로깅
            epoch_time = time.time() - epoch_start_time
            avg_loss = epoch_loss / len(self.dataloader)
            
            logger.info(
                f"에폭 {epoch+1} 완료: 평균 손실={avg_loss:.4f}, "
                f"소요 시간={epoch_time:.2f}초"
            )
            
            # 중간 체크포인트는 저장하지 않음
        
        # 최종 모델만 저장 (안전한 저장 방식 사용)
        final_path = os.path.join(self.output_dir, "final-model.pth")
        
        try:
            # CPU로 이동하여 메모리 효율적으로 저장
            model_to_save = {}
            logger.info("최종 모델 저장 준비 중...")
            
            # 파라미터를 CPU로 이동하여 메모리 효율적으로 저장
            for name, param in self.dia_model.model.state_dict().items():
                model_to_save[name] = param.detach().cpu()
            
            logger.info(f"최종 모델 저장 중: {final_path}")
            torch.save(model_to_save, final_path)
            logger.info(f"최종 모델 저장 완료: {final_path}")
        except Exception as e:
            logger.error(f"최종 모델 저장 실패: {e}")
        
        return final_path

def main():
    parser = argparse.ArgumentParser(description="한국어 TTS를 위한 Dia 모델 파인튜닝")
    
    # 모델 및 토크나이저 인수
    parser.add_argument("--config-path", type=str, default=None, 
                        help="Dia 모델 설정 파일 경로")
    parser.add_argument("--checkpoint-path", type=str, default=None, 
                        help="Dia 모델 체크포인트 경로")
    parser.add_argument("--compute-dtype", type=str, default="float32", 
                        choices=["float32", "float16", "bfloat16"],
                        help="계산 데이터 타입")
    
    # 데이터셋 인수
    parser.add_argument("--kss", action="store_true", help="KSS 데이터셋 사용")
    parser.add_argument("--zeroth", action="store_true", help="Zeroth 데이터셋 사용")
    parser.add_argument("--all-datasets", action="store_true", help="사용 가능한 모든 데이터셋 사용")
    parser.add_argument("--max-samples", type=int, default=None, 
                        help="처리할 최대 샘플 수 (None이면 전체 데이터셋 사용)")
    
    # 훈련 인수
    parser.add_argument("--batch-size", type=int, default=8, help="훈련 배치 크기")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, 
                        help="경사 누적 단계")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="학습률")
    parser.add_argument("--epochs", type=int, default=50, help="학습 에폭 수")
    parser.add_argument("--output-dir", type=str, default="checkpoints", 
                        help="모델 체크포인트를 저장할 디렉토리")
    parser.add_argument("--fp16", action="store_true", help="혼합 정밀도 학습 활성화")
    parser.add_argument("--save-path", type=str, default=None, help="처리된 데이터셋 저장 경로")
    parser.add_argument("--load-path", type=str, default=None, help="저장된 처리 데이터셋 로드 경로")
    parser.add_argument("--preprocess-only", action="store_true", 
                        help="데이터 전처리만 수행하고 학습은 진행하지 않음")
    
    args = parser.parse_args()
    
    # 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Dia 모델 로드
    try:
        if args.config_path and args.checkpoint_path:
            logger.info(f"로컬 파일에서 Dia 모델 로드 중: {args.checkpoint_path}")
            dia = Dia.from_local(
                config_path=args.config_path,
                checkpoint_path=args.checkpoint_path,
                compute_dtype=args.compute_dtype,
                load_dac=True  # 오디오 인코딩/디코딩 필요
            )
        else:
            logger.info("Dia 모델을 Hugging Face에서 로드 중")
            dia = Dia.from_pretrained(
                model_name="nari-labs/Dia-1.6B",
                compute_dtype=args.compute_dtype,
                load_dac=True  # 오디오 인코딩/디코딩 필요
            )
        
        logger.info("Dia 모델 로드 성공")
        
        # Dia 모델에 직접 커스텀 Gradient Checkpointing 구현
        logger.info("커스텀 Gradient Checkpointing 활성화")
        
        # Encoder Layer에 checkpointing 적용 (인자가 단순함)
        for layer in dia.model.encoder.layers:
            # 원래 forward 함수 저장
            original_forward = layer.forward
            
            # checkpoint wrapper 함수 정의
            def get_checkpointed_forward(original_fn):
                def checkpointed_forward(*args, **kwargs):
                    # encoder는 인자가 단순하여 그대로 전달
                    return torch.utils.checkpoint.checkpoint(
                        original_fn, 
                        *args, 
                        use_reentrant=False  # 명시적으로 use_reentrant=False 설정
                    )
                return checkpointed_forward
            
            # forward 메소드를 checkpoint 버전으로 교체
            layer.forward = get_checkpointed_forward(original_forward)
        
        # Decoder Layer에 checkpointing 적용 (키워드 인자가 있음)
        for layer in dia.model.decoder.layers:
            # 원래 forward 함수 저장
            original_forward = layer.forward
            
            # checkpoint wrapper 함수 정의 (키워드 인자 문제 해결)
            def get_checkpointed_forward(original_fn):
                def checkpointed_forward(x, state, **kwargs):
                    # 키워드 인자는 잠시 무시하고 필수 인자만 checkpoint에 전달
                    # checkpoint 내부에서 키워드 인자를 다시 처리하는 함수 생성
                    def custom_forward(x_inner, state_inner):
                        return original_fn(x_inner, state_inner, **kwargs)
                    
                    return torch.utils.checkpoint.checkpoint(
                        custom_forward, 
                        x, 
                        state,
                        use_reentrant=False  # 명시적으로 use_reentrant=False 설정
                    )
                return checkpointed_forward
            
            # forward 메소드를 checkpoint 버전으로 교체
            layer.forward = get_checkpointed_forward(original_forward)
        
    except Exception as e:
        logger.error(f"Dia 모델 로드 실패: {e}")
        raise
    
    # 데이터 디렉토리 및 메타데이터 파일 준비
    data_dirs = []
    metadata_files = []
    
    if args.all_datasets or args.kss:
        data_dirs.append("data/kss")
        metadata_files.append("data/kss/metadata_jamo.csv")
    
    if args.all_datasets or args.zeroth:
        data_dirs.append("data/zeroth")
        metadata_files.append("data/zeroth/metadata_jamo.csv")
    
    if not data_dirs:
        parser.error("최소한 하나의 데이터셋을 지정해야 합니다 (--kss, --zeroth, 또는 --all-datasets)")
    
    # 데이터셋 핸들러 생성 및 데이터셋 로드
    dataset_handler = KoreanTTSDataset(dia, data_dirs, metadata_files)
    raw_dataset = dataset_handler.load_datasets()
    
    # 저장 경로가 지정되지 않았지만 전처리만 하는 경우 기본 경로 사용
    if args.preprocess_only and args.save_path is None:
        args.save_path = os.path.join(args.output_dir, "processed_dataset.pt")
    
    processed_dataset = dataset_handler.prepare_dataset(
        raw_dataset, 
        args.batch_size,
        max_samples=args.max_samples, 
        save_path=args.save_path, 
        load_path=args.load_path
    )
    
    logger.info(f"학습용 {len(processed_dataset)} 샘플 로드됨")
    
    # 전처리만 수행하고 종료
    if args.preprocess_only:
        logger.info("전처리 완료. 학습은 수행하지 않음.")
        return
    
    # 트레이너 초기화
    trainer = DiaTrainer(
        dia_model=dia,
        dataset=processed_dataset,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        fp16=args.fp16,
    )
    
    # 학습 시작
    logger.info("학습 시작...")
    final_model_path = trainer.train()
    
    logger.info(f"학습 완료. 최종 모델 저장됨: {final_model_path}")

if __name__ == "__main__":
    main()