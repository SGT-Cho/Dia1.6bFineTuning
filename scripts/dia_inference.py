#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국어 TTS를 위한 파인튜닝된 Dia 모델 추론 스크립트
- 한국어 텍스트에서 음성 생성
- 한국어 자모 처리 지원
"""

import os
import argparse
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from jamo import h2j, j2hcj
from pathlib import Path
import time
import sys

# Dia 모델 임포트를 위한 경로 추가
sys.path.append(os.path.abspath('/workspace/dia/Dia'))

from dia.config import DiaConfig
from dia.model import Dia, ComputeDtype

def hangul_to_jamo(text):
    """한글 텍스트를 분해된 자모 표현으로 변환"""
    return j2hcj(h2j(text))

def load_model(model_path):
    """파인튜닝된 모델 로드"""
    print(f"모델 로드 중: {model_path}")
    
    # 모델 경로 확인
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # 기본 Dia 모델 로드
    base_model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype="float16", load_dac=True)
    
    # 파인튜닝된 가중치 로드
    state_dict = torch.load(model_path, map_location="cpu")
    base_model.model.load_state_dict(state_dict, strict=False)
    
    # GPU로 이동 (가능한 경우)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model.model.to(device)
    base_model.model.eval()
    
    print(f"모델이 {device}에 로드되었습니다.")
    return base_model, device

def generate_speech(model, text, device, max_tokens=3000, cfg_scale=3.0, temperature=1.3, 
                    top_p=0.95, cfg_filter_top_k=30, output_path=None, plot=False, 
                    speed_factor=0.94):
    """
    한국어 텍스트 입력에서 음성 생성
    
    Args:
        model: 파인튜닝된 Dia 모델
        text: 한국어 텍스트 입력
        device: torch 디바이스
        max_tokens: 생성할 최대 토큰 수
        cfg_scale: CFG 스케일 값 (높을수록 텍스트 프롬프트에 더 충실)
        temperature: 샘플링 온도 (높을수록 무작위성 증가)
        top_p: 누적 확률 임계값 (Nucleus sampling)
        cfg_filter_top_k: CFG 필터링 상위 k 값
        output_path: 생성된 오디오를 저장할 경로
        plot: 오디오 파형을 플롯할지 여부
        speed_factor: 오디오 속도 조절 인자 (0.94 = 6% 느리게)
    
    Returns:
        생성된 오디오 파형
    """
    print(f"텍스트 생성 중: {text}")
    
    # 한글을 자모로 변환
    jamo_text = hangul_to_jamo(text)
    print(f"자모 표현: {jamo_text}")
    
    # 화자 태그 추가 (Dia 모델은 [S1], [S2] 태그 사용)
    if not jamo_text.startswith("[S1]") and not jamo_text.startswith("[S2]"):
        jamo_text = f"[S1] {jamo_text}"
        print(f"화자 태그 추가됨: {jamo_text}")
    
    # 시간 측정 시작
    start_time = time.time()
    
    # 음성 생성
    with torch.inference_mode():
        audio_np = model.generate(
            jamo_text,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature, 
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            use_torch_compile=False,
            verbose=True
        )
    
    generation_time = time.time() - start_time
    print(f"생성 완료: {generation_time:.2f}초")
    
    # 오디오가 생성되었는지 확인
    if audio_np is None:
        print("오디오 생성 실패")
        return None
    
    # 속도 조절
    if speed_factor != 1.0:
        original_len = len(audio_np)
        # 속도 인자가 지나치게 작거나 크지 않도록 제한
        speed_factor = max(0.1, min(speed_factor, 5.0))
        target_len = int(original_len / speed_factor)
        
        if target_len != original_len and target_len > 0:
            x_original = np.arange(original_len)
            x_resampled = np.linspace(0, original_len - 1, target_len)
            audio_np = np.interp(x_resampled, x_original, audio_np)
            print(f"오디오 속도 조절: 원본 길이 {original_len}에서 {target_len}로 변경 (속도 인자: {speed_factor:.2f}x)")
    
    # 오디오 정규화
    audio_np = np.clip(audio_np, -1.0, 1.0)
    
    # 플롯 생성 (요청된 경우)
    if plot:
        plt.figure(figsize=(10, 4))
        # 상단: 파형
        plt.subplot(2, 1, 1)
        plt.plot(audio_np)
        plt.title(f"파형: {text}")
        plt.xlabel("샘플")
        plt.ylabel("진폭")
        
        # 하단: 스펙트로그램
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)
        librosa.display.specshow(D, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        
        # 플롯 저장 (출력 경로가 제공된 경우)
        if output_path:
            plot_path = f"{os.path.splitext(output_path)[0]}.png"
            plt.savefig(plot_path)
            print(f"스펙트로그램 저장됨: {plot_path}")
        else:
            plt.show()
    
    # 오디오 저장 (출력 경로가 제공된 경우)
    if output_path:
        sf.write(output_path, audio_np, 44100)  # Dia 모델의 기본 샘플 레이트는 44.1kHz
        print(f"오디오 저장됨: {output_path}")
    
    return audio_np

def main():
    parser = argparse.ArgumentParser(description="파인튜닝된 Dia 모델을 사용한 한국어 음성 생성")
    parser.add_argument("--model-path", type=str, required=True, help="파인튜닝된 모델 파일 경로")
    parser.add_argument("--text", type=str, default="안녕하세요", help="음성으로 합성할 한국어 텍스트")
    parser.add_argument("--output", type=str, default="output.wav", help="출력 오디오 파일 경로")
    parser.add_argument("--max-tokens", type=int, default=3000, help="생성할 최대 토큰 수")
    parser.add_argument("--cfg-scale", type=float, default=3.0, help="CFG 스케일 값")
    parser.add_argument("--temperature", type=float, default=1.3, help="샘플링 온도")
    parser.add_argument("--top-p", type=float, default=0.95, help="Nucleus sampling 임계값")
    parser.add_argument("--top-k", type=int, default=30, help="CFG 필터링 상위 k 값")
    parser.add_argument("--speed", type=float, default=0.94, help="오디오 속도 인자 (1.0 = 원본 속도)")
    parser.add_argument("--plot", action="store_true", help="오디오 파형과 스펙트로그램 플롯")
    args = parser.parse_args()
    
    # 모델 로드
    model, device = load_model(args.model_path)
    
    # 음성 생성
    generate_speech(
        model, 
        args.text, 
        device,
        max_tokens=args.max_tokens,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_p=args.top_p,
        cfg_filter_top_k=args.top_k,
        output_path=args.output,
        plot=args.plot,
        speed_factor=args.speed
    )

if __name__ == "__main__":
    main()