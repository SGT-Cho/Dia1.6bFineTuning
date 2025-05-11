# Dia 1.6B Korean TTS Fine-Tuning

This repository contains code for fine-tuning the Dia 1.6B model for Korean text-to-speech (TTS) synthesis. The project demonstrates how to adapt a large language model for high-quality Korean speech synthesis.

## English Documentation

### Overview

This project fine-tunes Nari Labs' Dia 1.6B model for Korean TTS. It includes scripts for data preprocessing, model training, and inference.

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Dia library
- jamo
- librosa
- soundfile
- matplotlib
- numpy

### Installation

1. Clone this repository:
```bash
git clone https://github.com/SGT-Cho/Dia1.6bFineTuning.git
cd Dia1.6bFineTuning
```

2. Create a virtual environment and install the required packages:
```bash
python -m venv korean-tts
source korean-tts/bin/activate  # On Windows: korean-tts\Scripts\activate
pip install -r requirements.txt
```

3. Install the Dia library:
```bash
git clone https://github.com/nari-ai/Dia.git
cd Dia
pip install -e .
cd ..
```

### Dataset Preparation

The model was fine-tuned on a Korean speech dataset. To prepare your own dataset:

1. Organize your audio files in WAV format
2. Create transcription files in txt format
3. Process the dataset using the preprocessing script:
```bash
python scripts/preprocess_dataset.py --audio-dir /path/to/audio --text-dir /path/to/transcripts --output-dir data/processed
```

### Training

To fine-tune the Dia model on your dataset:

```bash
python scripts/train.py --model-name "nari-labs/Dia-1.6B" --data-path data/processed --output-dir checkpoints --epochs 10 --batch-size 8
```

### Inference

To generate speech from text using the fine-tuned model:

```bash
python scripts/inference.py --model-dir checkpoints/final-model.pth --text "안녕하세요" --output output.wav --plot
```

Parameters:
- `--model-dir`: Path to the fine-tuned model
- `--tokenizer-dir`: (Optional) Path to the tokenizer directory
- `--text`: Korean text to synthesize
- `--output`: Output audio file path
- `--plot`: Flag to plot the mel spectrogram

### Workflow

1. **Data Preparation**: Collect and organize Korean audio and text data
2. **Preprocessing**: Convert audio to compatible format and tokenize text
3. **Fine-tuning**: Train the Dia model on the Korean dataset
4. **Inference**: Generate speech from Korean text using the fine-tuned model

### Technical Details

The system works by:
1. Converting Hangul text to Jamo representation
2. Tokenizing the Jamo text
3. Generating mel spectrograms using the fine-tuned Dia model
4. Converting mel spectrograms to waveform using Griffin-Lim algorithm
5. Post-processing audio to trim silence

## 한국어 문서

### 개요

이 프로젝트는 나리 랩스의 Dia 1.6B 모델을 한국어 음성 합성(TTS)을 위해 미세 조정합니다. 데이터 전처리, 모델 훈련 및 추론을 위한 스크립트가 포함되어 있습니다.

### 요구사항

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Dia 라이브러리
- jamo
- librosa
- soundfile
- matplotlib
- numpy

### 설치

1. 저장소 복제:
```bash
git clone https://github.com/SGT-Cho/Dia1.6bFineTuning.git
cd Dia1.6bFineTuning
```

2. 가상 환경 생성 및 필요 패키지 설치:
```bash
python -m venv korean-tts
source korean-tts/bin/activate  # Windows의 경우: korean-tts\Scripts\activate
pip install -r requirements.txt
```

3. Dia 라이브러리 설치:
```bash
git clone https://github.com/nari-ai/Dia.git
cd Dia
pip install -e .
cd ..
```

### 데이터셋 준비

모델은 한국어 음성 데이터셋으로 미세 조정되었습니다. 자체 데이터셋을 준비하려면:

1. WAV 형식의 오디오 파일 준비
2. txt 형식의 전사(transcription) 파일 생성
3. 전처리 스크립트를 사용하여 데이터셋 처리:
```bash
python scripts/preprocess_dataset.py --audio-dir /path/to/audio --text-dir /path/to/transcripts --output-dir data/processed
```

### 훈련

데이터셋으로 Dia 모델을 미세 조정하려면:

```bash
python scripts/train.py --model-name "nari-labs/Dia-1.6B" --data-path data/processed --output-dir checkpoints --epochs 10 --batch-size 8
```

### 추론

미세 조정된 모델을 사용하여 텍스트에서 음성을 생성하려면:

```bash
python scripts/inference.py --model-dir checkpoints/final-model.pth --text "안녕하세요" --output output.wav --plot
```

매개변수:
- `--model-dir`: 미세 조정된 모델 경로
- `--tokenizer-dir`: (선택사항) 토크나이저 디렉토리 경로
- `--text`: 합성할 한국어 텍스트
- `--output`: 출력 오디오 파일 경로
- `--plot`: 멜 스펙트로그램 플롯 생성 플래그

### 작업 흐름

1. **데이터 준비**: 한국어 오디오 및 텍스트 데이터 수집 및 정리
2. **전처리**: 오디오를 호환 가능한 형식으로 변환하고 텍스트 토큰화
3. **미세 조정**: 한국어 데이터셋으로 Dia 모델 학습
4. **추론**: 미세 조정된 모델을 사용하여 한국어 텍스트에서 음성 생성

### 기술적 세부사항

시스템은 다음과 같이 작동합니다:
1. 한글 텍스트를 자모 표현으로 변환
2. 자모 텍스트 토큰화
3. 미세 조정된 Dia 모델을 사용하여 멜 스펙트로그램 생성
4. Griffin-Lim 알고리즘을 사용하여 멜 스펙트로그램을 파형으로 변환
5. 무음을 제거하기 위한 오디오 후처리