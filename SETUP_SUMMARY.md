# AI_Assist 설정 요약

## 발생한 에러와 해결 방법

### 1. TorchInductor C++ 컴파일 에러
**에러:** `fatal error C1083: Cannot open include file: 'algorithm': No such file or directory`

**원인:** 
- Windows에서 PyTorch의 TorchInductor가 C++ 커널을 컴파일하려고 시도
- Visual Studio 환경 변수가 제대로 설정되지 않음

**해결:**
- 환경 변수 설정: `TORCH_COMPILE_DISABLE=1`, `TORCH_DYNAMO_DISABLE=1`
- `.env` 파일에 환경 변수 저장

### 2. CPU 버전 PyTorch 설치
**에러:** GPU를 사용하지 않고 CPU 모드로만 실행

**원인:**
- PyPI에서 기본적으로 CPU 버전 PyTorch 다운로드
- 가상환경에 CUDA 버전 설치 실패

**해결:**
- 시스템 Python에 CUDA 버전 PyTorch 설치
- `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118`

### 3. TorchCodec 의존성 문제
**에러:** `ImportError: TorchCodec is required for load_with_torchcodec`

**원인:**
- 최신 torchaudio가 torchcodec 필요
- FFmpeg 설치 문제

**해결:**
- 이전 버전 torchaudio 사용 (2.4.1)
- 또는 torchcodec 제거

## 최종 설정

### 사용 환경
- **Python:** 시스템 Python 3.11
- **PyTorch:** 2.7.1+cu118 (CUDA 지원)
- **실행 방법:**
  - CMD: `set TORCH_COMPILE_DISABLE=1 && set TORCH_DYNAMO_DISABLE=1 && python sample.py`
  - Git Bash: `bash run.sh`

### 필수 환경 변수
```
TORCH_COMPILE_DISABLE=1
TORCH_DYNAMO_DISABLE=1
```

### 성능
- GPU 사용: 50+ it/s
- CPU 사용: 10 it/s

## 파일 구조
- `sample.py` - 메인 스크립트
- `run.sh` - Git Bash 실행 스크립트
- `.env` - 환경 변수 설정
- `pyproject.toml` - 프로젝트 설정
