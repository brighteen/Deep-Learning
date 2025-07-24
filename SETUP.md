# Setup Guide | 환경 설정 가이드

## 🛠️ Environment Setup | 환경 설정

### Option 1: Using pip (권장)

```bash
# 1. Create virtual environment (가상환경 생성)
python -m venv deep_learning_env

# 2. Activate virtual environment (가상환경 활성화)
# Windows:
deep_learning_env\Scripts\activate
# macOS/Linux:
source deep_learning_env/bin/activate

# 3. Install dependencies (의존성 설치)
pip install -r requirements.txt

# 4. Verify installation (설치 확인)
python -c "import torch; import cv2; import numpy; print('All packages installed successfully!')"
```

### Option 2: Using conda

```bash
# 1. Create conda environment (conda 환경 생성)
conda create -n deep_learning python=3.8

# 2. Activate environment (환경 활성화)
conda activate deep_learning

# 3. Install packages (패키지 설치)
conda install pytorch torchvision -c pytorch
conda install opencv numpy pandas matplotlib jupyter scikit-learn
pip install ultralytics

# 4. Install remaining packages
pip install -r requirements.txt
```

## 🧪 Quick Test | 빠른 테스트

환경이 올바르게 설정되었는지 확인:

```python
# test_environment.py
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

print("✅ All packages imported successfully!")
print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## 🚨 Common Issues | 일반적인 문제들

### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### OpenCV Issues
```bash
# If opencv-python conflicts with other packages
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python-headless
```

### YOLO Model Download
첫 실행 시 YOLO 모델이 자동으로 다운로드됩니다:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads automatically
```

## 🎯 Ready to Go! | 준비 완료!

환경 설정이 완료되면 다음 명령어로 시작할 수 있습니다:

```bash
# Start Jupyter Notebook
jupyter notebook

# Run a simple test
cd "날코딩"
python "08-01.선형변환 레이어1개.py"
```