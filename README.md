# Deep Learning Repository 🧠

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

> 딥러닝 학습 및 실습을 위한 종합적인 저장소 | Comprehensive repository for Deep Learning study and practice

## 📚 Repository Overview | 저장소 개요

이 저장소는 딥러닝의 다양한 분야를 다루는 교육 및 실습 자료를 포함하고 있습니다.

This repository contains educational materials and practical implementations covering various areas of Deep Learning.

## 🗂️ Directory Structure | 디렉토리 구조

```
Deep-Learning/
├── 📁 Convolution Neural Network/    # CNN 관련 자료 및 구현
├── 📁 Deep Neural Network/           # DNN 이론 및 실습
├── 📁 Demension/                     # 차원축소 (PCA, AutoEncoder)
├── 📁 Natural Language Processing/   # NLP (Attention, Transformer, BERT, GPT)
├── 📁 Object_Detection/             # 객체탐지 프로젝트 (YOLO 기반)
├── 📁 Unsupervised Learning/        # 비지도학습
├── 📁 날코딩/                        # 날코딩 구현 (Raw implementations)
├── 📄 AVAILABLE_TASKS.md            # 작업 가능한 항목 목록
└── 📄 README.md                     # 이 파일
```

## 🚀 Quick Start | 빠른 시작

### Prerequisites | 사전 요구사항

```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install numpy opencv-python matplotlib jupyter pandas scikit-learn ultralytics
```

### Usage | 사용법

1. **CNN 학습하기**
   ```bash
   cd "Convolution Neural Network"
   # CNN 관련 마크다운 파일들과 실습 노트북 확인
   ```

2. **객체 탐지 실행하기**
   ```bash
   cd Object_Detection
   # YOLO 기반 객체 탐지 프로젝트들 실행
   ```

3. **날코딩 실습하기**
   ```bash
   cd 날코딩
   python "08-01.선형변환 레이어1개.py"
   ```

## 📖 Contents | 주요 내용

### 🧠 Deep Learning Fundamentals
- **CNN (Convolution Neural Network)**: VGGNet, Inception, ResNet, Computer Vision
- **DNN (Deep Neural Network)**: Transfer Learning, 기본 신경망 이론
- **Dimension Reduction**: PCA, AutoEncoder 구현 및 비교

### 🔤 Natural Language Processing
- **Attention Mechanism**: Self-attention, Multi-head attention
- **Transformer Architecture**: 트랜스포머 구조 이해
- **Modern Models**: BERT, GPT, Vision Transformer (ViT)

### 👁️ Computer Vision
- **Object Detection**: YOLO 기반 실시간 객체 탐지
- **Specialized Projects**: 닭 탐지 시스템, MHI(Motion History Image)
- **Image Processing**: OpenCV 활용 영상 처리

### 🎯 Hands-on Implementations
- **날코딩 (Raw Coding)**: 밑바닥부터 구현하는 신경망
- **Reinforcement Learning**: Q-Learning과 신경망 결합
- **Practical Projects**: 실제 데이터를 활용한 프로젝트들

## 🛠️ Technologies Used | 사용 기술

- **Python**: 주 프로그래밍 언어
- **NumPy**: 수치 연산
- **OpenCV**: 컴퓨터 비전
- **Jupyter Notebook**: 실습 환경
- **YOLO (Ultralytics)**: 객체 탐지
- **Scikit-learn**: 머신러닝 도구
- **Matplotlib**: 데이터 시각화

## 📈 Learning Path | 학습 경로

### 초급 (Beginner)
1. `날코딩/` - 기본 신경망 구현 이해
2. `Deep Neural Network/` - DNN 이론 학습
3. `Demension/` - 차원축소 기법 실습

### 중급 (Intermediate)
1. `Convolution Neural Network/` - CNN 심화 학습
2. `Object_Detection/` - 실제 프로젝트 구현
3. `Unsupervised Learning/` - 비지도학습 응용

### 고급 (Advanced)
1. `Natural Language Processing/` - 최신 NLP 모델 이해
2. Custom projects and optimization
3. Research and paper implementation

## 🤝 Contributing | 기여하기

이 저장소는 학습 목적으로 제작되었습니다. 개선사항이나 추가하고 싶은 내용이 있다면:

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Available Tasks | 작업 가능 항목

현재 이 저장소에서 작업할 수 있는 다양한 개선사항들이 있습니다:

- 📋 **[전체 작업 목록 보기](AVAILABLE_TASKS.md)**
- 🛠️ **[환경 설정 가이드](SETUP.md)**
- 🤝 **[기여 방법](CONTRIBUTING.md)**
- 🧪 **[환경 테스트](test_environment.py)**

주요 개선 영역:
- 🗂️ 저장소 구조 개선
- 📚 문서화 보강
- 🧪 테스트 코드 추가
- 🌐 영어 번역
- ⚡ 성능 최적화

### 즉시 시작 가능한 작업 (Quick Start Tasks)
1. **환경 설정**: `pip install -r requirements.txt`
2. **환경 테스트**: `python test_environment.py`
3. **예제 실행**: 각 디렉토리의 노트북 및 Python 파일 실행
4. **문서 개선**: 마크다운 파일들의 내용 보강 및 번역

## 📄 License | 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact | 연락처

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 통해 연락해 주세요.

---

**Happy Learning! 즐거운 학습 되세요! 🎓**