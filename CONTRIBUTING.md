# Contributing Guide | 기여 가이드

## 🤝 How to Contribute | 기여하는 방법

이 저장소는 딥러닝 학습과 연구를 위한 공개 프로젝트입니다. 여러분의 기여를 환영합니다!

This repository is an open project for Deep Learning education and research. We welcome your contributions!

## 🎯 Types of Contributions | 기여 유형

### 1. 📚 Documentation Improvements | 문서 개선
- README 파일 개선
- 코드 주석 추가 또는 개선
- 튜토리얼 및 가이드 작성
- 영어 번역 추가

### 2. 💻 Code Contributions | 코드 기여
- 버그 수정
- 새로운 기능 추가
- 성능 최적화
- 코드 리팩토링

### 3. 🧪 Testing | 테스트
- 단위 테스트 추가
- 통합 테스트 작성
- 버그 리포트

### 4. 📖 Educational Content | 교육 컨텐츠
- 새로운 예제 추가
- Jupyter 노트북 개선
- 실습 가이드 작성

## 🚀 Getting Started | 시작하기

### Step 1: Fork and Clone | 포크 및 클론

```bash
# 1. Fork this repository on GitHub
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Deep-Learning.git
cd Deep-Learning

# 3. Add upstream remote
git remote add upstream https://github.com/brighteen/Deep-Learning.git
```

### Step 2: Set Up Environment | 환경 설정

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Test environment
python test_environment.py
```

### Step 3: Create Feature Branch | 기능 브랜치 생성

```bash
# Create and switch to new branch
git checkout -b feature/your-feature-name

# Example branch names:
# feature/add-cnn-examples
# docs/improve-readme
# fix/object-detection-bug
# translate/korean-to-english
```

## 📝 Development Guidelines | 개발 가이드라인

### Code Style | 코드 스타일

```python
# Follow PEP 8 style guide
# Use meaningful variable and function names
# Add docstrings to functions and classes

def detect_objects(image_path: str, confidence_threshold: float = 0.5) -> List[Dict]:
    """
    Detect objects in an image using YOLO model.
    
    Args:
        image_path: Path to the input image
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        List of detected objects with their information
    """
    # Implementation here
    pass
```

### Documentation | 문서화

- **한국어/영어 병행**: 가능하면 한국어와 영어를 모두 제공
- **명확한 예제**: 실행 가능한 코드 예제 포함
- **이미지/다이어그램**: 복잡한 개념은 시각적 자료 추가

### Testing | 테스트

```bash
# Run tests before submitting
python -m pytest tests/

# Test specific functionality
python test_environment.py
```

## 📋 Contribution Checklist | 기여 체크리스트

### Before Submitting | 제출 전 확인사항

- [ ] 코드가 PEP 8 스타일을 따르는가?
- [ ] 새로운 기능에 대한 문서가 추가되었는가?
- [ ] 테스트가 통과하는가?
- [ ] 커밋 메시지가 명확한가?
- [ ] 관련 이슈가 있다면 링크되었는가?

### Commit Message Format | 커밋 메시지 형식

```
type: brief description

Detailed explanation if needed.

- Add specific changes
- Fix specific issues
- Improve specific functionality

Closes #issue_number
```

**Types:**
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅
- `refactor`: 코드 리팩토링
- `test`: 테스트 추가
- `chore`: 기타 변경사항

### Examples | 예시

```bash
feat: add BERT implementation for Korean text analysis

- Implement BERT model with Korean tokenizer
- Add preprocessing functions for Korean text
- Include example notebook with sample data
- Add comprehensive documentation

Closes #15

---

docs: improve README with English translation

- Add English version of project description
- Include setup instructions for both languages
- Add badges and better formatting
- Update directory structure explanation

---

fix: resolve YOLO model loading issue in object detection

- Fix path resolution for model files
- Add error handling for missing models
- Improve logging for debugging
- Update documentation

Closes #23
```

## 🔍 Code Review Process | 코드 리뷰 프로세스

1. **Submit Pull Request** | PR 제출
2. **Automated Tests** | 자동 테스트 실행
3. **Peer Review** | 동료 리뷰
4. **Address Feedback** | 피드백 반영
5. **Final Approval** | 최종 승인
6. **Merge** | 병합

## 🏷️ Issues and Labels | 이슈 및 라벨

### Common Labels | 주요 라벨

- `good first issue`: 초보자에게 적합한 이슈
- `help wanted`: 도움이 필요한 이슈
- `bug`: 버그 리포트
- `enhancement`: 기능 개선
- `documentation`: 문서 관련
- `translation`: 번역 작업
- `question`: 질문

### Issue Template | 이슈 템플릿

```markdown
## Description | 설명
Brief description of the issue or feature request.

## Current Behavior | 현재 상황
What currently happens?

## Expected Behavior | 예상 결과
What should happen?

## Steps to Reproduce | 재현 단계
1. Step 1
2. Step 2
3. Step 3

## Environment | 환경
- OS: 
- Python version:
- Package versions:

## Additional Context | 추가 정보
Any other context or screenshots.
```

## 🎉 Recognition | 기여자 인정

모든 기여자는 다음과 같이 인정받게 됩니다:

- **Contributors 섹션**에 이름 추가
- **Release Notes**에 기여 내용 명시
- **Social Media**에서 감사 인사

## 📞 Getting Help | 도움 받기

질문이나 도움이 필요하시면:

1. **GitHub Issues**를 통해 질문 등록
2. **Discussion** 탭에서 토론 참여
3. **Email**: [maintainer_email] (if available)

## 📄 License | 라이선스

기여함으로써 당신의 코드가 프로젝트와 같은 라이선스 하에 배포됨에 동의하는 것입니다.

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

**Thank you for contributing! 기여해 주셔서 감사합니다! 🙏**