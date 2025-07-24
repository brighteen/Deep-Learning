#!/usr/bin/env python3
"""
Deep Learning Repository Demo | 딥러닝 저장소 데모

This script demonstrates the capabilities and content of the Deep Learning repository.
이 스크립트는 딥러닝 저장소의 기능과 내용을 데모합니다.

Run this script to get an overview of what's available in the repository.
저장소에서 사용 가능한 것들의 개요를 보려면 이 스크립트를 실행하세요.
"""

import os
import sys
from datetime import datetime

# Import our utilities
from utils import ProjectManager, validate_deep_learning_environment, get_project_structure


def print_banner():
    """Print a welcome banner."""
    print("🎓" + "=" * 60 + "🎓")
    print("🧠        DEEP LEARNING REPOSITORY DEMO        🧠")
    print("🎓" + "=" * 60 + "🎓")
    print()


def print_section(title: str, emoji: str = "📋"):
    """Print a section header."""
    print(f"\n{emoji} {title}")
    print("-" * (len(title) + 4))


def demonstrate_repository_capabilities():
    """Main demonstration function."""
    print_banner()
    
    # Current date and time
    print(f"📅 Demo Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Repository Path: {os.getcwd()}")
    
    # Answer the main question
    print_section("현재 작업 가능한 항목들 (Available Tasks)", "🎯")
    print("이 저장소에서 현재 작업할 수 있는 항목들:")
    print()
    
    available_tasks = [
        "1. 📚 교육 컨텐츠 학습 및 실행",
        "   - CNN, DNN, NLP 이론 및 실습",
        "   - 객체 탐지 프로젝트 (YOLO 기반)",
        "   - 차원 축소 기법 (PCA, AutoEncoder)",
        "   - 비지도 학습 및 강화학습",
        "",
        "2. 💻 코드 실습 및 개발",
        "   - 날코딩 (밑바닥부터 신경망 구현)",
        "   - Jupyter 노트북 실습",
        "   - 실제 프로젝트 구현 및 개선",
        "",
        "3. 🛠️ 저장소 개선 작업",
        "   - 코드 리팩토링 및 최적화",
        "   - 문서화 개선 및 번역",
        "   - 테스트 코드 추가",
        "   - 새로운 기능 개발",
        "",
        "4. 🔬 연구 및 실험",
        "   - 최신 논문 구현",
        "   - 모델 성능 개선",
        "   - 새로운 데이터셋 적용",
        "",
        "5. 🌐 커뮤니티 기여",
        "   - 이슈 리포팅 및 해결",
        "   - 새로운 예제 추가",
        "   - 다른 개발자와의 협업"
    ]
    
    for task in available_tasks:
        print(task)
    
    # Repository analysis
    print_section("저장소 분석 (Repository Analysis)", "🔍")
    pm = ProjectManager()
    pm.print_summary()
    
    # Environment check
    print_section("환경 검증 (Environment Validation)", "🧪")
    is_valid, issues = validate_deep_learning_environment()
    
    if is_valid:
        print("✅ 환경이 올바르게 설정되었습니다!")
        print("✅ Environment is properly configured!")
    else:
        print("⚠️ 환경 설정에 문제가 있습니다:")
        print("⚠️ Environment configuration issues found:")
        for issue in issues:
            print(f"   - {issue}")
        print()
        print("💡 해결 방법 (Solution):")
        print("   pip install -r requirements.txt")
        print("   또는 SETUP.md 파일을 참조하세요")
    
    # Quick start examples
    print_section("빠른 시작 예제 (Quick Start Examples)", "🚀")
    
    examples = [
        {
            "title": "1. 환경 테스트",
            "command": "python test_environment.py",
            "description": "개발 환경이 올바르게 설정되었는지 확인"
        },
        {
            "title": "2. 날코딩 신경망 실습",
            "command": "cd '날코딩' && python '08-01.선형변환 레이어1개.py'",
            "description": "기본 신경망 구현 학습"
        },
        {
            "title": "3. CNN 이론 학습",
            "command": "cat 'Convolution Neural Network/01. CNN to VGGNet.md'",
            "description": "CNN 기본 원리 및 VGGNet 구조 이해"
        },
        {
            "title": "4. 객체 탐지 프로젝트",
            "command": "cd Object_Detection && ls -la",
            "description": "YOLO 기반 객체 탐지 프로젝트 탐색"
        },
        {
            "title": "5. Jupyter 노트북 실행",
            "command": "jupyter notebook",
            "description": "대화형 학습 환경에서 실습"
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}:")
        print(f"   명령어: {example['command']}")
        print(f"   설명: {example['description']}")
    
    # Available resources
    print_section("사용 가능한 자료 (Available Resources)", "📚")
    
    resources = [
        "📖 AVAILABLE_TASKS.md - 전체 작업 목록 (60+ 개선 항목)",
        "🛠️ SETUP.md - 환경 설정 가이드",
        "🤝 CONTRIBUTING.md - 기여 방법 안내",
        "📋 requirements.txt - 필요한 패키지 목록",
        "🧪 test_environment.py - 환경 테스트 도구",
        "🔧 utils.py - 공통 유틸리티 함수들"
    ]
    
    for resource in resources:
        print(f"   {resource}")
    
    # Learning paths
    print_section("학습 경로 추천 (Recommended Learning Paths)", "🎓")
    
    paths = [
        "🔰 초급자 (Beginner):",
        "   1. README.md 읽기 → 환경 설정 → 날코딩 실습",
        "   2. Deep Neural Network 이론 학습",
        "   3. 간단한 예제부터 시작",
        "",
        "🔥 중급자 (Intermediate):",
        "   1. CNN 이론 및 실습 → 객체 탐지 프로젝트",
        "   2. NLP 모델 (Transformer, BERT) 학습",
        "   3. 실제 데이터로 프로젝트 진행",
        "",
        "🚀 고급자 (Advanced):",
        "   1. 최신 논문 구현 → 모델 개선",
        "   2. 새로운 기능 개발 → 커뮤니티 기여",
        "   3. 연구 및 실험 진행"
    ]
    
    for path in paths:
        print(f"   {path}")
    
    # Footer
    print_section("시작해보세요! (Get Started!)", "🎉")
    print("1. 먼저 환경을 설정하세요: pip install -r requirements.txt")
    print("2. 환경을 테스트하세요: python test_environment.py")
    print("3. 관심 있는 분야의 폴더를 탐색하세요")
    print("4. Jupyter 노트북을 실행하여 실습해보세요")
    print("5. 질문이나 개선사항이 있으면 이슈를 등록하세요")
    print()
    print("🌟 Happy Learning! 즐거운 학습 되세요! 🌟")
    print("🎓" + "=" * 60 + "🎓")


if __name__ == "__main__":
    try:
        demonstrate_repository_capabilities()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error occurred during demo: {e}")
        print("Please check the environment and try again.")
        sys.exit(1)