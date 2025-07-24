#!/usr/bin/env python3
"""
Environment Test Script | 환경 테스트 스크립트
Tests if all required packages are properly installed and working.
필요한 패키지들이 올바르게 설치되고 작동하는지 테스트합니다.
"""

import sys
import traceback
from typing import List, Tuple

def test_package_import(package_name: str, import_statement: str) -> Tuple[bool, str]:
    """
    Test if a package can be imported successfully.
    패키지가 성공적으로 import되는지 테스트합니다.
    """
    try:
        exec(import_statement)
        return True, f"✅ {package_name} imported successfully"
    except ImportError as e:
        return False, f"❌ {package_name} failed to import: {str(e)}"
    except Exception as e:
        return False, f"❌ {package_name} error: {str(e)}"

def main():
    """Main test function | 메인 테스트 함수"""
    print("🧪 Deep Learning Environment Test")
    print("=" * 50)
    
    # List of packages to test
    test_packages = [
        ("NumPy", "import numpy as np; print(f'NumPy version: {np.__version__}')"),
        ("Pandas", "import pandas as pd; print(f'Pandas version: {pd.__version__}')"),
        ("Matplotlib", "import matplotlib.pyplot as plt; print(f'Matplotlib version: {plt.matplotlib.__version__}')"),
        ("OpenCV", "import cv2; print(f'OpenCV version: {cv2.__version__}')"),
        ("Scikit-learn", "import sklearn; print(f'Scikit-learn version: {sklearn.__version__}')"),
        ("PyTorch", "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"),
        ("Jupyter", "import jupyter; print('Jupyter core imported')"),
    ]
    
    # Optional packages
    optional_packages = [
        ("Ultralytics YOLO", "from ultralytics import YOLO; print('YOLO imported successfully')"),
        ("TensorFlow", "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"),
        ("Transformers", "import transformers; print(f'Transformers version: {transformers.__version__}')"),
    ]
    
    # Test core packages
    print("\n📦 Testing Core Packages:")
    all_passed = True
    for package_name, import_statement in test_packages:
        success, message = test_package_import(package_name, import_statement)
        print(f"  {message}")
        if not success:
            all_passed = False
    
    # Test optional packages
    print("\n📦 Testing Optional Packages:")
    for package_name, import_statement in optional_packages:
        success, message = test_package_import(package_name, import_statement)
        print(f"  {message}")
    
    # Basic functionality test
    print("\n🔧 Testing Basic Functionality:")
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        
        # Create a simple plot
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        
        # Test if matplotlib can create a figure (without showing it)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x, y)
        ax.set_title("Simple Sine Wave Test")
        plt.close(fig)  # Close to avoid display issues
        
        print("  ✅ NumPy and Matplotlib basic functionality working")
        
        # Test OpenCV basic functionality
        import cv2
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        print("  ✅ OpenCV basic functionality working")
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {str(e)}")
        all_passed = False
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All core packages are working correctly!")
        print("✨ You're ready to start your Deep Learning journey!")
    else:
        print("⚠️  Some core packages have issues.")
        print("📖 Please check the SETUP.md file for troubleshooting.")
    
    print("\n🚀 Quick Start Commands:")
    print("  jupyter notebook                 # Start Jupyter")
    print("  cd '날코딩' && python '08-01.선형변환 레이어1개.py'  # Run example")
    print("  cd 'Object_Detection'            # Explore object detection")

if __name__ == "__main__":
    main()