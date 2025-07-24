"""
Common Utilities Module | 공통 유틸리티 모듈

This module contains common utility functions used across different
deep learning projects in this repository.

이 모듈은 저장소의 다양한 딥러닝 프로젝트에서 공통으로 사용되는
유틸리티 함수들을 포함합니다.
"""

import os
import sys
from typing import List, Tuple, Optional, Union
import logging

# Setup logging | 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_project_path(project_name: str) -> str:
    """
    Setup and return the project path.
    프로젝트 경로를 설정하고 반환합니다.
    
    Args:
        project_name: Name of the project directory
        
    Returns:
        Absolute path to the project directory
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.join(current_dir, project_name)
    
    if not os.path.exists(project_path):
        logger.warning(f"Project path does not exist: {project_path}")
    
    # Add to Python path if not already there
    if project_path not in sys.path:
        sys.path.append(project_path)
    
    return project_path


def check_file_exists(file_path: str, create_if_missing: bool = False) -> bool:
    """
    Check if a file exists and optionally create it.
    파일이 존재하는지 확인하고 선택적으로 생성합니다.
    
    Args:
        file_path: Path to the file to check
        create_if_missing: Whether to create the file if it doesn't exist
        
    Returns:
        True if file exists (or was created), False otherwise
    """
    if os.path.exists(file_path):
        return True
    
    if create_if_missing:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            # Create empty file
            with open(file_path, 'w') as f:
                f.write("")
            logger.info(f"Created file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to create file {file_path}: {e}")
            return False
    
    return False


def list_files_by_extension(directory: str, extensions: List[str]) -> List[str]:
    """
    List all files with specified extensions in a directory.
    디렉토리에서 지정된 확장자를 가진 모든 파일을 나열합니다.
    
    Args:
        directory: Directory to search in
        extensions: List of file extensions (e.g., ['.py', '.ipynb'])
        
    Returns:
        List of file paths matching the extensions
    """
    matching_files = []
    
    if not os.path.exists(directory):
        logger.error(f"Directory does not exist: {directory}")
        return matching_files
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                matching_files.append(os.path.join(root, file))
    
    return sorted(matching_files)


def get_project_structure(root_dir: str = None, max_depth: int = 3) -> str:
    """
    Generate a tree structure of the project.
    프로젝트의 트리 구조를 생성합니다.
    
    Args:
        root_dir: Root directory to start from (default: current directory)
        max_depth: Maximum depth to traverse
        
    Returns:
        String representation of the directory tree
    """
    if root_dir is None:
        root_dir = os.getcwd()
    
    def _build_tree(directory: str, prefix: str = "", depth: int = 0) -> str:
        if depth > max_depth:
            return ""
        
        items = []
        try:
            for item in sorted(os.listdir(directory)):
                if item.startswith('.'):
                    continue
                
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    items.append(f"{prefix}📁 {item}/")
                    if depth < max_depth:
                        sub_items = _build_tree(item_path, prefix + "  ", depth + 1)
                        if sub_items:
                            items.append(sub_items)
                else:
                    # Show only important file types
                    if any(item.endswith(ext) for ext in ['.py', '.ipynb', '.md', '.txt']):
                        items.append(f"{prefix}📄 {item}")
        except PermissionError:
            items.append(f"{prefix}❌ Permission denied")
        
        return "\n".join(items)
    
    return f"📁 {os.path.basename(root_dir)}/\n" + _build_tree(root_dir)


def validate_deep_learning_environment() -> Tuple[bool, List[str]]:
    """
    Validate that the deep learning environment is properly set up.
    딥러닝 환경이 올바르게 설정되었는지 검증합니다.
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check Python version
    if sys.version_info < (3, 7):
        issues.append("Python 3.7+ is required")
    
    # Check for common packages
    required_packages = ['numpy', 'matplotlib']
    optional_packages = ['torch', 'tensorflow', 'cv2', 'sklearn']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            issues.append(f"Required package missing: {package}")
    
    missing_optional = []
    for package in optional_packages:
        try:
            __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_optional:
        issues.append(f"Optional packages missing: {', '.join(missing_optional)}")
    
    return len(issues) == 0, issues


class ProjectManager:
    """
    A simple project manager for organizing deep learning experiments.
    딥러닝 실험을 정리하기 위한 간단한 프로젝트 매니저입니다.
    """
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.getcwd()
        self.projects = {}
        self._discover_projects()
    
    def _discover_projects(self):
        """Discover existing projects in the repository."""
        project_dirs = [
            "Convolution Neural Network",
            "Deep Neural Network", 
            "Demension",
            "Natural Language Processing",
            "Object_Detection",
            "Unsupervised Learning",
            "날코딩"
        ]
        
        for proj_dir in project_dirs:
            full_path = os.path.join(self.base_dir, proj_dir)
            if os.path.exists(full_path):
                self.projects[proj_dir] = {
                    'path': full_path,
                    'files': list_files_by_extension(full_path, ['.py', '.ipynb', '.md'])
                }
    
    def list_projects(self) -> List[str]:
        """List all discovered projects."""
        return list(self.projects.keys())
    
    def get_project_info(self, project_name: str) -> Optional[dict]:
        """Get information about a specific project."""
        return self.projects.get(project_name)
    
    def print_summary(self):
        """Print a summary of all projects."""
        print("🎯 Deep Learning Repository Summary")
        print("=" * 50)
        
        for name, info in self.projects.items():
            print(f"\n📁 {name}")
            print(f"   Path: {info['path']}")
            print(f"   Files: {len(info['files'])}")
            
            # Count file types
            py_files = sum(1 for f in info['files'] if f.endswith('.py'))
            nb_files = sum(1 for f in info['files'] if f.endswith('.ipynb'))
            md_files = sum(1 for f in info['files'] if f.endswith('.md'))
            
            print(f"   - Python files: {py_files}")
            print(f"   - Notebooks: {nb_files}")
            print(f"   - Documentation: {md_files}")


# Example usage | 사용 예제
if __name__ == "__main__":
    print("🛠️ Deep Learning Utilities Demo")
    print("=" * 40)
    
    # Create project manager
    pm = ProjectManager()
    pm.print_summary()
    
    # Validate environment
    print("\n🧪 Environment Validation:")
    is_valid, issues = validate_deep_learning_environment()
    if is_valid:
        print("✅ Environment is properly configured!")
    else:
        print("⚠️ Environment issues found:")
        for issue in issues:
            print(f"  - {issue}")
    
    # Show project structure
    print(f"\n📁 Project Structure (limited view):")
    structure = get_project_structure(max_depth=2)
    print(structure)