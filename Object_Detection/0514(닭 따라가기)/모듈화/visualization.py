import cv2
import numpy as np
from utils import put_text_on_image

def add_status_info(display_frame, info):
    """
    화면에 상태 정보를 표시합니다.
    
    Args:
        display_frame (numpy.ndarray): 표시할 프레임
        info (dict): 상태 정보를 담은 딕셔너리
        
    Returns:
        numpy.ndarray: 정보가 표시된 프레임
    """
    y_offset = 30
    for i, (key, value) in enumerate(info.items()):
        if isinstance(value, float):
            text = f"{key}: {value:.2f}"
        else:
            text = f"{key}: {value}"
        
        display_frame = put_text_on_image(
            display_frame,
            text,
            (10, y_offset + i * 35),  # 간격을 30에서 35로 늘려 겹침 방지
            25,
            info.get(f"{key}_color", (0, 255, 0))
        )
    
    return display_frame

def add_tracking_info(display_frame, tracker):
    """
    화면에 추적 통계 정보를 표시합니다.
    
    Args:
        display_frame (numpy.ndarray): 표시할 프레임
        tracker (ChickenTracker): 닭 추적기 인스턴스
        
    Returns:
        numpy.ndarray: 정보가 표시된 프레임
    """
    if not hasattr(tracker, 'get_tracking_stats'):
        return display_frame
    
    stats = tracker.get_tracking_stats()
    
    # 추적 정보 표시
    display_frame = put_text_on_image(display_frame, 
                f"ID 집합: {stats['id_sets']}개", 
                (10, 250), 25, (0, 255, 255))  # Y 좌표 240 -> 250
    display_frame = put_text_on_image(display_frame, 
                f"총 고유 ID: {stats['total_unique_ids']}개", 
                (10, 285), 25, (0, 255, 255))  # Y 좌표 270 -> 285
    display_frame = put_text_on_image(display_frame, 
                f"재등장 이벤트: {stats['reappearance_events']}회", 
                (10, 320), 25, (0, 255, 255))  # Y 좌표 300 -> 320
    
    return display_frame

def show_skipped_frame_info(frame, cell_coords, tracker, current_time):
    """
    프레임 스킵 정보를 표시합니다.
    
    Args:
        frame (numpy.ndarray): 원본 프레임
        cell_coords (tuple): 셀 좌표 정보 (y, x, height, width)
        tracker (ChickenTracker): 닭 추적기 인스턴스
        current_time (float): 현재 시간(초)
        
    Returns:
        numpy.ndarray: 수정된 프레임
    """
    y, x, cell_height, cell_width = cell_coords
    
    if hasattr(tracker, 'last_results') and tracker.last_results is not None:
        # 이전 결과 표시
        frame[y:y+cell_height, x:x+cell_width] = tracker.last_results
        
        # 현재 시간 정보 업데이트
        if hasattr(tracker, 'last_time'):
            display_time_skipped = put_text_on_image(
                frame[y:y+cell_height, x:x+cell_width].copy(),
                f"마지막 탐지: {tracker.last_time:.1f}초", 
                (10, 60), 20, (0, 0, 255))  # Y 좌표 30 -> 60으로 변경
            frame[y:y+cell_height, x:x+cell_width] = display_time_skipped
    
    return frame

def create_prompt_frame(display_frame, message, position=(10, 400), font_size=30, color=(255, 0, 0)):
    """
    사용자에게 프롬프트 메시지를 표시합니다.
    
    Args:
        display_frame (numpy.ndarray): 표시할 프레임
        message (str): 표시할 메시지
        position (tuple): 메시지 위치
        font_size (int): 폰트 크기
        color (tuple): 텍스트 색상
        
    Returns:
        numpy.ndarray: 메시지가 표시된 프레임
    """
    prompt_frame = display_frame.copy()
    prompt_frame = put_text_on_image(prompt_frame, message, position, font_size, color)
    return prompt_frame
