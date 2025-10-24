"""
检测器模块初始化
仅使用YOLO进行人体姿态检测
"""

# YOLO检测器
try:
    from .pose_anomaly_detector_yolo import PoseAnomalyDetectorYOLO, get_pose_detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    get_pose_detector = None

__all__ = [
    'PoseAnomalyDetectorYOLO',
    'get_pose_detector',
    'YOLO_AVAILABLE'
]