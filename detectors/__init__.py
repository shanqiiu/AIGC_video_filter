"""
检测器模块初始化
"""

from .image_quality_detector import ImageQualityDetector
from .pose_anomaly_detector import PoseAnomalyDetector
from .temporal_consistency_detector import TemporalConsistencyDetector
from .object_interaction_detector import ObjectInteractionDetector

# 可选：YOLO增强版本（需要安装ultralytics）
try:
    from .pose_anomaly_detector_yolo import PoseAnomalyDetectorYOLO, get_pose_detector
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    get_pose_detector = None

__all__ = [
    'ImageQualityDetector',
    'PoseAnomalyDetector',
    'TemporalConsistencyDetector',
    'ObjectInteractionDetector',
    'PoseAnomalyDetectorYOLO',
    'get_pose_detector',
    'YOLO_AVAILABLE'
]

