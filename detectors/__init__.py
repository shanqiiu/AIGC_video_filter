"""
�����ģ���ʼ��
��ʹ��YOLO����������̬���
"""

# YOLO�����
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