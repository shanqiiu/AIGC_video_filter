"""
妫€娴嬪櫒妯″潡鍒濆鍖�
浠呬娇鐢╕OLO杩涜浜轰綋濮挎€佹娴�
"""

# YOLO妫€娴嬪櫒
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