"""
����YOLO��������̬�쳣���ģ��

������
pip install ultralytics  # YOLOv8
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import math

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("����: ultralyticsδ��װ��YOLO��ⲻ���á�ʹ�� 'pip install ultralytics' ��װ")


class PoseAnomalyDetectorYOLO:
    """����YOLO��������̬�쳣�����"""
    
    def __init__(self, config: dict = None, model_path: str = 'yolov8n-pose.pt'):
        """
        ��ʼ��YOLO��̬�����
        
        Args:
            config: �����ֵ�
            model_path: YOLOģ��·����Ĭ��ʹ��yolov8n-pose��
        """
        if not YOLO_AVAILABLE:
            raise ImportError("��Ҫ��װultralytics: pip install ultralytics")
        
        self.config = config or {}
        self.model = YOLO(model_path)
        
        # YOLO-Pose�ؼ��㶨�壨17���ؼ��㣩
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        self.confidence_threshold = 0.5
    
    def detect_pose(self, image: np.ndarray) -> Optional[Dict]:
        """
        ʹ��YOLO���������̬
        
        Args:
            image: ����ͼ�� (BGR��ʽ)
            
        Returns:
            ��̬�����
        """
        # YOLO����
        results = self.model(image, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return None
        
        # ��ȡ���Ŷ���ߵ�����
        keypoints_data = results[0].keypoints
        if keypoints_data.xy.shape[0] == 0:
            return None
        
        # ȡ��һ����⵽���ˣ������Ŷ���ߵģ�
        keypoints_xy = keypoints_data.xy[0].cpu().numpy()  # [17, 2]
        keypoints_conf = keypoints_data.conf[0].cpu().numpy() if keypoints_data.conf is not None else np.ones(17)
        
        # ת��Ϊ�ֵ��ʽ
        landmarks = []
        for i, (x, y) in enumerate(keypoints_xy):
            # ��һ������
            h, w = image.shape[:2]
            landmarks.append({
                'x': float(x / w),
                'y': float(y / h),
                'z': 0.0,  # YOLO-Pose���ṩ�����Ϣ
                'visibility': float(keypoints_conf[i]),
                'name': self.keypoint_names[i]
            })
        
        return {
            'landmarks': landmarks,
            'bbox': results[0].boxes.xyxy[0].cpu().numpy() if len(results[0].boxes) > 0 else None,
            'confidence': float(results[0].boxes.conf[0]) if len(results[0].boxes) > 0 else 0.0
        }
    
    def check_body_proportions(self, landmarks: List[Dict], image_shape: Tuple) -> Dict[str, any]:
        """
        ����������
        
        Args:
            landmarks: �ؼ����б�
            image_shape: ͼ��ߴ�
            
        Returns:
            ���������
        """
        results = {}
        anomalies = []
        
        # �ؼ�������
        NOSE = 0
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_KNEE = 13
        RIGHT_KNEE = 14
        LEFT_ANKLE = 15
        RIGHT_ANKLE = 16
        
        def is_visible(idx):
            return idx < len(landmarks) and landmarks[idx]['visibility'] > self.confidence_threshold
        
        def calculate_distance(point1, point2):
            h, w = image_shape[:2]
            x1, y1 = point1['x'] * w, point1['y'] * h
            x2, y2 = point2['x'] * w, point2['y'] * h
            return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        try:
            # ������Ƿ����
            if is_visible(LEFT_SHOULDER) and is_visible(RIGHT_SHOULDER):
                shoulder_width = calculate_distance(
                    landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER]
                )
                
                if is_visible(NOSE):
                    # ͷ����
                    head_shoulder_dist = calculate_distance(
                        landmarks[NOSE], landmarks[LEFT_SHOULDER]
                    )
                    
                    if shoulder_width > 0:
                        head_shoulder_ratio = head_shoulder_dist / shoulder_width
                        results['head_shoulder_ratio'] = head_shoulder_ratio
                        
                        # ͷ����������Ӧ�ú���
                        if head_shoulder_ratio < 0.2 or head_shoulder_ratio > 0.6:
                            anomalies.append(f"ͷ������쳣: {head_shoulder_ratio:.2f}")
            
            # ����С�ȱ���
            if all(is_visible(i) for i in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
                left_thigh = calculate_distance(landmarks[LEFT_HIP], landmarks[LEFT_KNEE])
                left_calf = calculate_distance(landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
                right_thigh = calculate_distance(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE])
                right_calf = calculate_distance(landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
                
                avg_thigh = (left_thigh + right_thigh) / 2
                avg_calf = (left_calf + right_calf) / 2
                
                if avg_thigh > 0:
                    thigh_calf_ratio = avg_calf / avg_thigh
                    results['thigh_calf_ratio'] = thigh_calf_ratio
                    
                    if thigh_calf_ratio > 1.3 or thigh_calf_ratio < 0.5:
                        anomalies.append(f"��С�ȱ����쳣: {thigh_calf_ratio:.2f}")
        
        except Exception as e:
            results['error'] = str(e)
        
        results['anomalies'] = anomalies
        results['has_proportion_anomaly'] = len(anomalies) > 0
        
        return results
    
    def check_joint_angles(self, landmarks: List[Dict], image_shape: Tuple) -> Dict[str, any]:
        """���ؽڽǶ�"""
        results = {}
        anomalies = []
        
        # �ؼ�������
        LEFT_SHOULDER = 5
        LEFT_ELBOW = 7
        LEFT_WRIST = 9
        LEFT_HIP = 11
        LEFT_KNEE = 13
        LEFT_ANKLE = 15
        
        def is_visible(idx):
            return idx < len(landmarks) and landmarks[idx]['visibility'] > self.confidence_threshold
        
        def calculate_angle(p1, p2, p3):
            """���������γɵĽǶ�"""
            h, w = image_shape[:2]
            v1 = np.array([p1['x'] * w - p2['x'] * w, p1['y'] * h - p2['y'] * h])
            v2 = np.array([p3['x'] * w - p2['x'] * w, p3['y'] * h - p2['y'] * h])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return math.degrees(angle)
        
        try:
            # �����ؽ�
            if all(is_visible(i) for i in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]):
                left_elbow_angle = calculate_angle(
                    landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST]
                )
                results['left_elbow_angle'] = left_elbow_angle
                
                if left_elbow_angle < 20 or left_elbow_angle > 210:
                    anomalies.append(f"����ؽڽǶ��쳣: {left_elbow_angle:.1f}��")
            
            # ���ϥ�ؽ�
            if all(is_visible(i) for i in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
                left_knee_angle = calculate_angle(
                    landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]
                )
                results['left_knee_angle'] = left_knee_angle
                
                if left_knee_angle < 20 or left_knee_angle > 210:
                    anomalies.append(f"��ϥ�ؽڽǶ��쳣: {left_knee_angle:.1f}��")
        
        except Exception as e:
            results['error'] = str(e)
        
        results['anomalies'] = anomalies
        results['has_angle_anomaly'] = len(anomalies) > 0
        
        return results
    
    def evaluate(self, image: np.ndarray) -> Dict[str, any]:
        """
        �ۺ�����������̬
        
        Args:
            image: ����ͼ��
            
        Returns:
            ��������̬�������
        """
        # YOLO���
        pose_result = self.detect_pose(image)
        
        if pose_result is None:
            return {
                'has_person': False,
                'pose_detected': False,
                'message': 'δ��⵽����',
                'method': 'YOLO'
            }
        
        landmarks = pose_result['landmarks']
        
        # ������
        proportion_results = self.check_body_proportions(landmarks, image.shape)
        angle_results = self.check_joint_angles(landmarks, image.shape)
        
        # �ռ������쳣
        all_anomalies = (
            proportion_results.get('anomalies', []) +
            angle_results.get('anomalies', [])
        )
        
        # �������Ŷ�
        avg_visibility = np.mean([lm['visibility'] for lm in landmarks])
        
        # ������������
        anomaly_count = len(all_anomalies)
        anomaly_score = min(anomaly_count / 5.0, 1.0)
        pose_quality_score = (1.0 - anomaly_score) * avg_visibility
        
        return {
            'has_person': True,
            'pose_detected': True,
            'method': 'YOLO',
            'bbox': pose_result.get('bbox'),
            'detection_confidence': pose_result.get('confidence', 0.0),
            'avg_confidence': float(avg_visibility),
            'proportion_check': proportion_results,
            'joint_angle_check': angle_results,
            'all_anomalies': all_anomalies,
            'anomaly_count': anomaly_count,
            'pose_quality_score': float(pose_quality_score),
            'is_pose_normal': anomaly_count == 0 and avg_visibility > 0.5,
            'note': f'ʹ��YOLO��⣬���Ŷ�: {avg_visibility:.2f}'
        }


# ��ݺ�������ȡYOLO�����
def get_pose_detector(config: dict = None):
    """
    ��ȡYOLO��̬�����
    
    Args:
        config: �����ֵ�
        
    Returns:
        YOLO��̬�����ʵ��
    """
    if YOLO_AVAILABLE:
        return PoseAnomalyDetectorYOLO(config)
    else:
        raise ImportError("YOLO�����ã��밲װultralytics: pip install ultralytics")