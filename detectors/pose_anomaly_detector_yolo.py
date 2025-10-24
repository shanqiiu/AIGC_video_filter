"""
基于YOLO的人体姿态异常检测模块（可选）

优势：
- 更robust的人体检测
- 支持实例分割
- 处理遮挡和部分身体更准确
- 适合复杂场景

依赖：
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
    print("警告: ultralytics未安装，YOLO检测不可用。使用 'pip install ultralytics' 安装")


class PoseAnomalyDetectorYOLO:
    """基于YOLO的人体姿态异常检测器"""
    
    def __init__(self, config: dict = None, model_path: str = 'yolov8n-pose.pt'):
        """
        初始化YOLO姿态检测器
        
        Args:
            config: 配置字典
            model_path: YOLO模型路径（默认使用yolov8n-pose）
        """
        if not YOLO_AVAILABLE:
            raise ImportError("需要安装ultralytics: pip install ultralytics")
        
        self.config = config or {}
        self.model = YOLO(model_path)
        
        # YOLO-Pose关键点定义（17个关键点）
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # 身体部位分组
        self.body_parts = {
            'head': [0, 1, 2, 3, 4],
            'upper_body': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'lower_body': [11, 12, 13, 14, 15, 16],
            'arms': [5, 6, 7, 8, 9, 10],
            'legs': [11, 12, 13, 14, 15, 16]
        }
        
        self.confidence_threshold = 0.5
    
    def detect_pose(self, image: np.ndarray) -> Optional[Dict]:
        """
        使用YOLO检测人体姿态
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            姿态检测结果
        """
        # YOLO推理
        results = self.model(image, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return None
        
        # 获取置信度最高的人物
        keypoints_data = results[0].keypoints
        if keypoints_data.xy.shape[0] == 0:
            return None
        
        # 取第一个检测到的人（或置信度最高的）
        keypoints_xy = keypoints_data.xy[0].cpu().numpy()  # [17, 2]
        keypoints_conf = keypoints_data.conf[0].cpu().numpy() if keypoints_data.conf is not None else np.ones(17)
        
        # 转换为字典格式
        landmarks = []
        for i, (x, y) in enumerate(keypoints_xy):
            # 归一化坐标
            h, w = image.shape[:2]
            landmarks.append({
                'x': float(x / w),
                'y': float(y / h),
                'z': 0.0,  # YOLO-Pose不提供深度信息
                'visibility': float(keypoints_conf[i]),
                'name': self.keypoint_names[i]
            })
        
        return {
            'landmarks': landmarks,
            'bbox': results[0].boxes.xyxy[0].cpu().numpy() if len(results[0].boxes) > 0 else None,
            'confidence': float(results[0].boxes.conf[0]) if len(results[0].boxes) > 0 else 0.0
        }
    
    def detect_visible_body_parts(self, landmarks: List[Dict]) -> Dict[str, bool]:
        """检测可见的身体部位"""
        visible_parts = {}
        
        for part_name, keypoint_indices in self.body_parts.items():
            visible_count = sum(
                1 for idx in keypoint_indices
                if idx < len(landmarks) and landmarks[idx]['visibility'] > self.confidence_threshold
            )
            visibility_ratio = visible_count / len(keypoint_indices)
            visible_parts[part_name] = visibility_ratio > 0.5
        
        return visible_parts
    
    def get_detection_mode(self, visible_parts: Dict[str, bool]) -> str:
        """确定检测模式"""
        has_upper = visible_parts.get('upper_body', False)
        has_lower = visible_parts.get('lower_body', False)
        
        if has_upper and has_lower:
            return 'full_body'
        elif has_upper and not has_lower:
            return 'upper_only'
        elif has_lower and not has_upper:
            return 'lower_only'
        else:
            return 'partial'
    
    def calculate_distance(self, point1: Dict, point2: Dict, image_shape: Tuple) -> float:
        """计算两点间距离（像素单位）"""
        h, w = image_shape[:2]
        x1, y1 = point1['x'] * w, point1['y'] * h
        x2, y2 = point2['x'] * w, point2['y'] * h
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def check_body_proportions(self, landmarks: List[Dict], detection_mode: str, 
                               image_shape: Tuple) -> Dict[str, any]:
        """
        检查身体比例（YOLO 17点模式）
        
        Args:
            landmarks: 关键点列表
            detection_mode: 检测模式
            image_shape: 图像尺寸
            
        Returns:
            比例检查结果
        """
        results = {'detection_mode': detection_mode}
        anomalies = []
        
        # YOLO关键点索引
        NOSE = 0
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_HIP = 11
        RIGHT_HIP = 12
        LEFT_KNEE = 13
        RIGHT_KNEE = 14
        LEFT_ANKLE = 15
        RIGHT_ANKLE = 16
        LEFT_EAR = 3
        RIGHT_EAR = 4
        
        def is_visible(idx):
            return idx < len(landmarks) and landmarks[idx]['visibility'] > self.confidence_threshold
        
        try:
            if detection_mode == 'full_body' or detection_mode == 'upper_only':
                # 检查肩宽是否合理
                if is_visible(LEFT_SHOULDER) and is_visible(RIGHT_SHOULDER):
                    shoulder_width = self.calculate_distance(
                        landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER], image_shape
                    )
                    
                    if is_visible(NOSE):
                        # 头宽肩比
                        head_shoulder_dist = self.calculate_distance(
                            landmarks[NOSE], landmarks[LEFT_SHOULDER], image_shape
                        )
                        
                        if shoulder_width > 0:
                            head_shoulder_ratio = head_shoulder_dist / shoulder_width
                            results['head_shoulder_ratio'] = head_shoulder_ratio
                            
                            # 头部到肩膀距离应该合理
                            if head_shoulder_ratio < 0.2 or head_shoulder_ratio > 0.6:
                                anomalies.append(f"头肩比例异常: {head_shoulder_ratio:.2f}")
            
            if detection_mode == 'full_body' or detection_mode == 'lower_only':
                # 检查大小腿比例
                if all(is_visible(i) for i in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]):
                    left_thigh = self.calculate_distance(
                        landmarks[LEFT_HIP], landmarks[LEFT_KNEE], image_shape
                    )
                    left_calf = self.calculate_distance(
                        landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE], image_shape
                    )
                    right_thigh = self.calculate_distance(
                        landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE], image_shape
                    )
                    right_calf = self.calculate_distance(
                        landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE], image_shape
                    )
                    
                    avg_thigh = (left_thigh + right_thigh) / 2
                    avg_calf = (left_calf + right_calf) / 2
                    
                    if avg_thigh > 0:
                        thigh_calf_ratio = avg_calf / avg_thigh
                        results['thigh_calf_ratio'] = thigh_calf_ratio
                        
                        if thigh_calf_ratio > 1.3 or thigh_calf_ratio < 0.5:
                            anomalies.append(f"大小腿比例异常: {thigh_calf_ratio:.2f}")
            
            # 完整身体检查
            if detection_mode == 'full_body':
                if all(is_visible(i) for i in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]):
                    torso_length = (
                        self.calculate_distance(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP], image_shape) +
                        self.calculate_distance(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP], image_shape)
                    ) / 2
                    
                    if all(is_visible(i) for i in [LEFT_HIP, LEFT_ANKLE, RIGHT_HIP, RIGHT_ANKLE]):
                        leg_length = (
                            self.calculate_distance(landmarks[LEFT_HIP], landmarks[LEFT_ANKLE], image_shape) +
                            self.calculate_distance(landmarks[RIGHT_HIP], landmarks[RIGHT_ANKLE], image_shape)
                        ) / 2
                        
                        if (torso_length + leg_length) > 0:
                            upper_lower_ratio = torso_length / (torso_length + leg_length)
                            results['upper_lower_ratio'] = upper_lower_ratio
                            
                            if upper_lower_ratio < 0.35 or upper_lower_ratio > 0.65:
                                anomalies.append(f"上下身比例异常: {upper_lower_ratio:.2f}")
        
        except Exception as e:
            results['error'] = str(e)
        
        results['anomalies'] = anomalies
        results['has_proportion_anomaly'] = len(anomalies) > 0
        
        return results
    
    def check_joint_angles(self, landmarks: List[Dict], image_shape: Tuple) -> Dict[str, any]:
        """检查关节角度"""
        results = {}
        anomalies = []
        
        # 关键点索引
        LEFT_SHOULDER = 5
        LEFT_ELBOW = 7
        LEFT_WRIST = 9
        RIGHT_SHOULDER = 6
        RIGHT_ELBOW = 8
        RIGHT_WRIST = 10
        LEFT_HIP = 11
        LEFT_KNEE = 13
        LEFT_ANKLE = 15
        RIGHT_HIP = 12
        RIGHT_KNEE = 14
        RIGHT_ANKLE = 16
        
        def is_visible(idx):
            return idx < len(landmarks) and landmarks[idx]['visibility'] > self.confidence_threshold
        
        def calculate_angle(p1, p2, p3):
            """计算三点形成的角度"""
            h, w = image_shape[:2]
            v1 = np.array([p1['x'] * w - p2['x'] * w, p1['y'] * h - p2['y'] * h])
            v2 = np.array([p3['x'] * w - p2['x'] * w, p3['y'] * h - p2['y'] * h])
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return math.degrees(angle)
        
        try:
            # 检查肘关节
            if all(is_visible(i) for i in [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]):
                left_elbow_angle = calculate_angle(
                    landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST]
                )
                results['left_elbow_angle'] = left_elbow_angle
                
                if left_elbow_angle < 20 or left_elbow_angle > 210:
                    anomalies.append(f"左肘关节角度异常: {left_elbow_angle:.1f}°")
            
            # 检查膝关节
            if all(is_visible(i) for i in [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE]):
                left_knee_angle = calculate_angle(
                    landmarks[LEFT_HIP], landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE]
                )
                results['left_knee_angle'] = left_knee_angle
                
                if left_knee_angle < 20 or left_knee_angle > 210:
                    anomalies.append(f"左膝关节角度异常: {left_knee_angle:.1f}°")
        
        except Exception as e:
            results['error'] = str(e)
        
        results['anomalies'] = anomalies
        results['has_angle_anomaly'] = len(anomalies) > 0
        
        return results
    
    def evaluate(self, image: np.ndarray) -> Dict[str, any]:
        """
        综合评估人体姿态
        
        Args:
            image: 输入图像
            
        Returns:
            完整的姿态评估结果
        """
        # YOLO检测
        pose_result = self.detect_pose(image)
        
        if pose_result is None:
            return {
                'has_person': False,
                'pose_detected': False,
                'message': '未检测到人体',
                'method': 'YOLO'
            }
        
        landmarks = pose_result['landmarks']
        
        # 检测可见部位和模式
        visible_parts = self.detect_visible_body_parts(landmarks)
        detection_mode = self.get_detection_mode(visible_parts)
        
        # 各项检查
        proportion_results = self.check_body_proportions(landmarks, detection_mode, image.shape)
        angle_results = self.check_joint_angles(landmarks, image.shape)
        
        # 收集所有异常
        all_anomalies = (
            proportion_results.get('anomalies', []) +
            angle_results.get('anomalies', [])
        )
        
        # 计算置信度
        avg_visibility = np.mean([lm['visibility'] for lm in landmarks])
        
        # 计算质量分数
        anomaly_count = len(all_anomalies)
        anomaly_score = min(anomaly_count / 5.0, 1.0)
        pose_quality_score = (1.0 - anomaly_score) * avg_visibility
        
        return {
            'has_person': True,
            'pose_detected': True,
            'method': 'YOLO',
            'detection_mode': detection_mode,
            'visible_body_parts': visible_parts,
            'bbox': pose_result.get('bbox'),
            'detection_confidence': pose_result.get('confidence', 0.0),
            'avg_confidence': float(avg_visibility),
            'proportion_check': proportion_results,
            'joint_angle_check': angle_results,
            'all_anomalies': all_anomalies,
            'anomaly_count': anomaly_count,
            'pose_quality_score': float(pose_quality_score),
            'is_pose_normal': anomaly_count == 0 and avg_visibility > 0.5,
            'note': f'检测模式: {detection_mode}（{"完整身体" if detection_mode == "full_body" else "部分身体"}）使用YOLO'
        }


# 便捷函数：根据配置选择检测器
def get_pose_detector(config: dict = None, use_yolo: bool = False):
    """
    获取姿态检测器
    
    Args:
        config: 配置字典
        use_yolo: 是否使用YOLO（需要安装ultralytics）
        
    Returns:
        姿态检测器实例
    """
    if use_yolo:
        if YOLO_AVAILABLE:
            return PoseAnomalyDetectorYOLO(config)
        else:
            print("YOLO不可用，回退到MediaPipe")
            from .pose_anomaly_detector import PoseAnomalyDetector
            return PoseAnomalyDetector(config)
    else:
        from .pose_anomaly_detector import PoseAnomalyDetector
        return PoseAnomalyDetector(config)

