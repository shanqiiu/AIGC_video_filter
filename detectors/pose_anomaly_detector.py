"""
人体姿态异常检测模块
检测人物身体部位扭曲、比例异常等问题

支持：
- 完整人体检测
- 部分身体检测（上半身/下半身）
- 自适应评估策略
- 可选YOLO集成（更robust）
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Optional, Tuple, Set
import math


class PoseAnomalyDetector:
    """人体姿态异常检测器（支持部分身体检测）"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
        # 初始化MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        
        # 定义身体部位分组（用于部分身体检测）
        self.body_parts = {
            'upper_body': [0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 头部、肩膀、手臂
            'lower_body': [23, 24, 25, 26, 27, 28, 29, 30, 31, 32],  # 臀部、腿部
            'torso': [11, 12, 23, 24],  # 躯干核心
            'head': [0, 7, 8, 9, 10],  # 头部
            'arms': [11, 12, 13, 14, 15, 16],  # 手臂
            'legs': [23, 24, 25, 26, 27, 28]  # 腿部
        }
        
        # 可见度阈值（判断关键点是否存在）
        self.visibility_threshold = 0.3
        
        # 定义正常的身体比例参考值
        self.normal_body_ratios = {
            'head_to_torso': (0.15, 0.25),      # 头部与躯干的比例
            'upper_to_lower_body': (0.45, 0.55),  # 上半身与下半身
            'arm_to_body': (0.35, 0.45),         # 手臂长度与身体高度
            'leg_to_body': (0.45, 0.55),         # 腿长与身体高度
            'shoulder_to_hip': (0.15, 0.25)      # 肩宽与臀宽
        }
        
        # 定义正常的关节角度范围（度）
        self.normal_joint_angles = {
            'elbow': (0, 180),
            'knee': (0, 180),
            'shoulder': (0, 180),
            'hip': (0, 180)
        }
    
    def detect_pose(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测图像中的人体姿态
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            姿态检测结果，包含关键点坐标和置信度
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 进行姿态检测
        results = self.pose.process(image_rgb)
        
        if not results.pose_landmarks:
            return None
        
        # 提取关键点
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        
        return {
            'landmarks': landmarks,
            'pose_landmarks': results.pose_landmarks
        }
    
    def detect_visible_body_parts(self, landmarks: List[Dict]) -> Dict[str, bool]:
        """
        检测哪些身体部位可见
        
        Args:
            landmarks: 关键点列表
            
        Returns:
            各身体部位的可见性字典
        """
        visible_parts = {}
        
        for part_name, keypoint_indices in self.body_parts.items():
            # 计算该部位的可见关键点比例
            visible_count = sum(
                1 for idx in keypoint_indices 
                if idx < len(landmarks) and 
                landmarks[idx]['visibility'] > self.visibility_threshold
            )
            
            # 如果超过50%的关键点可见，认为该部位可见
            visibility_ratio = visible_count / len(keypoint_indices)
            visible_parts[part_name] = visibility_ratio > 0.5
        
        return visible_parts
    
    def get_detection_mode(self, visible_parts: Dict[str, bool]) -> str:
        """
        根据可见的身体部位确定检测模式
        
        Args:
            visible_parts: 身体部位可见性字典
            
        Returns:
            检测模式：'full_body', 'upper_only', 'lower_only', 'torso_only', 'partial'
        """
        has_upper = visible_parts.get('upper_body', False)
        has_lower = visible_parts.get('lower_body', False)
        has_torso = visible_parts.get('torso', False)
        
        if has_upper and has_lower:
            return 'full_body'
        elif has_upper and not has_lower:
            return 'upper_only'
        elif has_lower and not has_upper:
            return 'lower_only'
        elif has_torso:
            return 'torso_only'
        else:
            return 'partial'
    
    def calculate_distance(self, point1: Dict, point2: Dict) -> float:
        """计算两个关键点之间的欧氏距离"""
        dx = point1['x'] - point2['x']
        dy = point1['y'] - point2['y']
        dz = point1['z'] - point2['z']
        return math.sqrt(dx**2 + dy**2 + dz**2)
    
    def calculate_angle(self, point1: Dict, point2: Dict, point3: Dict) -> float:
        """
        计算三个点形成的角度（以point2为顶点）
        
        Returns:
            角度值（度）
        """
        # 向量1: point2 -> point1
        v1 = np.array([point1['x'] - point2['x'], 
                       point1['y'] - point2['y'],
                       point1['z'] - point2['z']])
        
        # 向量2: point2 -> point3
        v2 = np.array([point3['x'] - point2['x'],
                       point3['y'] - point2['y'],
                       point3['z'] - point2['z']])
        
        # 计算夹角
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return math.degrees(angle)
    
    def check_body_proportions(self, landmarks: List[Dict], detection_mode: str = 'full_body') -> Dict[str, any]:
        """
        检查身体比例是否正常（自适应检测模式）
        
        Args:
            landmarks: 关键点列表
            detection_mode: 检测模式
            
        Returns:
            身体比例检查结果
        """
        # MediaPipe关键点索引
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        LEFT_EAR = 7
        RIGHT_EAR = 8
        
        results = {'detection_mode': detection_mode}
        anomalies = []
        
        # 检查关键点可见性
        def is_visible(idx):
            return idx < len(landmarks) and landmarks[idx]['visibility'] > self.visibility_threshold
        
        try:
            # 根据检测模式选择性地检查比例
            if detection_mode == 'full_body':
                # 完整身体检查
                if all(is_visible(i) for i in [LEFT_EAR, RIGHT_EAR, NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP, LEFT_ANKLE, RIGHT_ANKLE]):
                    # 计算头部高度（耳朵到鼻子）
                    head_height = (self.calculate_distance(landmarks[LEFT_EAR], landmarks[NOSE]) +
                                  self.calculate_distance(landmarks[RIGHT_EAR], landmarks[NOSE])) / 2
                    
                    # 计算躯干长度（肩膀到臀部）
                    torso_length = (self.calculate_distance(landmarks[LEFT_SHOULDER], landmarks[LEFT_HIP]) +
                                   self.calculate_distance(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_HIP])) / 2
                    
                    # 计算腿长（臀部到脚踝）
                    leg_length = (self.calculate_distance(landmarks[LEFT_HIP], landmarks[LEFT_ANKLE]) +
                                 self.calculate_distance(landmarks[RIGHT_HIP], landmarks[RIGHT_ANKLE])) / 2
                    
                    # 计算肩宽
                    shoulder_width = self.calculate_distance(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
                    
                    # 计算臀宽
                    hip_width = self.calculate_distance(landmarks[LEFT_HIP], landmarks[RIGHT_HIP])
                    
                    # 计算总身高
                    total_height = head_height + torso_length + leg_length
                    
                    # 检查头身比
                    if total_height > 0:
                        head_to_body_ratio = head_height / total_height
                        results['head_to_body_ratio'] = head_to_body_ratio
                        
                        expected_range = self.normal_body_ratios['head_to_torso']
                        if not (expected_range[0] <= head_to_body_ratio <= expected_range[1]):
                            anomalies.append(f"头身比异常: {head_to_body_ratio:.2f}")
                    
                    # 检查上下身比例
                    if (torso_length + leg_length) > 0:
                        upper_to_lower = torso_length / (torso_length + leg_length)
                        results['upper_to_lower_ratio'] = upper_to_lower
                        
                        expected_range = self.normal_body_ratios['upper_to_lower_body']
                        if not (expected_range[0] <= upper_to_lower <= expected_range[1]):
                            anomalies.append(f"上下身比例异常: {upper_to_lower:.2f}")
                    
                    # 检查肩臀比
                    if hip_width > 0:
                        shoulder_to_hip_ratio = shoulder_width / hip_width
                        results['shoulder_to_hip_ratio'] = shoulder_to_hip_ratio
                        
                        if shoulder_to_hip_ratio < 0.7 or shoulder_to_hip_ratio > 2.0:
                            anomalies.append(f"肩臀比异常: {shoulder_to_hip_ratio:.2f}")
            
            elif detection_mode == 'upper_only':
                # 只检查上半身比例（头、肩膀、手臂）
                if all(is_visible(i) for i in [LEFT_SHOULDER, RIGHT_SHOULDER]):
                    shoulder_width = self.calculate_distance(landmarks[LEFT_SHOULDER], landmarks[RIGHT_SHOULDER])
                    results['shoulder_width'] = shoulder_width
                    
                    # 检查头肩比（如果头部可见）
                    if is_visible(NOSE) and is_visible(LEFT_EAR):
                        head_height = self.calculate_distance(landmarks[LEFT_EAR], landmarks[NOSE])
                        head_to_shoulder = head_height / shoulder_width if shoulder_width > 0 else 0
                        results['head_to_shoulder_ratio'] = head_to_shoulder
                        
                        # 头宽肩比应在合理范围
                        if head_to_shoulder < 0.3 or head_to_shoulder > 0.8:
                            anomalies.append(f"头肩比异常: {head_to_shoulder:.2f}")
            
            elif detection_mode == 'lower_only':
                # 只检查下半身比例（臀部、腿）
                if all(is_visible(i) for i in [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]):
                    # 计算大腿和小腿比例
                    left_thigh = self.calculate_distance(landmarks[LEFT_HIP], landmarks[LEFT_KNEE])
                    left_calf = self.calculate_distance(landmarks[LEFT_KNEE], landmarks[LEFT_ANKLE])
                    right_thigh = self.calculate_distance(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE])
                    right_calf = self.calculate_distance(landmarks[RIGHT_KNEE], landmarks[RIGHT_ANKLE])
                    
                    avg_thigh = (left_thigh + right_thigh) / 2
                    avg_calf = (left_calf + right_calf) / 2
                    
                    if avg_thigh > 0:
                        thigh_calf_ratio = avg_calf / avg_thigh
                        results['thigh_calf_ratio'] = thigh_calf_ratio
                        
                        # 大腿通常比小腿长
                        if thigh_calf_ratio > 1.2 or thigh_calf_ratio < 0.6:
                            anomalies.append(f"大小腿比例异常: {thigh_calf_ratio:.2f}")
            
        except Exception as e:
            results['error'] = str(e)
        
        results['anomalies'] = anomalies
        results['has_proportion_anomaly'] = len(anomalies) > 0
        
        return results
    
    def check_joint_angles(self, landmarks: List[Dict]) -> Dict[str, any]:
        """
        检查关节角度是否异常
        
        Args:
            landmarks: 关键点列表
            
        Returns:
            关节角度检查结果
        """
        # 关键点索引
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28
        
        results = {}
        anomalies = []
        
        try:
            # 检查肘关节角度
            left_elbow_angle = self.calculate_angle(
                landmarks[LEFT_SHOULDER],
                landmarks[LEFT_ELBOW],
                landmarks[LEFT_WRIST]
            )
            right_elbow_angle = self.calculate_angle(
                landmarks[RIGHT_SHOULDER],
                landmarks[RIGHT_ELBOW],
                landmarks[RIGHT_WRIST]
            )
            
            results['left_elbow_angle'] = left_elbow_angle
            results['right_elbow_angle'] = right_elbow_angle
            
            # 检查膝关节角度
            left_knee_angle = self.calculate_angle(
                landmarks[LEFT_HIP],
                landmarks[LEFT_KNEE],
                landmarks[LEFT_ANKLE]
            )
            right_knee_angle = self.calculate_angle(
                landmarks[RIGHT_HIP],
                landmarks[RIGHT_KNEE],
                landmarks[RIGHT_ANKLE]
            )
            
            results['left_knee_angle'] = left_knee_angle
            results['right_knee_angle'] = right_knee_angle
            
            # 检查肩关节角度
            left_shoulder_angle = self.calculate_angle(
                landmarks[LEFT_ELBOW],
                landmarks[LEFT_SHOULDER],
                landmarks[LEFT_HIP]
            )
            right_shoulder_angle = self.calculate_angle(
                landmarks[RIGHT_ELBOW],
                landmarks[RIGHT_SHOULDER],
                landmarks[RIGHT_HIP]
            )
            
            results['left_shoulder_angle'] = left_shoulder_angle
            results['right_shoulder_angle'] = right_shoulder_angle
            
            # 检查异常角度（过度弯曲或不自然）
            # 肘关节反向弯曲检测
            if left_elbow_angle < 30 or left_elbow_angle > 200:
                anomalies.append(f"左肘关节角度异常: {left_elbow_angle:.1f}°")
            if right_elbow_angle < 30 or right_elbow_angle > 200:
                anomalies.append(f"右肘关节角度异常: {right_elbow_angle:.1f}°")
            
            # 膝关节反向弯曲检测
            if left_knee_angle < 30 or left_knee_angle > 200:
                anomalies.append(f"左膝关节角度异常: {left_knee_angle:.1f}°")
            if right_knee_angle < 30 or right_knee_angle > 200:
                anomalies.append(f"右膝关节角度异常: {right_knee_angle:.1f}°")
                
        except Exception as e:
            results['error'] = str(e)
        
        results['anomalies'] = anomalies
        results['has_angle_anomaly'] = len(anomalies) > 0
        
        return results
    
    def check_symmetry(self, landmarks: List[Dict]) -> Dict[str, any]:
        """
        检查身体对称性
        
        Args:
            landmarks: 关键点列表
            
        Returns:
            对称性检查结果
        """
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        
        results = {}
        anomalies = []
        
        try:
            # 计算左右肢体长度
            left_arm_length = (
                self.calculate_distance(landmarks[LEFT_SHOULDER], landmarks[LEFT_ELBOW]) +
                self.calculate_distance(landmarks[LEFT_ELBOW], landmarks[LEFT_WRIST])
            )
            right_arm_length = (
                self.calculate_distance(landmarks[RIGHT_SHOULDER], landmarks[RIGHT_ELBOW]) +
                self.calculate_distance(landmarks[RIGHT_ELBOW], landmarks[RIGHT_WRIST])
            )
            
            left_leg_length = (
                self.calculate_distance(landmarks[LEFT_HIP], landmarks[LEFT_KNEE]) +
                self.calculate_distance(landmarks[LEFT_KNEE], landmarks[27])  # 脚踝
            )
            right_leg_length = (
                self.calculate_distance(landmarks[RIGHT_HIP], landmarks[RIGHT_KNEE]) +
                self.calculate_distance(landmarks[RIGHT_KNEE], landmarks[28])  # 脚踝
            )
            
            # 计算对称性比例
            arm_symmetry = min(left_arm_length, right_arm_length) / (max(left_arm_length, right_arm_length) + 1e-6)
            leg_symmetry = min(left_leg_length, right_leg_length) / (max(left_leg_length, right_leg_length) + 1e-6)
            
            results['arm_symmetry'] = arm_symmetry
            results['leg_symmetry'] = leg_symmetry
            
            # 对称性阈值（0.85表示15%的差异容忍度）
            symmetry_threshold = 0.85
            
            if arm_symmetry < symmetry_threshold:
                anomalies.append(f"手臂长度不对称: {arm_symmetry:.2f}")
            
            if leg_symmetry < symmetry_threshold:
                anomalies.append(f"腿长不对称: {leg_symmetry:.2f}")
                
        except Exception as e:
            results['error'] = str(e)
        
        results['anomalies'] = anomalies
        results['has_symmetry_anomaly'] = len(anomalies) > 0
        
        return results
    
    def evaluate(self, image: np.ndarray) -> Dict[str, any]:
        """
        综合评估人体姿态（自适应检测模式）
        
        Args:
            image: 输入图像
            
        Returns:
            完整的姿态评估结果
        """
        # 检测姿态
        pose_result = self.detect_pose(image)
        
        if pose_result is None:
            return {
                'has_person': False,
                'pose_detected': False,
                'message': '未检测到人体'
            }
        
        landmarks = pose_result['landmarks']
        
        # 检测可见的身体部位
        visible_parts = self.detect_visible_body_parts(landmarks)
        detection_mode = self.get_detection_mode(visible_parts)
        
        # 各项检查（使用自适应模式）
        proportion_results = self.check_body_proportions(landmarks, detection_mode)
        angle_results = self.check_joint_angles(landmarks)
        symmetry_results = self.check_symmetry(landmarks)
        
        # 收集所有异常
        all_anomalies = (
            proportion_results.get('anomalies', []) +
            angle_results.get('anomalies', []) +
            symmetry_results.get('anomalies', [])
        )
        
        # 计算置信度分数
        avg_visibility = np.mean([lm['visibility'] for lm in landmarks])
        
        # 计算异常分数（0-1，越低越好）
        anomaly_count = len(all_anomalies)
        anomaly_score = min(anomaly_count / 5.0, 1.0)  # 假设5个以上异常为严重
        
        # 综合评分（0-1，越高越好）
        pose_quality_score = (1.0 - anomaly_score) * avg_visibility
        
        return {
            'has_person': True,
            'pose_detected': True,
            'detection_mode': detection_mode,  # 新增：检测模式
            'visible_body_parts': visible_parts,  # 新增：可见部位
            'avg_confidence': float(avg_visibility),
            'proportion_check': proportion_results,
            'joint_angle_check': angle_results,
            'symmetry_check': symmetry_results,
            'all_anomalies': all_anomalies,
            'anomaly_count': anomaly_count,
            'pose_quality_score': float(pose_quality_score),
            'is_pose_normal': anomaly_count == 0 and avg_visibility > 0.5,
            'note': f'检测模式: {detection_mode}（{"完整身体" if detection_mode == "full_body" else "部分身体"}）'
        }
    
    def __del__(self):
        """释放资源"""
        if hasattr(self, 'pose'):
            self.pose.close()

