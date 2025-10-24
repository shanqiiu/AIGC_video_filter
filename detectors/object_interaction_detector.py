"""
物体交互与融合检测模块
检测AIGC视频中人物与物体的异常融合、不自然遮挡等问题

针对场景：
- 手臂与栏杆融合
- 身体部位与背景物体异常融合
- 侧身、多样姿势下的异常
- 深度关系不一致
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import ndimage
from skimage import segmentation, measure


class ObjectInteractionDetector:
    """物体交互与融合异常检测器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
    def detect_edge_abnormality(self, image: np.ndarray, 
                                pose_landmarks: Optional[List] = None) -> Dict[str, any]:
        """
        检测人体边缘异常（融合、模糊等）
        
        原理：
        - AIGC生成的融合问题通常在边缘表现为异常模糊或不连续
        - 正常遮挡有清晰边界，融合则边界模糊不清
        
        Args:
            image: 输入图像
            pose_landmarks: 姿态关键点（可选，用于定位人体区域）
            
        Returns:
            边缘异常检测结果
        """
        results = {}
        
        # 1. 使用多尺度边缘检测
        edges_fine = cv2.Canny(image, 50, 150)  # 细粒度边缘
        edges_coarse = cv2.Canny(image, 100, 200)  # 粗粒度边缘
        
        # 2. 计算边缘一致性
        # 正常边缘在不同尺度下应该一致，融合区域会不一致
        edge_consistency = cv2.bitwise_and(edges_fine, edges_coarse)
        consistency_ratio = np.sum(edge_consistency > 0) / (np.sum(edges_fine > 0) + 1e-6)
        
        results['edge_consistency'] = float(consistency_ratio)
        
        # 3. 如果有姿态信息，重点检查人体边缘
        if pose_landmarks is not None:
            body_edge_score = self._check_body_edge_quality(
                image, edges_fine, pose_landmarks
            )
            results['body_edge_quality'] = body_edge_score
        
        # 4. 检测异常模糊区域
        blur_map = self._compute_local_blur_map(image)
        edge_blur_score = self._analyze_edge_blur(edges_fine, blur_map)
        results['edge_blur_score'] = edge_blur_score
        
        # 5. 综合判断
        # 融合问题通常表现为：边缘一致性低、边缘区域模糊
        fusion_likelihood = 0.0
        
        if consistency_ratio < 0.6:  # 边缘不一致
            fusion_likelihood += 0.3
        
        if edge_blur_score > 0.7:  # 边缘过度模糊
            fusion_likelihood += 0.4
        
        if pose_landmarks and body_edge_score < 0.5:  # 人体边缘质量差
            fusion_likelihood += 0.3
        
        results['fusion_likelihood'] = min(fusion_likelihood, 1.0)
        results['has_fusion_anomaly'] = fusion_likelihood > 0.5
        
        return results
    
    def _check_body_edge_quality(self, image: np.ndarray, edges: np.ndarray,
                                 pose_landmarks: List) -> float:
        """检查人体边缘区域的质量"""
        # 创建人体区域mask
        h, w = image.shape[:2]
        body_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 基于关键点创建凸包
        visible_points = []
        for lm in pose_landmarks:
            if lm.get('visibility', 0) > 0.3:
                x = int(lm['x'] * w)
                y = int(lm['y'] * h)
                visible_points.append([x, y])
        
        if len(visible_points) < 3:
            return 0.5  # 关键点不足，无法判断
        
        # 绘制凸包作为人体区域
        hull = cv2.convexHull(np.array(visible_points))
        cv2.fillConvexPoly(body_mask, hull, 255)
        
        # 扩展边界（检查人体轮廓周围）
        kernel = np.ones((15, 15), np.uint8)
        body_boundary = cv2.dilate(body_mask, kernel) - body_mask
        
        # 计算边界区域的边缘质量
        boundary_edges = cv2.bitwise_and(edges, body_boundary)
        edge_density = np.sum(boundary_edges > 0) / (np.sum(body_boundary > 0) + 1e-6)
        
        # 正常人体边缘应该有合理的边缘密度（不太高也不太低）
        # 太低：可能融合了；太高：可能有很多噪声
        if 0.1 < edge_density < 0.4:
            return 0.8  # 正常
        elif edge_density < 0.05:
            return 0.2  # 疑似融合
        else:
            return 0.5  # 中等
    
    def _compute_local_blur_map(self, image: np.ndarray, 
                                window_size: int = 31) -> np.ndarray:
        """
        计算局部模糊图
        
        Returns:
            模糊图，值越大越模糊（0-1归一化）
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # 使用Laplacian方差作为清晰度指标
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        
        # 计算局部方差
        blur_map = np.zeros_like(gray, dtype=np.float32)
        half_win = window_size // 2
        
        for i in range(half_win, gray.shape[0] - half_win, 10):  # 降采样加速
            for j in range(half_win, gray.shape[1] - half_win, 10):
                window = laplacian[i-half_win:i+half_win, j-half_win:j+half_win]
                blur_map[i, j] = np.var(window)
        
        # 插值填充
        blur_map = cv2.resize(blur_map, (gray.shape[1], gray.shape[0]))
        
        # 归一化（反转：高方差=清晰，低方差=模糊）
        if np.max(blur_map) > 0:
            blur_map = 1.0 - (blur_map / np.max(blur_map))
        
        return blur_map
    
    def _analyze_edge_blur(self, edges: np.ndarray, blur_map: np.ndarray) -> float:
        """分析边缘区域的模糊程度"""
        # 扩展边缘区域
        kernel = np.ones((5, 5), np.uint8)
        edge_region = cv2.dilate(edges, kernel)
        
        # 计算边缘区域的平均模糊度
        edge_blur = blur_map[edge_region > 0]
        
        if len(edge_blur) == 0:
            return 0.5
        
        avg_blur = np.mean(edge_blur)
        return float(avg_blur)
    
    def detect_occlusion_consistency(self, image: np.ndarray,
                                    pose_landmarks: Optional[List] = None) -> Dict[str, any]:
        """
        检测遮挡一致性
        
        原理：
        - 正常遮挡：被遮挡部分有明确的深度顺序
        - 异常融合：深度关系混乱，颜色过渡不自然
        
        Args:
            image: 输入图像
            pose_landmarks: 姿态关键点
            
        Returns:
            遮挡一致性检测结果
        """
        results = {}
        
        # 1. 使用超像素分割识别物体边界
        segments = segmentation.slic(image, n_segments=100, compactness=10)
        
        # 2. 分析相邻区域的颜色过渡
        color_transition_score = self._analyze_color_transition(image, segments)
        results['color_transition_score'] = color_transition_score
        
        # 3. 如果有姿态信息，检查手臂等容易融合的部位
        if pose_landmarks is not None:
            limb_occlusion_score = self._check_limb_occlusion(
                image, segments, pose_landmarks
            )
            results['limb_occlusion_score'] = limb_occlusion_score
        else:
            limb_occlusion_score = 0.5
        
        # 4. 综合判断
        # 颜色过渡不自然 + 肢体遮挡异常 = 可能有融合问题
        is_consistent = (
            color_transition_score > 0.6 and
            limb_occlusion_score > 0.5
        )
        
        results['is_occlusion_consistent'] = is_consistent
        results['occlusion_anomaly_score'] = 1.0 - (
            color_transition_score * 0.5 + limb_occlusion_score * 0.5
        )
        
        return results
    
    def _analyze_color_transition(self, image: np.ndarray, segments: np.ndarray) -> float:
        """分析相邻区域的颜色过渡是否自然"""
        # 计算每个超像素的平均颜色
        num_segments = np.max(segments) + 1
        segment_colors = np.zeros((num_segments, 3))
        
        for i in range(num_segments):
            mask = segments == i
            if np.sum(mask) > 0:
                segment_colors[i] = np.mean(image[mask], axis=0)
        
        # 找相邻超像素并计算颜色差异
        color_diffs = []
        
        # 简化：随机采样一些边界
        for _ in range(min(50, num_segments)):
            i = np.random.randint(0, num_segments)
            
            # 找i的相邻区域
            mask_i = (segments == i)
            dilated = ndimage.binary_dilation(mask_i)
            boundary = dilated & (~mask_i)
            
            neighbors = np.unique(segments[boundary])
            
            for j in neighbors:
                if i != j:
                    color_diff = np.linalg.norm(segment_colors[i] - segment_colors[j])
                    color_diffs.append(color_diff)
        
        if not color_diffs:
            return 0.5
        
        # 分析颜色差异分布
        # 正常场景：有明显差异（物体边界）
        # 融合场景：过渡过于平滑或完全混乱
        avg_diff = np.mean(color_diffs)
        std_diff = np.std(color_diffs)
        
        # 归一化评分
        # 适度的差异和变化是正常的
        if 20 < avg_diff < 80 and std_diff > 10:
            return 0.8  # 正常
        elif avg_diff < 10:
            return 0.3  # 过渡过于平滑，疑似融合
        else:
            return 0.6  # 中等
    
    def _check_limb_occlusion(self, image: np.ndarray, segments: np.ndarray,
                              pose_landmarks: List) -> float:
        """检查手臂等肢体的遮挡是否合理"""
        h, w = image.shape[:2]
        
        # 定义手臂关键点索引（MediaPipe）
        limb_pairs = [
            (11, 13), (13, 15),  # 左臂：肩-肘-腕
            (12, 14), (14, 16),  # 右臂：肩-肘-腕
            (23, 25), (25, 27),  # 左腿：臀-膝-踝
            (24, 26), (26, 28),  # 右腿：臀-膝-踝
        ]
        
        limb_scores = []
        
        for idx1, idx2 in limb_pairs:
            if idx1 >= len(pose_landmarks) or idx2 >= len(pose_landmarks):
                continue
            
            lm1 = pose_landmarks[idx1]
            lm2 = pose_landmarks[idx2]
            
            # 两个关键点都要可见
            if lm1.get('visibility', 0) < 0.3 or lm2.get('visibility', 0) < 0.3:
                continue
            
            # 获取坐标
            x1, y1 = int(lm1['x'] * w), int(lm1['y'] * h)
            x2, y2 = int(lm2['x'] * w), int(lm2['y'] * h)
            
            # 检查连线上的颜色一致性
            # 正常肢体应该颜色相对一致，融合则会出现突变
            line_consistency = self._check_line_consistency(image, (x1, y1), (x2, y2))
            limb_scores.append(line_consistency)
        
        if not limb_scores:
            return 0.5
        
        return float(np.mean(limb_scores))
    
    def _check_line_consistency(self, image: np.ndarray, 
                               point1: Tuple[int, int], 
                               point2: Tuple[int, int]) -> float:
        """检查两点连线的颜色一致性"""
        # 获取连线上的像素
        num_samples = 20
        x_samples = np.linspace(point1[0], point2[0], num_samples, dtype=int)
        y_samples = np.linspace(point1[1], point2[1], num_samples, dtype=int)
        
        # 确保在图像范围内
        h, w = image.shape[:2]
        valid_samples = []
        
        for x, y in zip(x_samples, y_samples):
            if 0 <= x < w and 0 <= y < h:
                valid_samples.append(image[y, x])
        
        if len(valid_samples) < 5:
            return 0.5
        
        valid_samples = np.array(valid_samples)
        
        # 计算颜色方差
        color_std = np.mean(np.std(valid_samples, axis=0))
        
        # 正常肢体：颜色相对一致（低方差）
        # 融合：颜色突变（高方差）
        if color_std < 20:
            return 0.8  # 一致性好
        elif color_std > 50:
            return 0.3  # 一致性差，疑似融合
        else:
            return 0.6  # 中等
    
    def detect_pose_plausibility(self, pose_landmarks: List) -> Dict[str, any]:
        """
        检测姿势的合理性（针对各种姿势，不只是正面站立）
        
        原理：
        - 不限制具体姿势，而是检查物理规律（如重心、支撑等）
        - 检测不自然的扭曲和违反人体工学的姿势
        
        Args:
            pose_landmarks: 姿态关键点
            
        Returns:
            姿势合理性评估结果
        """
        results = {}
        anomalies = []
        
        # 1. 检查身体对称性（即使侧身也应该基本对称）
        symmetry_score = self._check_symmetry_general(pose_landmarks)
        results['symmetry_score'] = symmetry_score
        
        if symmetry_score < 0.7:
            anomalies.append(f"身体对称性异常: {symmetry_score:.2f}")
        
        # 2. 检查关节连续性（相邻关节应该在合理距离内）
        continuity_score = self._check_joint_continuity(pose_landmarks)
        results['continuity_score'] = continuity_score
        
        if continuity_score < 0.7:
            anomalies.append(f"关节连续性异常: {continuity_score:.2f}")
        
        # 3. 检查姿势稳定性（重心是否合理）
        stability_score = self._check_pose_stability(pose_landmarks)
        results['stability_score'] = stability_score
        
        if stability_score < 0.5:
            anomalies.append(f"姿势不稳定: {stability_score:.2f}")
        
        # 4. 综合评分
        plausibility_score = (
            symmetry_score * 0.3 +
            continuity_score * 0.4 +
            stability_score * 0.3
        )
        
        results['plausibility_score'] = float(plausibility_score)
        results['anomalies'] = anomalies
        results['is_plausible'] = plausibility_score > 0.6 and len(anomalies) == 0
        
        return results
    
    def _check_symmetry_general(self, pose_landmarks: List) -> float:
        """检查一般对称性（不要求完全对称，但左右应该协调）"""
        # 计算左右关键点的相对位置
        # 即使侧身，左右手/腿的相对位置也应该协调
        
        pairs = [
            (11, 12),  # 左右肩
            (23, 24),  # 左右髋
            (13, 14),  # 左右肘
            (25, 26),  # 左右膝
        ]
        
        symmetry_scores = []
        
        for left_idx, right_idx in pairs:
            if left_idx >= len(pose_landmarks) or right_idx >= len(pose_landmarks):
                continue
            
            left_lm = pose_landmarks[left_idx]
            right_lm = pose_landmarks[right_idx]
            
            if left_lm.get('visibility', 0) < 0.3 or right_lm.get('visibility', 0) < 0.3:
                continue
            
            # 计算深度差异（z坐标）
            # 正常情况下，左右的深度差不应该太大（除非明显侧身）
            z_diff = abs(left_lm.get('z', 0) - right_lm.get('z', 0))
            
            # y坐标（高度）也应该相近
            y_diff = abs(left_lm['y'] - right_lm['y'])
            
            # 评分：差异越小越好，但允许一定差异（适应侧身等）
            score = 1.0 / (1.0 + z_diff * 10 + y_diff * 2)
            symmetry_scores.append(score)
        
        if not symmetry_scores:
            return 0.5
        
        return float(np.mean(symmetry_scores))
    
    def _check_joint_continuity(self, pose_landmarks: List) -> float:
        """检查关节连续性"""
        # 相邻关节之间的距离应该在合理范围内
        # 如果突然出现很大的gap，可能是融合或扭曲
        
        joint_chains = [
            [11, 13, 15],  # 左臂链
            [12, 14, 16],  # 右臂链
            [23, 25, 27],  # 左腿链
            [24, 26, 28],  # 右腿链
        ]
        
        continuity_scores = []
        
        for chain in joint_chains:
            distances = []
            
            for i in range(len(chain) - 1):
                idx1, idx2 = chain[i], chain[i+1]
                
                if idx1 >= len(pose_landmarks) or idx2 >= len(pose_landmarks):
                    continue
                
                lm1 = pose_landmarks[idx1]
                lm2 = pose_landmarks[idx2]
                
                if lm1.get('visibility', 0) < 0.3 or lm2.get('visibility', 0) < 0.3:
                    continue
                
                # 计算3D距离
                dist = np.sqrt(
                    (lm1['x'] - lm2['x'])**2 +
                    (lm1['y'] - lm2['y'])**2 +
                    (lm1.get('z', 0) - lm2.get('z', 0))**2
                )
                distances.append(dist)
            
            if len(distances) >= 1:
                # 检查距离的一致性（同一条链上的段长应该相近）
                if len(distances) > 1:
                    dist_std = np.std(distances)
                    dist_mean = np.mean(distances)
                    
                    # 变异系数：标准差/均值
                    cv = dist_std / (dist_mean + 1e-6)
                    
                    # cv越小越好（表示长度一致）
                    score = 1.0 / (1.0 + cv * 5)
                    continuity_scores.append(score)
        
        if not continuity_scores:
            return 0.5
        
        return float(np.mean(continuity_scores))
    
    def _check_pose_stability(self, pose_landmarks: List) -> float:
        """检查姿势稳定性（重心是否合理）"""
        # 计算重心位置
        visible_points = []
        
        for lm in pose_landmarks:
            if lm.get('visibility', 0) > 0.3:
                visible_points.append([lm['x'], lm['y'], lm.get('z', 0)])
        
        if len(visible_points) < 5:
            return 0.5
        
        visible_points = np.array(visible_points)
        center_of_mass = np.mean(visible_points, axis=0)
        
        # 检查重心是否在合理位置
        # 正常站立/坐姿：重心应该在中下部
        # 这里简化处理：检查重心的y坐标
        
        # 获取脚踝位置（支撑点）
        foot_indices = [27, 28]  # 左右脚踝
        foot_points = []
        
        for idx in foot_indices:
            if idx < len(pose_landmarks):
                lm = pose_landmarks[idx]
                if lm.get('visibility', 0) > 0.3:
                    foot_points.append(lm['y'])
        
        if not foot_points:
            return 0.6  # 看不到脚，无法判断
        
        avg_foot_y = np.mean(foot_points)
        
        # 重心应该在脚踝上方
        if center_of_mass[1] < avg_foot_y:
            stability = 0.8  # 合理
        else:
            stability = 0.4  # 重心在脚下方，不合理
        
        return float(stability)
    
    def evaluate(self, image: np.ndarray, pose_landmarks: Optional[List] = None) -> Dict[str, any]:
        """
        综合评估物体交互和融合问题
        
        Args:
            image: 输入图像
            pose_landmarks: 姿态关键点（可选）
            
        Returns:
            完整的物体交互评估结果
        """
        results = {}
        
        # 1. 边缘异常检测
        edge_results = self.detect_edge_abnormality(image, pose_landmarks)
        results['edge_analysis'] = edge_results
        
        # 2. 遮挡一致性检测
        occlusion_results = self.detect_occlusion_consistency(image, pose_landmarks)
        results['occlusion_analysis'] = occlusion_results
        
        # 3. 姿势合理性检测
        if pose_landmarks is not None:
            pose_results = self.detect_pose_plausibility(pose_landmarks)
            results['pose_plausibility'] = pose_results
        else:
            pose_results = {'plausibility_score': 0.5, 'is_plausible': True}
            results['pose_plausibility'] = pose_results
        
        # 4. 综合评分
        fusion_score = edge_results.get('fusion_likelihood', 0)
        occlusion_score = occlusion_results.get('occlusion_anomaly_score', 0)
        plausibility_score = 1.0 - pose_results.get('plausibility_score', 0.5)
        
        # 加权计算总体异常分数
        overall_anomaly_score = (
            fusion_score * 0.4 +          # 融合问题权重最高
            occlusion_score * 0.3 +       # 遮挡异常
            plausibility_score * 0.3      # 姿势不合理
        )
        
        results['overall_anomaly_score'] = float(overall_anomaly_score)
        results['has_interaction_anomaly'] = overall_anomaly_score > 0.5
        
        # 5. 问题汇总
        issues = []
        
        if edge_results.get('has_fusion_anomaly', False):
            issues.append(f"检测到融合问题（置信度: {fusion_score:.2f}）")
        
        if not occlusion_results.get('is_occlusion_consistent', True):
            issues.append(f"遮挡不一致（异常分数: {occlusion_score:.2f}）")
        
        if not pose_results.get('is_plausible', True):
            issues.append(f"姿势不合理（置信度: {plausibility_score:.2f}）")
            issues.extend([f"  - {a}" for a in pose_results.get('anomalies', [])])
        
        results['issues'] = issues
        results['recommendation'] = (
            "未检测到明显的融合或交互异常" if overall_anomaly_score < 0.5
            else f"检测到潜在问题，建议人工复查: {'; '.join(issues[:2])}"
        )
        
        return results

