"""
时序一致性检测模块
检测视频帧间的连续性、抖动、闪烁等问题
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from scipy.signal import savgol_filter
from scipy.spatial.distance import cosine


class TemporalConsistencyDetector:
    """时序一致性检测器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
    def calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        计算两帧之间的差异
        
        Args:
            frame1, frame2: 输入帧
            
        Returns:
            帧间差异值
        """
        # 转换为灰度图
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 计算绝对差
        diff = cv2.absdiff(gray1, gray2)
        
        # 返回平均差异
        return float(np.mean(diff))
    
    def calculate_optical_flow(self, frame1: np.ndarray, frame2: np.ndarray) -> Dict[str, float]:
        """
        计算光流以分析运动
        
        Args:
            frame1, frame2: 连续两帧
            
        Returns:
            光流统计信息
        """
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # 使用Farneback光流算法
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # 计算光流的幅值和方向
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        return {
            'mean_magnitude': float(np.mean(magnitude)),
            'max_magnitude': float(np.max(magnitude)),
            'std_magnitude': float(np.std(magnitude)),
            'mean_angle': float(np.mean(angle))
        }
    
    def detect_flicker(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        检测视频闪烁
        
        Args:
            frames: 视频帧列表
            
        Returns:
            闪烁检测结果
        """
        if len(frames) < 3:
            return {'has_flicker': False, 'flicker_score': 0.0}
        
        # 计算每帧的平均亮度
        brightness_values = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightness_values.append(brightness)
        
        brightness_array = np.array(brightness_values)
        
        # 计算亮度变化率
        brightness_diff = np.abs(np.diff(brightness_array))
        
        # 检测突然的亮度变化
        flicker_threshold = 10.0  # 亮度变化阈值
        flicker_count = np.sum(brightness_diff > flicker_threshold)
        flicker_ratio = flicker_count / len(brightness_diff)
        
        # 计算亮度方差（稳定性指标）
        brightness_std = np.std(brightness_array)
        
        return {
            'brightness_values': brightness_values,
            'brightness_std': float(brightness_std),
            'flicker_count': int(flicker_count),
            'flicker_ratio': float(flicker_ratio),
            'has_flicker': flicker_ratio > 0.15,  # 15%以上的帧有闪烁
            'flicker_score': float(flicker_ratio)
        }
    
    def detect_jitter(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        检测视频抖动
        
        Args:
            frames: 视频帧列表
            
        Returns:
            抖动检测结果
        """
        if len(frames) < 3:
            return {'has_jitter': False, 'jitter_score': 0.0}
        
        # 使用SIFT或ORB特征点进行帧间配准
        orb = cv2.ORB_create()
        
        # 计算帧间的变换矩阵
        transform_params = []
        
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # 检测特征点
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                # 匹配特征点
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # 使用前N个最佳匹配
                good_matches = matches[:min(50, len(matches))]
                
                if len(good_matches) >= 4:
                    # 提取匹配点坐标
                    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
                    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
                    
                    # 计算仿射变换矩阵
                    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
                    
                    if M is not None:
                        # 提取平移参数
                        tx, ty = M[0, 2], M[1, 2]
                        transform_params.append((tx, ty))
                    else:
                        transform_params.append((0, 0))
                else:
                    transform_params.append((0, 0))
            else:
                transform_params.append((0, 0))
        
        if not transform_params:
            return {'has_jitter': False, 'jitter_score': 0.0}
        
        # 分析平移参数的稳定性
        tx_values = [p[0] for p in transform_params]
        ty_values = [p[1] for p in transform_params]
        
        # 计算平移的标准差（抖动程度）
        tx_std = np.std(tx_values)
        ty_std = np.std(ty_values)
        jitter_magnitude = np.sqrt(tx_std**2 + ty_std**2)
        
        # 归一化抖动分数
        jitter_score = min(jitter_magnitude / 10.0, 1.0)
        
        return {
            'transform_params': transform_params,
            'tx_std': float(tx_std),
            'ty_std': float(ty_std),
            'jitter_magnitude': float(jitter_magnitude),
            'jitter_score': float(jitter_score),
            'has_jitter': jitter_magnitude > 5.0
        }
    
    def calculate_motion_smoothness(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        计算运动平滑度
        
        Args:
            frames: 视频帧列表
            
        Returns:
            运动平滑度评估结果
        """
        if len(frames) < 3:
            return {'smoothness_score': 1.0}
        
        # 计算连续帧间的光流
        flow_magnitudes = []
        
        for i in range(len(frames) - 1):
            flow_info = self.calculate_optical_flow(frames[i], frames[i + 1])
            flow_magnitudes.append(flow_info['mean_magnitude'])
        
        flow_array = np.array(flow_magnitudes)
        
        # 使用Savitzky-Golay滤波器平滑运动曲线
        if len(flow_array) >= 5:
            smoothed_flow = savgol_filter(flow_array, window_length=5, polyorder=2)
            
            # 计算原始和平滑曲线的差异
            smoothness_error = np.mean(np.abs(flow_array - smoothed_flow))
        else:
            smoothness_error = np.std(flow_array)
        
        # 计算运动的突变次数
        flow_diff = np.abs(np.diff(flow_array))
        sudden_changes = np.sum(flow_diff > np.mean(flow_diff) * 2)
        
        # 归一化平滑度分数（0-1，越高越平滑）
        smoothness_score = 1.0 / (1.0 + smoothness_error)
        
        return {
            'flow_magnitudes': flow_magnitudes,
            'smoothness_error': float(smoothness_error),
            'sudden_changes': int(sudden_changes),
            'smoothness_score': float(smoothness_score),
            'is_smooth': smoothness_score > 0.7
        }
    
    def detect_scene_cuts(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        检测场景切换（可能表示生成不连贯）
        
        Args:
            frames: 视频帧列表
            
        Returns:
            场景切换检测结果
        """
        if len(frames) < 2:
            return {'scene_cuts': [], 'cut_count': 0}
        
        scene_cuts = []
        frame_diffs = []
        
        for i in range(len(frames) - 1):
            diff = self.calculate_frame_difference(frames[i], frames[i + 1])
            frame_diffs.append(diff)
            
            # 如果差异超过阈值，认为是场景切换
            if diff > 50.0:  # 阈值可调
                scene_cuts.append(i)
        
        return {
            'frame_differences': frame_diffs,
            'scene_cuts': scene_cuts,
            'cut_count': len(scene_cuts),
            'has_scene_cuts': len(scene_cuts) > 0
        }
    
    def calculate_color_consistency(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        计算色彩一致性
        
        Args:
            frames: 视频帧列表
            
        Returns:
            色彩一致性评估结果
        """
        if len(frames) < 2:
            return {'color_consistency_score': 1.0}
        
        # 计算每帧的颜色直方图
        color_histograms = []
        
        for frame in frames:
            # 转换到HSV空间
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 计算H和S通道的直方图
            h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            
            # 归一化
            h_hist = h_hist.flatten() / (h_hist.sum() + 1e-6)
            s_hist = s_hist.flatten() / (s_hist.sum() + 1e-6)
            
            color_histograms.append((h_hist, s_hist))
        
        # 计算相邻帧的直方图相似度
        similarities = []
        
        for i in range(len(color_histograms) - 1):
            h_sim = 1.0 - cosine(color_histograms[i][0], color_histograms[i + 1][0])
            s_sim = 1.0 - cosine(color_histograms[i][1], color_histograms[i + 1][1])
            
            avg_sim = (h_sim + s_sim) / 2
            similarities.append(avg_sim)
        
        # 计算平均一致性
        color_consistency = np.mean(similarities) if similarities else 1.0
        
        return {
            'frame_similarities': similarities,
            'color_consistency_score': float(color_consistency),
            'is_color_consistent': color_consistency > 0.85
        }
    
    def evaluate(self, frames: List[np.ndarray]) -> Dict[str, any]:
        """
        综合评估时序一致性
        
        Args:
            frames: 视频帧列表
            
        Returns:
            完整的时序一致性评估结果
        """
        if len(frames) < 2:
            return {
                'error': '帧数不足',
                'temporal_consistency_score': 0.0
            }
        
        # 各项检测
        flicker_results = self.detect_flicker(frames)
        jitter_results = self.detect_jitter(frames)
        smoothness_results = self.calculate_motion_smoothness(frames)
        scene_cut_results = self.detect_scene_cuts(frames)
        color_consistency_results = self.calculate_color_consistency(frames)
        
        # 计算综合时序一致性分数
        # 各项指标权重
        flicker_weight = 0.20
        jitter_weight = 0.25
        smoothness_weight = 0.25
        scene_cut_weight = 0.10
        color_weight = 0.20
        
        # 归一化各项分数（转换为0-1，越高越好）
        flicker_score = 1.0 - flicker_results.get('flicker_score', 0.0)
        jitter_score = 1.0 - jitter_results.get('jitter_score', 0.0)
        smoothness_score = smoothness_results.get('smoothness_score', 1.0)
        scene_cut_score = 1.0 if scene_cut_results.get('cut_count', 0) == 0 else 0.5
        color_score = color_consistency_results.get('color_consistency_score', 1.0)
        
        # 加权计算总分
        temporal_consistency_score = (
            flicker_score * flicker_weight +
            jitter_score * jitter_weight +
            smoothness_score * smoothness_weight +
            scene_cut_score * scene_cut_weight +
            color_score * color_weight
        )
        
        return {
            'frame_count': len(frames),
            'flicker_detection': flicker_results,
            'jitter_detection': jitter_results,
            'motion_smoothness': smoothness_results,
            'scene_cuts': scene_cut_results,
            'color_consistency': color_consistency_results,
            'temporal_consistency_score': float(temporal_consistency_score),
            'is_temporally_consistent': temporal_consistency_score > 0.7
        }

