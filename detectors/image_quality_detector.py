"""
图像质量检测模块
检测图像失真、模糊度、噪声等问题
"""

import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, restoration
from typing import Dict, Tuple


class ImageQualityDetector:
    """图像质量检测器"""
    
    def __init__(self, config: dict = None):
        self.config = config or {}
        
    def detect_blur(self, image: np.ndarray) -> float:
        """
        检测图像模糊度（使用Laplacian方差）
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            模糊度分数，值越大越清晰
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(laplacian_var)
    
    def detect_noise(self, image: np.ndarray) -> float:
        """
        估计图像噪声水平
        
        Args:
            image: 输入图像
            
        Returns:
            噪声水平估计值，值越小噪声越少
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯拉普拉斯算子估计噪声
        sigma_est = restoration.estimate_sigma(gray, average_sigmas=True)
        return float(sigma_est)
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        计算图像锐度
        
        Args:
            image: 输入图像
            
        Returns:
            锐度分数
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用Sobel算子
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # 计算梯度幅值
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        sharpness = np.mean(magnitude)
        
        return float(sharpness)
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """
        计算图像对比度（使用标准差）
        
        Args:
            image: 输入图像
            
        Returns:
            对比度分数
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std()
        return float(contrast)
    
    def detect_artifacts(self, image: np.ndarray) -> Dict[str, float]:
        """
        检测图像伪影（包括块效应、振铃效应等）
        
        Args:
            image: 输入图像
            
        Returns:
            伪影检测结果字典
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 检测块效应（JPEG压缩伪影）
        block_score = self._detect_blocking_artifacts(gray)
        
        # 检测振铃效应
        ringing_score = self._detect_ringing_artifacts(gray)
        
        return {
            'blocking_artifacts': block_score,
            'ringing_artifacts': ringing_score,
            'overall_artifacts': (block_score + ringing_score) / 2
        }
    
    def _detect_blocking_artifacts(self, gray: np.ndarray) -> float:
        """检测块效应"""
        # 计算水平和垂直方向的差分
        h_diff = np.abs(np.diff(gray, axis=1))
        v_diff = np.abs(np.diff(gray, axis=0))
        
        # 检测8x8块边界的不连续性
        block_size = 8
        h_block_diff = []
        v_block_diff = []
        
        for i in range(block_size, gray.shape[1], block_size):
            if i < h_diff.shape[1]:
                h_block_diff.append(np.mean(h_diff[:, i-1:i+1]))
        
        for i in range(block_size, gray.shape[0], block_size):
            if i < v_diff.shape[0]:
                v_block_diff.append(np.mean(v_diff[i-1:i+1, :]))
        
        if h_block_diff and v_block_diff:
            block_score = (np.mean(h_block_diff) + np.mean(v_block_diff)) / 2
        else:
            block_score = 0.0
            
        return float(block_score)
    
    def _detect_ringing_artifacts(self, gray: np.ndarray) -> float:
        """检测振铃效应"""
        # 使用Canny边缘检测
        edges = cv2.Canny(gray, 50, 150)
        
        # 膨胀边缘
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=2)
        
        # 计算边缘附近的标准差（振铃通常在边缘附近）
        edge_region = gray[dilated_edges > 0]
        if len(edge_region) > 0:
            ringing_score = np.std(edge_region)
        else:
            ringing_score = 0.0
            
        return float(ringing_score)
    
    def detect_color_distortion(self, image: np.ndarray) -> Dict[str, float]:
        """
        检测色彩失真
        
        Args:
            image: 输入图像
            
        Returns:
            色彩失真指标
        """
        # 转换到HSV空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 分析饱和度分布
        saturation = hsv[:, :, 1]
        saturation_mean = np.mean(saturation)
        saturation_std = np.std(saturation)
        
        # 分析亮度分布
        value = hsv[:, :, 2]
        brightness_mean = np.mean(value)
        brightness_std = np.std(value)
        
        # 检测过曝和欠曝
        overexposed_ratio = np.sum(value > 240) / value.size
        underexposed_ratio = np.sum(value < 15) / value.size
        
        return {
            'saturation_mean': float(saturation_mean),
            'saturation_std': float(saturation_std),
            'brightness_mean': float(brightness_mean),
            'brightness_std': float(brightness_std),
            'overexposed_ratio': float(overexposed_ratio),
            'underexposed_ratio': float(underexposed_ratio)
        }
    
    def evaluate(self, image: np.ndarray) -> Dict[str, any]:
        """
        综合评估图像质量
        
        Args:
            image: 输入图像
            
        Returns:
            完整的质量评估结果
        """
        results = {
            'blur_score': self.detect_blur(image),
            'noise_level': self.detect_noise(image),
            'sharpness': self.calculate_sharpness(image),
            'contrast': self.calculate_contrast(image),
            'artifacts': self.detect_artifacts(image),
            'color_distortion': self.detect_color_distortion(image)
        }
        
        # 计算综合质量分数（0-1，越高越好）
        # 归一化各项指标
        blur_normalized = min(results['blur_score'] / 500.0, 1.0)  # 假设500为良好阈值
        noise_normalized = max(0, 1.0 - results['noise_level'] / 50.0)  # 噪声越低越好
        sharpness_normalized = min(results['sharpness'] / 100.0, 1.0)
        contrast_normalized = min(results['contrast'] / 80.0, 1.0)
        artifacts_normalized = max(0, 1.0 - results['artifacts']['overall_artifacts'] / 50.0)
        
        # 加权平均
        quality_score = (
            blur_normalized * 0.25 +
            noise_normalized * 0.20 +
            sharpness_normalized * 0.20 +
            contrast_normalized * 0.15 +
            artifacts_normalized * 0.20
        )
        
        results['quality_score'] = float(quality_score)
        results['is_good_quality'] = quality_score > self.config.get('quality_thresholds', {}).get('overall_pass_score', 0.6)
        
        return results

