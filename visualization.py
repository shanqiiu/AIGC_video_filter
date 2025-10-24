#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AIGC Video Quality Filter Visualization Module
Provides pose visualization, detection result display, anomaly annotation, etc.
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# 尝试导入可视化依赖，如果失败则提供降级功能
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    # 设置matplotlib中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: matplotlib未安装，统计图表功能不可用")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("警告: seaborn未安装，高级图表功能不可用")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("警告: plotly未安装，交互式仪表板功能不可用")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("警告: Pillow未安装，图像处理功能可能受限")

class VideoVisualizer:
    """视频质量评估可视化器"""
    
    def __init__(self, config: dict):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.output_dir = config.get('output', {}).get('output_dir', './results')
        self.viz_dir = config.get('output', {}).get('visualization_dir', './results/visualizations')
        
        # 创建可视化输出目录
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # 可视化参数
        self.pose_skeleton_color = tuple(self.viz_config.get('pose_skeleton_color', [0, 255, 0]))
        self.anomaly_color = tuple(self.viz_config.get('anomaly_color', [0, 0, 255]))
        self.confidence_color = tuple(self.viz_config.get('confidence_color', [255, 0, 0]))
        self.line_thickness = self.viz_config.get('line_thickness', 2)
        self.font_scale = self.viz_config.get('font_scale', 0.6)
        self.show_confidence = self.viz_config.get('show_confidence', True)
        self.show_keypoint_names = self.viz_config.get('show_keypoint_names', False)
        
        # 姿态骨架连接定义 (YOLO-Pose 17个关键点)
        self.skeleton_connections = [
            # 头部
            (0, 1), (0, 2), (1, 3), (2, 4),  # 鼻子-眼睛-耳朵
            # 上身
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # 肩膀-肘部-手腕
            # 下身
            (5, 11), (6, 12), (11, 12),  # 肩膀-臀部
            (11, 13), (12, 14), (13, 15), (14, 16)  # 臀部-膝盖-脚踝
        ]
        
        # 关键点名称
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def draw_pose_skeleton(self, image: np.ndarray, keypoints: np.ndarray, 
                          confidence: float = None, anomalies: List[int] = None) -> np.ndarray:
        """
        在图像上绘制姿态骨架
        
        Args:
            image: 输入图像
            keypoints: 关键点坐标 (17, 2)
            confidence: 检测置信度
            anomalies: 异常关键点索引列表
            
        Returns:
            绘制了骨架的图像
        """
        vis_image = image.copy()
        
        # 绘制骨架连接
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                keypoints[pt1_idx][0] > 0 and keypoints[pt1_idx][1] > 0 and
                keypoints[pt2_idx][0] > 0 and keypoints[pt2_idx][1] > 0):
                
                pt1 = tuple(map(int, keypoints[pt1_idx]))
                pt2 = tuple(map(int, keypoints[pt2_idx]))
                
                # 检查是否为异常连接
                is_anomaly = (anomalies and (pt1_idx in anomalies or pt2_idx in anomalies))
                color = self.anomaly_color if is_anomaly else self.pose_skeleton_color
                thickness = self.line_thickness + 1 if is_anomaly else self.line_thickness
                
                cv2.line(vis_image, pt1, pt2, color, thickness)
        
        # 绘制关键点
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:
                pt = (int(x), int(y))
                
                # 检查是否为异常关键点
                is_anomaly = anomalies and i in anomalies
                color = self.anomaly_color if is_anomaly else self.pose_skeleton_color
                radius = 4 if is_anomaly else 3
                
                cv2.circle(vis_image, pt, radius, color, -1)
                
                # 显示关键点名称
                if self.show_keypoint_names and i < len(self.keypoint_names):
                    cv2.putText(vis_image, self.keypoint_names[i], 
                              (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                              self.font_scale, color, 1)
        
        # 显示置信度
        if self.show_confidence and confidence is not None:
            conf_text = f"Confidence: {confidence:.2f}"
            cv2.putText(vis_image, conf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       self.font_scale, self.confidence_color, 2)
        
        return vis_image
    
    def create_comparison_image(self, original: np.ndarray, detected: np.ndarray, 
                              title: str = "Pose Detection Comparison") -> np.ndarray:
        """
        创建原始图像与检测结果的对比图
        
        Args:
            original: 原始图像
            detected: 检测结果图像
            title: 对比图标题
            
        Returns:
            对比图像
        """
        # 确保两个图像大小一致
        h1, w1 = original.shape[:2]
        h2, w2 = detected.shape[:2]
        
        if h1 != h2 or w1 != w2:
            detected = cv2.resize(detected, (w1, h1))
        
        # 水平拼接
        comparison = np.hstack([original, detected])
        
        # 添加分割线
        line_x = w1
        cv2.line(comparison, (line_x, 0), (line_x, h1), (255, 255, 255), 2)
        
        # 添加标题
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255, 255, 255), 2)
        cv2.putText(comparison, "Detected", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255, 255, 255), 2)
        
        return comparison
    
    def save_pose_visualization(self, frames: List[np.ndarray], results: List[Dict], 
                               video_name: str) -> List[str]:
        """
        保存姿态可视化结果
        
        Args:
            frames: 视频帧列表
            results: 检测结果列表
            video_name: 视频名称
            
        Returns:
            保存的图像文件路径列表
        """
        saved_paths = []
        
        for i, (frame, result) in enumerate(zip(frames, results)):
            if result.get('pose_detected', False):
                # 绘制姿态骨架
                keypoints = result.get('keypoints', np.array([]))
                confidence = result.get('confidence', 0.0)
                anomalies = result.get('anomalies', [])
                
                if len(keypoints) > 0:
                    vis_frame = self.draw_pose_skeleton(frame, keypoints, confidence, anomalies)
                    
                    # 保存图像
                    filename = f"{video_name}_pose_frame_{i:03d}.jpg"
                    filepath = os.path.join(self.viz_dir, filename)
                    cv2.imwrite(filepath, vis_frame)
                    saved_paths.append(filepath)
        
        return saved_paths
    
    def save_anomaly_visualization(self, frames: List[np.ndarray], results: List[Dict], 
                                 video_name: str) -> List[str]:
        """
        保存异常检测可视化结果
        
        Args:
            frames: 视频帧列表
            results: 检测结果列表
            video_name: 视频名称
            
        Returns:
            保存的图像文件路径列表
        """
        saved_paths = []
        
        for i, (frame, result) in enumerate(zip(frames, results)):
            if result.get('anomaly_count', 0) > 0:
                # 高亮异常区域
                vis_frame = frame.copy()
                
                # 在异常区域绘制红色边框
                h, w = frame.shape[:2]
                cv2.rectangle(vis_frame, (0, 0), (w, h), self.anomaly_color, 5)
                
                # 添加异常信息文本
                anomaly_text = f"Anomalies: {result.get('anomaly_count', 0)}"
                cv2.putText(vis_frame, anomaly_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, self.anomaly_color, 2)
                
                # 保存图像
                filename = f"{video_name}_anomaly_frame_{i:03d}.jpg"
                filepath = os.path.join(self.viz_dir, filename)
                cv2.imwrite(filepath, vis_frame)
                saved_paths.append(filepath)
        
        return saved_paths
    
    def create_statistics_charts(self, results: List[Dict], video_name: str) -> List[str]:
        """
        创建统计图表
        
        Args:
            results: 所有视频的评估结果
            video_name: 视频名称
            
        Returns:
            保存的图表文件路径列表
        """
        if not MATPLOTLIB_AVAILABLE:
            print("警告: matplotlib不可用，跳过统计图表生成")
            return []
        
        saved_paths = []
        
        try:
            # 提取数据
            detection_rates = []
            confidences = []
            anomaly_counts = []
            overall_scores = []
            
            for result in results:
                pose_quality = result.get('pose_quality', {})
                overall = result.get('overall_assessment', {})
                
                detection_rates.append(pose_quality.get('detection_rate', 0))
                confidences.append(pose_quality.get('avg_confidence', 0))
                anomaly_counts.append(pose_quality.get('avg_anomalies', 0))
                overall_scores.append(overall.get('overall_score', 0))
            
            # 创建子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Video Quality Analysis: {video_name}', fontsize=16)
            
            # 1. 检测率分布
            axes[0, 0].hist(detection_rates, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Detection Rate Distribution')
            axes[0, 0].set_xlabel('Detection Rate')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 置信度分布
            axes[0, 1].hist(confidences, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Confidence Distribution')
            axes[0, 1].set_xlabel('Average Confidence')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. 异常数量分布
            axes[1, 0].hist(anomaly_counts, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, 0].set_title('Anomaly Count Distribution')
            axes[1, 0].set_xlabel('Average Anomalies')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. 综合评分趋势
            axes[1, 1].plot(overall_scores, marker='o', linewidth=2, markersize=6)
            axes[1, 1].set_title('Overall Score Trend')
            axes[1, 1].set_xlabel('Frame Index')
            axes[1, 1].set_ylabel('Overall Score')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            chart_path = os.path.join(self.viz_dir, f"{video_name}_statistics.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_paths.append(chart_path)
            
        except Exception as e:
            print(f"创建统计图表时出错: {e}")
        
        return saved_paths
    
    def create_interactive_dashboard(self, results: List[Dict], video_name: str) -> str:
        """
        创建交互式仪表板
        
        Args:
            results: 所有视频的评估结果
            video_name: 视频名称
            
        Returns:
            保存的HTML文件路径
        """
        if not PLOTLY_AVAILABLE:
            print("警告: plotly不可用，跳过交互式仪表板生成")
            return ""
        
        try:
            # 提取数据
            frames = list(range(len(results)))
            detection_rates = [r.get('pose_quality', {}).get('detection_rate', 0) for r in results]
            confidences = [r.get('pose_quality', {}).get('avg_confidence', 0) for r in results]
            anomaly_counts = [r.get('pose_quality', {}).get('avg_anomalies', 0) for r in results]
            overall_scores = [r.get('overall_assessment', {}).get('overall_score', 0) for r in results]
            
            # 创建子图
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Detection Rate', 'Confidence', 'Anomaly Count', 'Overall Score'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # 添加轨迹
            fig.add_trace(
                go.Scatter(x=frames, y=detection_rates, mode='lines+markers', name='Detection Rate'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=frames, y=confidences, mode='lines+markers', name='Confidence'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=frames, y=anomaly_counts, mode='lines+markers', name='Anomaly Count'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=frames, y=overall_scores, mode='lines+markers', name='Overall Score'),
                row=2, col=2
            )
            
            # 更新布局
            fig.update_layout(
                title=f'Interactive Dashboard: {video_name}',
                height=800,
                showlegend=True
            )
            
            # 保存HTML文件
            dashboard_path = os.path.join(self.viz_dir, f"{video_name}_dashboard.html")
            fig.write_html(dashboard_path)
            
            return dashboard_path
            
        except Exception as e:
            print(f"创建交互式仪表板时出错: {e}")
            return ""
    
    def visualize_video_results(self, video_path: str, frames: List[np.ndarray], 
                              results: Dict) -> Dict[str, List[str]]:
        """
        为单个视频创建完整的可视化结果
        
        Args:
            video_path: 视频文件路径
            frames: 视频帧列表
            results: 评估结果
            
        Returns:
            保存的文件路径字典
        """
        video_name = Path(video_path).stem
        saved_files = {
            'pose_images': [],
            'anomaly_images': [],
            'comparison_images': [],
            'statistics_charts': [],
            'dashboard': []
        }
        
        # 检查是否启用可视化
        if not self.viz_config.get('enable_visualization', False):
            return saved_files
        
        # 姿态可视化
        if self.viz_config.get('save_pose_images', False):
            frame_results = results.get('pose_quality', {}).get('frame_results', [])
            saved_files['pose_images'] = self.save_pose_visualization(frames, frame_results, video_name)
        
        # 异常可视化
        if self.viz_config.get('save_anomaly_images', False):
            frame_results = results.get('pose_quality', {}).get('frame_results', [])
            saved_files['anomaly_images'] = self.save_anomaly_visualization(frames, frame_results, video_name)
        
        # 对比图像
        if self.viz_config.get('save_comparison_images', False):
            frame_results = results.get('pose_quality', {}).get('frame_results', [])
            for i, (frame, result) in enumerate(zip(frames, frame_results)):
                if result.get('pose_detected', False):
                    keypoints = result.get('keypoints', np.array([]))
                    if len(keypoints) > 0:
                        detected_frame = self.draw_pose_skeleton(frame, keypoints, 
                                                              result.get('confidence', 0))
                        comparison = self.create_comparison_image(frame, detected_frame)
                        
                        filename = f"{video_name}_comparison_frame_{i:03d}.jpg"
                        filepath = os.path.join(self.viz_dir, filename)
                        cv2.imwrite(filepath, comparison)
                        saved_files['comparison_images'].append(filepath)
        
        # 统计图表
        if self.viz_config.get('save_statistics_charts', False):
            frame_results = results.get('pose_quality', {}).get('frame_results', [])
            saved_files['statistics_charts'] = self.create_statistics_charts(frame_results, video_name)
        
        # 交互式仪表板
        if self.viz_config.get('save_statistics_charts', False):
            frame_results = results.get('pose_quality', {}).get('frame_results', [])
            dashboard_path = self.create_interactive_dashboard(frame_results, video_name)
            saved_files['dashboard'].append(dashboard_path)
        
        return saved_files
