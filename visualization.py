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

# ���Ե�����ӻ����������ʧ�����ṩ��������
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    # ����matplotlib��������
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("����: matplotlibδ��װ��ͳ��ͼ���ܲ�����")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("����: seabornδ��װ���߼�ͼ���ܲ�����")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("����: plotlyδ��װ������ʽ�Ǳ�幦�ܲ�����")

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
    print("����: Pillowδ��װ��ͼ�����ܿ�������")

class VideoVisualizer:
    """��Ƶ�����������ӻ���"""
    
    def __init__(self, config: dict):
        """
        ��ʼ�����ӻ���
        
        Args:
            config: �����ֵ�
        """
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.output_dir = config.get('output', {}).get('output_dir', './results')
        self.viz_dir = config.get('output', {}).get('visualization_dir', './results/visualizations')
        
        # �������ӻ����Ŀ¼
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # ���ӻ�����
        self.pose_skeleton_color = tuple(self.viz_config.get('pose_skeleton_color', [0, 255, 0]))
        self.anomaly_color = tuple(self.viz_config.get('anomaly_color', [0, 0, 255]))
        self.confidence_color = tuple(self.viz_config.get('confidence_color', [255, 0, 0]))
        self.line_thickness = self.viz_config.get('line_thickness', 2)
        self.font_scale = self.viz_config.get('font_scale', 0.6)
        self.show_confidence = self.viz_config.get('show_confidence', True)
        self.show_keypoint_names = self.viz_config.get('show_keypoint_names', False)
        
        # ��̬�Ǽ����Ӷ��� (YOLO-Pose 17���ؼ���)
        self.skeleton_connections = [
            # ͷ��
            (0, 1), (0, 2), (1, 3), (2, 4),  # ����-�۾�-����
            # ����
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # ���-�ⲿ-����
            # ����
            (5, 11), (6, 12), (11, 12),  # ���-�β�
            (11, 13), (12, 14), (13, 15), (14, 16)  # �β�-ϥ��-����
        ]
        
        # �ؼ�������
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
    
    def draw_pose_skeleton(self, image: np.ndarray, keypoints: np.ndarray, 
                          confidence: float = None, anomalies: List[int] = None) -> np.ndarray:
        """
        ��ͼ���ϻ�����̬�Ǽ�
        
        Args:
            image: ����ͼ��
            keypoints: �ؼ������� (17, 2)
            confidence: ������Ŷ�
            anomalies: �쳣�ؼ��������б�
            
        Returns:
            �����˹Ǽܵ�ͼ��
        """
        vis_image = image.copy()
        
        # ���ƹǼ�����
        for connection in self.skeleton_connections:
            pt1_idx, pt2_idx = connection
            if (pt1_idx < len(keypoints) and pt2_idx < len(keypoints) and
                keypoints[pt1_idx][0] > 0 and keypoints[pt1_idx][1] > 0 and
                keypoints[pt2_idx][0] > 0 and keypoints[pt2_idx][1] > 0):
                
                pt1 = tuple(map(int, keypoints[pt1_idx]))
                pt2 = tuple(map(int, keypoints[pt2_idx]))
                
                # ����Ƿ�Ϊ�쳣����
                is_anomaly = (anomalies and (pt1_idx in anomalies or pt2_idx in anomalies))
                color = self.anomaly_color if is_anomaly else self.pose_skeleton_color
                thickness = self.line_thickness + 1 if is_anomaly else self.line_thickness
                
                cv2.line(vis_image, pt1, pt2, color, thickness)
        
        # ���ƹؼ���
        for i, (x, y) in enumerate(keypoints):
            if x > 0 and y > 0:
                pt = (int(x), int(y))
                
                # ����Ƿ�Ϊ�쳣�ؼ���
                is_anomaly = anomalies and i in anomalies
                color = self.anomaly_color if is_anomaly else self.pose_skeleton_color
                radius = 4 if is_anomaly else 3
                
                cv2.circle(vis_image, pt, radius, color, -1)
                
                # ��ʾ�ؼ�������
                if self.show_keypoint_names and i < len(self.keypoint_names):
                    cv2.putText(vis_image, self.keypoint_names[i], 
                              (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                              self.font_scale, color, 1)
        
        # ��ʾ���Ŷ�
        if self.show_confidence and confidence is not None:
            conf_text = f"Confidence: {confidence:.2f}"
            cv2.putText(vis_image, conf_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       self.font_scale, self.confidence_color, 2)
        
        return vis_image
    
    def create_comparison_image(self, original: np.ndarray, detected: np.ndarray, 
                              title: str = "Pose Detection Comparison") -> np.ndarray:
        """
        ����ԭʼͼ���������ĶԱ�ͼ
        
        Args:
            original: ԭʼͼ��
            detected: �����ͼ��
            title: �Ա�ͼ����
            
        Returns:
            �Ա�ͼ��
        """
        # ȷ������ͼ���Сһ��
        h1, w1 = original.shape[:2]
        h2, w2 = detected.shape[:2]
        
        if h1 != h2 or w1 != w2:
            detected = cv2.resize(detected, (w1, h1))
        
        # ˮƽƴ��
        comparison = np.hstack([original, detected])
        
        # ��ӷָ���
        line_x = w1
        cv2.line(comparison, (line_x, 0), (line_x, h1), (255, 255, 255), 2)
        
        # ��ӱ���
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255, 255, 255), 2)
        cv2.putText(comparison, "Detected", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (255, 255, 255), 2)
        
        return comparison
    
    def save_pose_visualization(self, frames: List[np.ndarray], results: List[Dict], 
                               video_name: str) -> List[str]:
        """
        ������̬���ӻ����
        
        Args:
            frames: ��Ƶ֡�б�
            results: ������б�
            video_name: ��Ƶ����
            
        Returns:
            �����ͼ���ļ�·���б�
        """
        saved_paths = []
        
        for i, (frame, result) in enumerate(zip(frames, results)):
            if result.get('pose_detected', False):
                # ������̬�Ǽ�
                keypoints = result.get('keypoints', np.array([]))
                confidence = result.get('confidence', 0.0)
                anomalies = result.get('anomalies', [])
                
                if len(keypoints) > 0:
                    vis_frame = self.draw_pose_skeleton(frame, keypoints, confidence, anomalies)
                    
                    # ����ͼ��
                    filename = f"{video_name}_pose_frame_{i:03d}.jpg"
                    filepath = os.path.join(self.viz_dir, filename)
                    cv2.imwrite(filepath, vis_frame)
                    saved_paths.append(filepath)
        
        return saved_paths
    
    def save_anomaly_visualization(self, frames: List[np.ndarray], results: List[Dict], 
                                 video_name: str) -> List[str]:
        """
        �����쳣�����ӻ����
        
        Args:
            frames: ��Ƶ֡�б�
            results: ������б�
            video_name: ��Ƶ����
            
        Returns:
            �����ͼ���ļ�·���б�
        """
        saved_paths = []
        
        for i, (frame, result) in enumerate(zip(frames, results)):
            if result.get('anomaly_count', 0) > 0:
                # �����쳣����
                vis_frame = frame.copy()
                
                # ���쳣������ƺ�ɫ�߿�
                h, w = frame.shape[:2]
                cv2.rectangle(vis_frame, (0, 0), (w, h), self.anomaly_color, 5)
                
                # ����쳣��Ϣ�ı�
                anomaly_text = f"Anomalies: {result.get('anomaly_count', 0)}"
                cv2.putText(vis_frame, anomaly_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.0, self.anomaly_color, 2)
                
                # ����ͼ��
                filename = f"{video_name}_anomaly_frame_{i:03d}.jpg"
                filepath = os.path.join(self.viz_dir, filename)
                cv2.imwrite(filepath, vis_frame)
                saved_paths.append(filepath)
        
        return saved_paths
    
    def create_statistics_charts(self, results: List[Dict], video_name: str) -> List[str]:
        """
        ����ͳ��ͼ��
        
        Args:
            results: ������Ƶ���������
            video_name: ��Ƶ����
            
        Returns:
            �����ͼ���ļ�·���б�
        """
        if not MATPLOTLIB_AVAILABLE:
            print("����: matplotlib�����ã�����ͳ��ͼ������")
            return []
        
        saved_paths = []
        
        try:
            # ��ȡ����
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
            
            # ������ͼ
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Video Quality Analysis: {video_name}', fontsize=16)
            
            # 1. ����ʷֲ�
            axes[0, 0].hist(detection_rates, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Detection Rate Distribution')
            axes[0, 0].set_xlabel('Detection Rate')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. ���Ŷȷֲ�
            axes[0, 1].hist(confidences, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Confidence Distribution')
            axes[0, 1].set_xlabel('Average Confidence')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 3. �쳣�����ֲ�
            axes[1, 0].hist(anomaly_counts, bins=10, alpha=0.7, color='lightcoral', edgecolor='black')
            axes[1, 0].set_title('Anomaly Count Distribution')
            axes[1, 0].set_xlabel('Average Anomalies')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # 4. �ۺ���������
            axes[1, 1].plot(overall_scores, marker='o', linewidth=2, markersize=6)
            axes[1, 1].set_title('Overall Score Trend')
            axes[1, 1].set_xlabel('Frame Index')
            axes[1, 1].set_ylabel('Overall Score')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # ����ͼ��
            chart_path = os.path.join(self.viz_dir, f"{video_name}_statistics.png")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            saved_paths.append(chart_path)
            
        except Exception as e:
            print(f"����ͳ��ͼ��ʱ����: {e}")
        
        return saved_paths
    
    def create_interactive_dashboard(self, results: List[Dict], video_name: str) -> str:
        """
        ��������ʽ�Ǳ��
        
        Args:
            results: ������Ƶ���������
            video_name: ��Ƶ����
            
        Returns:
            �����HTML�ļ�·��
        """
        if not PLOTLY_AVAILABLE:
            print("����: plotly�����ã���������ʽ�Ǳ������")
            return ""
        
        try:
            # ��ȡ����
            frames = list(range(len(results)))
            detection_rates = [r.get('pose_quality', {}).get('detection_rate', 0) for r in results]
            confidences = [r.get('pose_quality', {}).get('avg_confidence', 0) for r in results]
            anomaly_counts = [r.get('pose_quality', {}).get('avg_anomalies', 0) for r in results]
            overall_scores = [r.get('overall_assessment', {}).get('overall_score', 0) for r in results]
            
            # ������ͼ
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Detection Rate', 'Confidence', 'Anomaly Count', 'Overall Score'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # ��ӹ켣
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
            
            # ���²���
            fig.update_layout(
                title=f'Interactive Dashboard: {video_name}',
                height=800,
                showlegend=True
            )
            
            # ����HTML�ļ�
            dashboard_path = os.path.join(self.viz_dir, f"{video_name}_dashboard.html")
            fig.write_html(dashboard_path)
            
            return dashboard_path
            
        except Exception as e:
            print(f"��������ʽ�Ǳ��ʱ����: {e}")
            return ""
    
    def visualize_video_results(self, video_path: str, frames: List[np.ndarray], 
                              results: Dict) -> Dict[str, List[str]]:
        """
        Ϊ������Ƶ���������Ŀ��ӻ����
        
        Args:
            video_path: ��Ƶ�ļ�·��
            frames: ��Ƶ֡�б�
            results: �������
            
        Returns:
            ������ļ�·���ֵ�
        """
        video_name = Path(video_path).stem
        saved_files = {
            'pose_images': [],
            'anomaly_images': [],
            'comparison_images': [],
            'statistics_charts': [],
            'dashboard': []
        }
        
        # ����Ƿ����ÿ��ӻ�
        if not self.viz_config.get('enable_visualization', False):
            return saved_files
        
        # ��̬���ӻ�
        if self.viz_config.get('save_pose_images', False):
            frame_results = results.get('pose_quality', {}).get('frame_results', [])
            saved_files['pose_images'] = self.save_pose_visualization(frames, frame_results, video_name)
        
        # �쳣���ӻ�
        if self.viz_config.get('save_anomaly_images', False):
            frame_results = results.get('pose_quality', {}).get('frame_results', [])
            saved_files['anomaly_images'] = self.save_anomaly_visualization(frames, frame_results, video_name)
        
        # �Ա�ͼ��
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
        
        # ͳ��ͼ��
        if self.viz_config.get('save_statistics_charts', False):
            frame_results = results.get('pose_quality', {}).get('frame_results', [])
            saved_files['statistics_charts'] = self.create_statistics_charts(frame_results, video_name)
        
        # ����ʽ�Ǳ��
        if self.viz_config.get('save_statistics_charts', False):
            frame_results = results.get('pose_quality', {}).get('frame_results', [])
            dashboard_path = self.create_interactive_dashboard(frame_results, video_name)
            saved_files['dashboard'].append(dashboard_path)
        
        return saved_files
