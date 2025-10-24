"""
AIGC视频质量筛选主模块
整合所有检测器，提供统一的视频质量评估接口
"""

import cv2
import numpy as np
import yaml
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from detectors import ImageQualityDetector, PoseAnomalyDetector, TemporalConsistencyDetector


class VideoQualityFilter:
    """AIGC视频质量筛选器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化视频质量筛选器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        
        # 初始化各个检测器
        self.image_quality_detector = ImageQualityDetector(self.config)
        self.pose_detector = PoseAnomalyDetector(self.config)
        self.temporal_detector = TemporalConsistencyDetector(self.config)
        
        # 创建输出目录
        output_dir = self.config.get('output', {}).get('output_dir', './results')
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'quality_thresholds': {
                'blur_threshold': 100.0,
                'noise_threshold': 30.0,
                'contrast_threshold': 30.0,
                'overall_pass_score': 0.6
            },
            'detection_modules': {
                'enable_blur_detection': True,
                'enable_noise_detection': True,
                'enable_pose_detection': True,
                'enable_temporal_consistency': True
            },
            'video_processing': {
                'sample_fps': 10,
                'max_frames': 30,
                'resize_height': 720,
                'resize_width': 1280
            },
            'output': {
                'save_report': True,
                'save_visualization': False,
                'output_dir': './results'
            }
        }
    
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            (帧列表, 视频信息)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'width': width,
            'height': height,
            'duration': duration
        }
        
        # 计算采样间隔
        sample_fps = self.config['video_processing']['sample_fps']
        frame_interval = max(1, int(fps / sample_fps)) if fps > 0 else 1
        max_frames = self.config['video_processing']['max_frames']
        
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 按间隔采样
            if frame_idx % frame_interval == 0:
                # 调整大小（可选）
                target_height = self.config['video_processing'].get('resize_height')
                target_width = self.config['video_processing'].get('resize_width')
                
                if target_height and target_width:
                    frame = cv2.resize(frame, (target_width, target_height))
                
                frames.append(frame)
                
                # 限制最大帧数
                if len(frames) >= max_frames:
                    break
            
            frame_idx += 1
        
        cap.release()
        
        return frames, video_info
    
    def evaluate_image_quality(self, frames: List[np.ndarray]) -> Dict:
        """
        评估图像质量（对多帧取平均）
        
        Args:
            frames: 视频帧列表
            
        Returns:
            图像质量评估结果
        """
        if not self.config['detection_modules']['enable_blur_detection']:
            return {'skipped': True}
        
        # 对部分帧进行评估（避免计算量过大）
        sample_indices = np.linspace(0, len(frames) - 1, min(10, len(frames)), dtype=int)
        
        all_results = []
        for idx in sample_indices:
            result = self.image_quality_detector.evaluate(frames[idx])
            all_results.append(result)
        
        # 计算平均值
        avg_result = {
            'blur_score': np.mean([r['blur_score'] for r in all_results]),
            'noise_level': np.mean([r['noise_level'] for r in all_results]),
            'sharpness': np.mean([r['sharpness'] for r in all_results]),
            'contrast': np.mean([r['contrast'] for r in all_results]),
            'quality_score': np.mean([r['quality_score'] for r in all_results]),
            'sampled_frames': len(sample_indices)
        }
        
        # 判断是否通过
        thresholds = self.config['quality_thresholds']
        avg_result['passed'] = (
            avg_result['blur_score'] >= thresholds.get('blur_threshold', 100.0) and
            avg_result['noise_level'] <= thresholds.get('noise_threshold', 30.0)
        )
        
        return avg_result
    
    def evaluate_pose_quality(self, frames: List[np.ndarray]) -> Dict:
        """
        评估人体姿态质量
        
        Args:
            frames: 视频帧列表
            
        Returns:
            姿态质量评估结果
        """
        if not self.config['detection_modules']['enable_pose_detection']:
            return {'skipped': True}
        
        # 对部分帧进行评估
        sample_indices = np.linspace(0, len(frames) - 1, min(10, len(frames)), dtype=int)
        
        all_results = []
        frames_with_person = 0
        frames_with_anomaly = 0
        
        for idx in sample_indices:
            result = self.pose_detector.evaluate(frames[idx])
            all_results.append(result)
            
            if result.get('has_person', False):
                frames_with_person += 1
                if result.get('anomaly_count', 0) > 0:
                    frames_with_anomaly += 1
        
        # 汇总结果
        summary = {
            'sampled_frames': len(sample_indices),
            'frames_with_person': frames_with_person,
            'frames_with_anomaly': frames_with_anomaly,
            'person_detection_rate': frames_with_person / len(sample_indices) if sample_indices.size > 0 else 0,
            'anomaly_rate': frames_with_anomaly / frames_with_person if frames_with_person > 0 else 0
        }
        
        # 如果检测到人物
        if frames_with_person > 0:
            # 收集所有异常
            all_anomalies = []
            for result in all_results:
                if result.get('has_person', False):
                    all_anomalies.extend(result.get('all_anomalies', []))
            
            # 计算平均姿态质量分数
            pose_scores = [r.get('pose_quality_score', 1.0) for r in all_results if r.get('has_person', False)]
            avg_pose_score = np.mean(pose_scores) if pose_scores else 0.0
            
            summary['all_detected_anomalies'] = all_anomalies
            summary['unique_anomaly_types'] = len(set(all_anomalies))
            summary['avg_pose_quality_score'] = float(avg_pose_score)
            summary['passed'] = summary['anomaly_rate'] < 0.3  # 30%以下的帧有异常则通过
        else:
            # 没有检测到人物，视为通过（可能是非人物视频）
            summary['passed'] = True
            summary['note'] = '未检测到人物，跳过姿态检查'
        
        return summary
    
    def evaluate_temporal_consistency(self, frames: List[np.ndarray]) -> Dict:
        """
        评估时序一致性
        
        Args:
            frames: 视频帧列表
            
        Returns:
            时序一致性评估结果
        """
        if not self.config['detection_modules']['enable_temporal_consistency']:
            return {'skipped': True}
        
        result = self.temporal_detector.evaluate(frames)
        
        # 添加通过判断
        threshold = self.config['quality_thresholds'].get('temporal_consistency_threshold', 0.7)
        result['passed'] = result.get('temporal_consistency_score', 0) >= threshold
        
        return result
    
    def evaluate_video(self, video_path: str, verbose: bool = True) -> Dict:
        """
        评估单个视频的质量
        
        Args:
            video_path: 视频文件路径
            verbose: 是否打印详细信息
            
        Returns:
            完整的视频质量评估结果
        """
        if verbose:
            print(f"\n正在评估视频: {video_path}")
        
        # 提取帧
        try:
            frames, video_info = self.extract_frames(video_path)
        except Exception as e:
            return {
                'video_path': video_path,
                'error': f"提取帧失败: {str(e)}",
                'passed': False
            }
        
        if verbose:
            print(f"提取了 {len(frames)} 帧，视频信息: {video_info}")
        
        # 各项评估
        results = {
            'video_path': video_path,
            'video_info': video_info,
            'extracted_frames': len(frames)
        }
        
        if verbose:
            print("1/3 评估图像质量...")
        results['image_quality'] = self.evaluate_image_quality(frames)
        
        if verbose:
            print("2/3 评估人体姿态...")
        results['pose_quality'] = self.evaluate_pose_quality(frames)
        
        if verbose:
            print("3/3 评估时序一致性...")
        results['temporal_consistency'] = self.evaluate_temporal_consistency(frames)
        
        # 计算综合评分
        results['overall_assessment'] = self._calculate_overall_score(results)
        
        if verbose:
            print(f"\n综合评估结果:")
            print(f"  - 图像质量分数: {results['image_quality'].get('quality_score', 0):.3f}")
            print(f"  - 姿态质量: {'通过' if results['pose_quality'].get('passed', False) else '未通过'}")
            print(f"  - 时序一致性分数: {results['temporal_consistency'].get('temporal_consistency_score', 0):.3f}")
            print(f"  - 总体分数: {results['overall_assessment']['overall_score']:.3f}")
            print(f"  - 筛选结果: {'? 通过' if results['overall_assessment']['passed'] else '? 未通过'}")
        
        # 保存报告
        if self.config['output']['save_report']:
            self._save_report(results)
        
        return results
    
    def _calculate_overall_score(self, results: Dict) -> Dict:
        """
        计算综合评分
        
        Args:
            results: 各项评估结果
            
        Returns:
            综合评估结果
        """
        # 提取各项分数
        image_score = results['image_quality'].get('quality_score', 0)
        pose_score = results['pose_quality'].get('avg_pose_quality_score', 1.0)
        temporal_score = results['temporal_consistency'].get('temporal_consistency_score', 0)
        
        # 加权计算总分
        weights = {
            'image': 0.35,
            'pose': 0.30,
            'temporal': 0.35
        }
        
        overall_score = (
            image_score * weights['image'] +
            pose_score * weights['pose'] +
            temporal_score * weights['temporal']
        )
        
        # 收集所有问题
        issues = []
        
        if not results['image_quality'].get('passed', True):
            issues.append('图像质量不达标')
        
        if not results['pose_quality'].get('passed', True):
            anomaly_rate = results['pose_quality'].get('anomaly_rate', 0)
            if anomaly_rate > 0:
                issues.append(f'人体姿态异常（异常率: {anomaly_rate:.1%}）')
        
        if not results['temporal_consistency'].get('passed', True):
            issues.append('时序一致性不足')
        
        # 检查是否有闪烁
        if results['temporal_consistency'].get('flicker_detection', {}).get('has_flicker', False):
            issues.append('检测到视频闪烁')
        
        # 检查是否有抖动
        if results['temporal_consistency'].get('jitter_detection', {}).get('has_jitter', False):
            issues.append('检测到视频抖动')
        
        # 判断是否通过
        pass_threshold = self.config['quality_thresholds']['overall_pass_score']
        passed = overall_score >= pass_threshold and len(issues) == 0
        
        return {
            'overall_score': float(overall_score),
            'passed': passed,
            'issues': issues,
            'recommendation': '该视频质量良好，建议保留' if passed else f'该视频存在问题，建议筛除。问题: {"; ".join(issues)}'
        }
    
    def _save_report(self, results: Dict):
        """保存评估报告"""
        video_name = Path(results['video_path']).stem
        report_path = os.path.join(self.output_dir, f"{video_name}_report.json")
        
        # 转换numpy类型为Python原生类型
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        results_serializable = convert_types(results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"报告已保存至: {report_path}")
    
    def batch_evaluate(self, video_dir: str, output_summary: str = None) -> List[Dict]:
        """
        批量评估视频文件夹中的所有视频
        
        Args:
            video_dir: 视频文件夹路径
            output_summary: 汇总报告保存路径
            
        Returns:
            所有视频的评估结果列表
        """
        # 支持的视频格式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        # 获取所有视频文件
        video_files = []
        for ext in video_extensions:
            # 搜索小写扩展名
            lower_files = list(Path(video_dir).glob(f"*{ext}"))
            # 搜索大写扩展名
            upper_files = list(Path(video_dir).glob(f"*{ext.upper()}"))
            
            # 合并并去重（避免在Windows系统中重复计算）
            all_files = lower_files + upper_files
            # 使用set去重，然后转换回list
            unique_files = list(set(all_files))
            video_files.extend(unique_files)
        
        if not video_files:
            print(f"在 {video_dir} 中未找到视频文件")
            return []
        
        print(f"找到 {len(video_files)} 个视频文件，开始批量评估...")
        
        all_results = []
        passed_count = 0
        failed_count = 0
        
        for video_path in tqdm(video_files, desc="评估进度"):
            result = self.evaluate_video(str(video_path), verbose=False)
            all_results.append(result)
            
            if result['overall_assessment']['passed']:
                passed_count += 1
            else:
                failed_count += 1
        
        # 打印汇总
        print(f"\n批量评估完成!")
        print(f"总计: {len(video_files)} 个视频")
        print(f"通过: {passed_count} 个 ({passed_count/len(video_files)*100:.1f}%)")
        print(f"未通过: {failed_count} 个 ({failed_count/len(video_files)*100:.1f}%)")
        
        # 保存汇总报告
        if output_summary:
            summary = {
                'total_videos': len(video_files),
                'passed': passed_count,
                'failed': failed_count,
                'pass_rate': passed_count / len(video_files) if video_files else 0,
                'results': all_results
            }
            
            # 转换numpy类型为Python原生类型
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(v) for v in obj]
                else:
                    return obj
            
            summary_serializable = convert_types(summary)
            
            with open(output_summary, 'w', encoding='utf-8') as f:
                json.dump(summary_serializable, f, indent=2, ensure_ascii=False)
            
            print(f"汇总报告已保存至: {output_summary}")
        
        return all_results


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AIGC视频质量筛选工具')
    parser.add_argument('input', help='输入视频文件或文件夹路径')
    parser.add_argument('--config', default='config.yaml', help='配置文件路径')
    parser.add_argument('--batch', action='store_true', help='批量处理模式')
    parser.add_argument('--output', help='输出报告路径')
    
    args = parser.parse_args()
    
    # 初始化筛选器
    filter = VideoQualityFilter(args.config)
    
    if args.batch:
        # 批量模式
        output_path = args.output or os.path.join(filter.output_dir, 'batch_summary.json')
        filter.batch_evaluate(args.input, output_path)
    else:
        # 单文件模式
        filter.evaluate_video(args.input)


if __name__ == '__main__':
    main()

