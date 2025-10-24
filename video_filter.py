"""
AIGC视频质量筛选主模块
仅使用YOLO进行人体姿态检测
"""

import cv2
import numpy as np
import yaml
import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from detectors import PoseAnomalyDetectorYOLO, YOLO_AVAILABLE
from visualization import VideoVisualizer


class VideoQualityFilter:
    """AIGC视频质量筛选器（仅使用YOLO）"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化视频质量筛选器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        
        # 初始化YOLO检测器
        if not YOLO_AVAILABLE:
            raise ImportError("需要安装ultralytics: pip install ultralytics")
        
        self.pose_detector = PoseAnomalyDetectorYOLO(self.config)
        
        # 创建输出目录
        output_dir = self.config.get('output', {}).get('output_dir', './results')
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # 初始化可视化器
        self.visualizer = VideoVisualizer(self.config)
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            except UnicodeDecodeError:
                # 尝试其他编码
                try:
                    with open(config_path, 'r', encoding='gbk') as f:
                        return yaml.safe_load(f)
                except UnicodeDecodeError:
                    with open(config_path, 'r', encoding='latin-1') as f:
                        return yaml.safe_load(f)
        else:
            print(f"配置文件 {config_path} 不存在，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """获取默认配置"""
        return {
            'quality_thresholds': {
                'pose_confidence_threshold': 0.5,
                'overall_pass_score': 0.6
            },
            'video_processing': {
                'sample_fps': 10,
                'max_frames': 30,
                'resize_height': 720,
                'resize_width': 1280
            },
            'output': {
                'save_report': True,
                'output_dir': './results'
            }
        }
    
    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """
        从视频中提取帧
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            帧列表和视频信息
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
        sample_fps = self.config.get('video_processing', {}).get('sample_fps', 10)
        max_frames = self.config.get('video_processing', {}).get('max_frames', 30)
        
        if fps > 0:
            frame_interval = max(1, int(fps / sample_fps))
        else:
            frame_interval = 1
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                # 调整图像大小
                resize_height = self.config.get('video_processing', {}).get('resize_height', 720)
                resize_width = self.config.get('video_processing', {}).get('resize_width', 1280)
                
                if resize_height > 0 and resize_width > 0:
                    frame = cv2.resize(frame, (resize_width, resize_height))
                
                frames.append(frame)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        return frames, video_info
    
    def evaluate_pose_quality(self, frames: List[np.ndarray]) -> Dict:
        """
        评估人体姿态质量
        
        Args:
            frames: 视频帧列表
            
        Returns:
            姿态质量评估结果
        """
        if not frames:
            return {
                'pose_detected': False,
                'message': '没有可处理的帧',
                'passed': False
            }
        
        # 对每一帧进行姿态检测
        frame_results = []
        total_anomalies = 0
        total_confidence = 0
        detected_frames = 0
        
        for frame in frames:
            result = self.pose_detector.evaluate(frame)
            frame_results.append(result)
            
            if result['pose_detected']:
                detected_frames += 1
                total_anomalies += result['anomaly_count']
                total_confidence += result['avg_confidence']
        
        # 计算统计信息
        detection_rate = detected_frames / len(frames) if frames else 0
        avg_anomalies = total_anomalies / detected_frames if detected_frames > 0 else 0
        avg_confidence = total_confidence / detected_frames if detected_frames > 0 else 0
        
        # 判断是否通过
        confidence_threshold = self.config.get('quality_thresholds', {}).get('pose_confidence_threshold', 0.5)
        passed = detection_rate > 0.5 and avg_confidence > confidence_threshold and avg_anomalies < 2
        
        return {
            'pose_detected': detected_frames > 0,
            'detection_rate': detection_rate,
            'avg_anomalies': avg_anomalies,
            'avg_confidence': avg_confidence,
            'detected_frames': detected_frames,
            'total_frames': len(frames),
            'passed': passed,
            'frame_results': frame_results
        }
    
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
        
        # 姿态评估
        results = {
            'video_path': video_path,
            'video_info': video_info,
            'extracted_frames': len(frames)
        }
        
        if verbose:
            print("评估人体姿态...")
        results['pose_quality'] = self.evaluate_pose_quality(frames)
        
        # 计算综合评分
        results['overall_assessment'] = self._calculate_overall_score(results)
        
        # 生成可视化结果
        if self.config.get('visualization', {}).get('enable_visualization', False):
            try:
                visualization_files = self.visualizer.visualize_video_results(video_path, frames, results)
                results['visualization_files'] = visualization_files
                
                if verbose:
                    print(f"\n可视化结果已保存:")
                    for viz_type, files in visualization_files.items():
                        if files:
                            print(f"  {viz_type}: {len(files)} 个文件")
            except Exception as e:
                if verbose:
                    print(f"可视化生成失败: {e}")
                results['visualization_error'] = str(e)
        
        if verbose:
            print(f"\n综合评估结果:")
            print(f"  姿态检测: {'通过' if results['pose_quality']['passed'] else '未通过'}")
            print(f"  检测率: {results['pose_quality']['detection_rate']:.2f}")
            print(f"  平均置信度: {results['pose_quality']['avg_confidence']:.2f}")
            print(f"  平均异常数: {results['pose_quality']['avg_anomalies']:.1f}")
            print(f"  综合评分: {results['overall_assessment']['overall_score']:.3f}")
            print(f"  最终结果: {'通过' if results['overall_assessment']['passed'] else '未通过'}")
        
        return results
    
    def _calculate_overall_score(self, results: Dict) -> Dict:
        """
        计算综合评分
        
        Args:
            results: 各项评估结果
            
        Returns:
            综合评估结果
        """
        pose_quality = results.get('pose_quality', {})
        
        # 基础分数
        detection_score = pose_quality.get('detection_rate', 0.0)
        confidence_score = pose_quality.get('avg_confidence', 0.0)
        anomaly_penalty = min(pose_quality.get('avg_anomalies', 0) / 5.0, 1.0)
        
        # 综合评分
        overall_score = (detection_score * 0.4 + confidence_score * 0.4 + (1 - anomaly_penalty) * 0.2)
        
        # 判断是否通过
        threshold = self.config.get('quality_thresholds', {}).get('overall_pass_score', 0.6)
        passed = overall_score >= threshold and pose_quality.get('passed', False)
        
        return {
            'overall_score': overall_score,
            'detection_score': detection_score,
            'confidence_score': confidence_score,
            'anomaly_penalty': anomaly_penalty,
            'passed': passed,
            'threshold': threshold
        }
    
    def batch_evaluate(self, video_dir: str, output_summary: str = None) -> List[Dict]:
        """
        批量评估视频
        
        Args:
            video_dir: 视频目录路径
            output_summary: 输出摘要文件路径
            
        Returns:
            所有视频的评估结果
        """
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise ValueError(f"视频目录不存在: {video_dir}")
        
        # 支持的视频格式
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        
        # 使用更精确的文件扫描逻辑，避免重复检测
        for ext in video_extensions:
            # 搜索小写扩展名
            video_files.extend(video_dir.glob(f'*{ext}'))
            # 只有当扩展名不是小写时才搜索大写版本
            if ext != ext.upper():
                video_files.extend(video_dir.glob(f'*{ext.upper()}'))
        
        # 转换为字符串路径进行去重，然后转换回Path对象
        unique_paths = set(str(path) for path in video_files)
        video_files = sorted([Path(path) for path in unique_paths])
        
        if not video_files:
            print(f"在目录 {video_dir} 中未找到视频文件")
            return []
        
        print(f"找到 {len(video_files)} 个视频文件:")
        for i, video_file in enumerate(video_files, 1):
            print(f"  {i}. {video_file.name}")
        
        all_results = []
        passed_count = 0
        failed_count = 0
        
        for video_file in tqdm(video_files, desc="处理视频"):
            try:
                result = self.evaluate_video(str(video_file), verbose=False)
                all_results.append(result)
                
                if result.get('overall_assessment', {}).get('passed', False):
                    passed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                error_result = {
                    'video_path': str(video_file),
                    'error': str(e),
                    'passed': False
                }
                all_results.append(error_result)
                failed_count += 1
        
        # 生成摘要
        summary = {
            'total_videos': len(video_files),
            'processed': len(all_results),
            'passed': passed_count,
            'failed': failed_count,
            'pass_rate': passed_count / len(video_files) if video_files else 0,
            'results': all_results
        }
        
        # 保存摘要
        if output_summary:
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
            
            print(f"\n摘要已保存到: {output_summary}")
        
        print(f"\n批量评估完成:")
        print(f"  总视频数: {len(video_files)}")
        print(f"  通过数: {passed_count}")
        print(f"  未通过数: {failed_count}")
        print(f"  通过率: {passed_count/len(video_files)*100:.1f}%")
        
        return all_results


def main():
    """命令行入口函数"""
    parser = argparse.ArgumentParser(description='AIGC视频质量筛选器（YOLO版本）')
    parser.add_argument('input', help='输入视频文件或目录路径')
    parser.add_argument('-o', '--output', help='输出摘要文件路径（可选）')
    parser.add_argument('-c', '--config', default='config.yaml', help='配置文件路径（默认：config.yaml）')
    parser.add_argument('-v', '--verbose', action='store_true', help='显示详细信息')
    parser.add_argument('--enable-viz', action='store_true', help='启用可视化功能')
    parser.add_argument('--save-pose', action='store_true', help='保存姿态检测图像')
    parser.add_argument('--save-anomaly', action='store_true', help='保存异常检测图像')
    parser.add_argument('--save-comparison', action='store_true', help='保存对比图像')
    parser.add_argument('--save-charts', action='store_true', help='保存统计图表')
    
    args = parser.parse_args()
    
    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"错误: 输入路径不存在: {input_path}")
        return 1
    
    try:
        # 初始化配置路径
        config_path = args.config
        
        # 处理可视化参数
        if args.enable_viz or args.save_pose or args.save_anomaly or args.save_comparison or args.save_charts:
            # 动态更新配置以启用可视化
            import yaml
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
            else:
                config = {}
            
            # 确保可视化配置存在
            if 'visualization' not in config:
                config['visualization'] = {}
            
            # 根据命令行参数更新可视化设置
            if args.enable_viz:
                config['visualization']['enable_visualization'] = True
            if args.save_pose:
                config['visualization']['save_pose_images'] = True
            if args.save_anomaly:
                config['visualization']['save_anomaly_images'] = True
            if args.save_comparison:
                config['visualization']['save_comparison_images'] = True
            if args.save_charts:
                config['visualization']['save_statistics_charts'] = True
            
            # 保存临时配置文件
            temp_config_path = 'temp_config_with_viz.yaml'
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            config_path = temp_config_path
        
        # 初始化筛选器
        print("初始化AIGC视频质量筛选器（YOLO版本）...")
        filter = VideoQualityFilter(config_path=config_path)
        
        if input_path.is_file():
            # 单个视频文件
            print(f"\n开始评估视频: {input_path}")
            print("="*60)
            
            result = filter.evaluate_video(str(input_path), verbose=args.verbose)
            
            # 显示结果
            if args.verbose:
                print("\n" + "="*60)
                print("评估结果详情:")
                print("="*60)
                
                # 视频基本信息
                print("\n【视频信息】")
                video_info = result['video_info']
                print(f"  分辨率: {video_info['width']}x{video_info['height']}")
                print(f"  帧率: {video_info['fps']:.2f} fps")
                print(f"  时长: {video_info['duration']:.2f} 秒")
                print(f"  总帧数: {video_info['total_frames']}")
                
                # 人体姿态
                print("\n【人体姿态检测】")
                pose_quality = result['pose_quality']
                print(f"  检测状态: {'检测到人体' if pose_quality['pose_detected'] else '未检测到人体'}")
                print(f"  检测率: {pose_quality['detection_rate']:.2f}")
                print(f"  平均置信度: {pose_quality['avg_confidence']:.2f}")
                print(f"  平均异常数: {pose_quality['avg_anomalies']:.1f}")
                print(f"  检测帧数: {pose_quality['detected_frames']}/{pose_quality['total_frames']}")
                print(f"  状态: {'✓ 通过' if pose_quality['passed'] else '✗ 未通过'}")
                
                # 综合评估
                print("\n【综合评估】")
                overall = result['overall_assessment']
                print(f"  综合评分: {overall['overall_score']:.3f}")
                print(f"  检测分数: {overall['detection_score']:.3f}")
                print(f"  置信度分数: {overall['confidence_score']:.3f}")
                print(f"  异常惩罚: {overall['anomaly_penalty']:.3f}")
                print(f"  通过阈值: {overall['threshold']:.3f}")
                print(f"  最终结果: {'✓ 通过' if overall['passed'] else '✗ 未通过'}")
            else:
                # 简化输出
                pose_quality = result['pose_quality']
                overall = result['overall_assessment']
                status = "✓ 通过" if overall['passed'] else "✗ 未通过"
                print(f"结果: {status} (评分: {overall['overall_score']:.3f}, 检测率: {pose_quality['detection_rate']:.2f})")
            
            return 0 if result['overall_assessment']['passed'] else 1
            
        else:
            # 批量处理目录
            print(f"\n开始批量评估视频目录: {input_path}")
            print("="*60)
            
            results = filter.batch_evaluate(str(input_path), args.output)
            
            if not results:
                print("没有找到视频文件或处理失败")
                return 1
            
            # 显示统计信息
            total_videos = len(results)
            passed_videos = sum(1 for r in results if r.get('overall_assessment', {}).get('passed', False))
            failed_videos = total_videos - passed_videos
            
            print(f"\n批量评估完成:")
            print(f"  总视频数: {total_videos}")
            print(f"  通过数: {passed_videos}")
            print(f"  未通过数: {failed_videos}")
            print(f"  通过率: {passed_videos/total_videos*100:.1f}%")
            
            if args.output:
                print(f"  详细结果已保存到: {args.output}")
            
            return 0 if passed_videos > 0 else 1
            
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保已安装所需依赖: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"运行错误: {e}")
        return 1
    finally:
        # 清理临时配置文件
        if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)