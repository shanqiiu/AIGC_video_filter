"""
AIGC��Ƶ����ɸѡ��ģ��
��ʹ��YOLO����������̬���
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


class VideoQualityFilter:
    """AIGC��Ƶ����ɸѡ������ʹ��YOLO��"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        ��ʼ����Ƶ����ɸѡ��
        
        Args:
            config_path: �����ļ�·��
        """
        self.config = self._load_config(config_path)
        
        # ��ʼ��YOLO�����
        if not YOLO_AVAILABLE:
            raise ImportError("��Ҫ��װultralytics: pip install ultralytics")
        
        self.pose_detector = PoseAnomalyDetectorYOLO(self.config)
        
        # �������Ŀ¼
        output_dir = self.config.get('output', {}).get('output_dir', './results')
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
    
    def _load_config(self, config_path: str) -> dict:
        """���������ļ�"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            print(f"�����ļ� {config_path} �����ڣ�ʹ��Ĭ������")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """��ȡĬ������"""
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
        ����Ƶ����ȡ֡
        
        Args:
            video_path: ��Ƶ�ļ�·��
            
        Returns:
            ֡�б����Ƶ��Ϣ
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"�޷�����Ƶ�ļ�: {video_path}")
        
        # ��ȡ��Ƶ��Ϣ
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
        
        # ����������
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
                # ����ͼ���С
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
        ����������̬����
        
        Args:
            frames: ��Ƶ֡�б�
            
        Returns:
            ��̬�����������
        """
        if not frames:
            return {
                'pose_detected': False,
                'message': 'û�пɴ����֡',
                'passed': False
            }
        
        # ��ÿһ֡������̬���
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
        
        # ����ͳ����Ϣ
        detection_rate = detected_frames / len(frames) if frames else 0
        avg_anomalies = total_anomalies / detected_frames if detected_frames > 0 else 0
        avg_confidence = total_confidence / detected_frames if detected_frames > 0 else 0
        
        # �ж��Ƿ�ͨ��
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
        ����������Ƶ������
        
        Args:
            video_path: ��Ƶ�ļ�·��
            verbose: �Ƿ��ӡ��ϸ��Ϣ
            
        Returns:
            ��������Ƶ�����������
        """
        if verbose:
            print(f"\n����������Ƶ: {video_path}")
        
        # ��ȡ֡
        try:
            frames, video_info = self.extract_frames(video_path)
        except Exception as e:
            return {
                'video_path': video_path,
                'error': f"��ȡ֡ʧ��: {str(e)}",
                'passed': False
            }
        
        if verbose:
            print(f"��ȡ�� {len(frames)} ֡����Ƶ��Ϣ: {video_info}")
        
        # ��̬����
        results = {
            'video_path': video_path,
            'video_info': video_info,
            'extracted_frames': len(frames)
        }
        
        if verbose:
            print("����������̬...")
        results['pose_quality'] = self.evaluate_pose_quality(frames)
        
        # �����ۺ�����
        results['overall_assessment'] = self._calculate_overall_score(results)
        
        if verbose:
            print(f"\n�ۺ��������:")
            print(f"  ��̬���: {'ͨ��' if results['pose_quality']['passed'] else 'δͨ��'}")
            print(f"  �����: {results['pose_quality']['detection_rate']:.2f}")
            print(f"  ƽ�����Ŷ�: {results['pose_quality']['avg_confidence']:.2f}")
            print(f"  ƽ���쳣��: {results['pose_quality']['avg_anomalies']:.1f}")
            print(f"  �ۺ�����: {results['overall_assessment']['overall_score']:.3f}")
            print(f"  ���ս��: {'ͨ��' if results['overall_assessment']['passed'] else 'δͨ��'}")
        
        return results
    
    def _calculate_overall_score(self, results: Dict) -> Dict:
        """
        �����ۺ�����
        
        Args:
            results: �����������
            
        Returns:
            �ۺ��������
        """
        pose_quality = results.get('pose_quality', {})
        
        # ��������
        detection_score = pose_quality.get('detection_rate', 0.0)
        confidence_score = pose_quality.get('avg_confidence', 0.0)
        anomaly_penalty = min(pose_quality.get('avg_anomalies', 0) / 5.0, 1.0)
        
        # �ۺ�����
        overall_score = (detection_score * 0.4 + confidence_score * 0.4 + (1 - anomaly_penalty) * 0.2)
        
        # �ж��Ƿ�ͨ��
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
        ����������Ƶ
        
        Args:
            video_dir: ��ƵĿ¼·��
            output_summary: ���ժҪ�ļ�·��
            
        Returns:
            ������Ƶ���������
        """
        video_dir = Path(video_dir)
        if not video_dir.exists():
            raise ValueError(f"��ƵĿ¼������: {video_dir}")
        
        # ֧�ֵ���Ƶ��ʽ
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(video_dir.glob(f'*{ext}'))
            video_files.extend(video_dir.glob(f'*{ext.upper()}'))
        
        if not video_files:
            print(f"��Ŀ¼ {video_dir} ��δ�ҵ���Ƶ�ļ�")
            return []
        
        print(f"�ҵ� {len(video_files)} ����Ƶ�ļ�")
        
        all_results = []
        passed_count = 0
        failed_count = 0
        
        for video_file in tqdm(video_files, desc="������Ƶ"):
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
        
        # ����ժҪ
        summary = {
            'total_videos': len(video_files),
            'processed': len(all_results),
            'passed': passed_count,
            'failed': failed_count,
            'pass_rate': passed_count / len(video_files) if video_files else 0,
            'results': all_results
        }
        
        # ����ժҪ
        if output_summary:
            # ת��numpy����ΪPythonԭ������
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
            
            print(f"\nժҪ�ѱ��浽: {output_summary}")
        
        print(f"\n�����������:")
        print(f"  ����Ƶ��: {len(video_files)}")
        print(f"  ͨ����: {passed_count}")
        print(f"  δͨ����: {failed_count}")
        print(f"  ͨ����: {passed_count/len(video_files)*100:.1f}%")
        
        return all_results


def main():
    """��������ں���"""
    parser = argparse.ArgumentParser(description='AIGC��Ƶ����ɸѡ����YOLO�汾��')
    parser.add_argument('input', help='������Ƶ�ļ���Ŀ¼·��')
    parser.add_argument('-o', '--output', help='���ժҪ�ļ�·������ѡ��')
    parser.add_argument('-c', '--config', default='config.yaml', help='�����ļ�·����Ĭ�ϣ�config.yaml��')
    parser.add_argument('-v', '--verbose', action='store_true', help='��ʾ��ϸ��Ϣ')
    
    args = parser.parse_args()
    
    # �������·��
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"����: ����·��������: {input_path}")
        return 1
    
    try:
        # ��ʼ��ɸѡ��
        print("��ʼ��AIGC��Ƶ����ɸѡ����YOLO�汾��...")
        filter = VideoQualityFilter(config_path=args.config)
        
        if input_path.is_file():
            # ������Ƶ�ļ�
            print(f"\n��ʼ������Ƶ: {input_path}")
            print("="*60)
            
            result = filter.evaluate_video(str(input_path), verbose=args.verbose)
            
            # ��ʾ���
            if args.verbose:
                print("\n" + "="*60)
                print("�����������:")
                print("="*60)
                
                # ��Ƶ������Ϣ
                print("\n����Ƶ��Ϣ��")
                video_info = result['video_info']
                print(f"  �ֱ���: {video_info['width']}x{video_info['height']}")
                print(f"  ֡��: {video_info['fps']:.2f} fps")
                print(f"  ʱ��: {video_info['duration']:.2f} ��")
                print(f"  ��֡��: {video_info['total_frames']}")
                
                # ������̬
                print("\n��������̬��⡿")
                pose_quality = result['pose_quality']
                print(f"  ���״̬: {'��⵽����' if pose_quality['pose_detected'] else 'δ��⵽����'}")
                print(f"  �����: {pose_quality['detection_rate']:.2f}")
                print(f"  ƽ�����Ŷ�: {pose_quality['avg_confidence']:.2f}")
                print(f"  ƽ���쳣��: {pose_quality['avg_anomalies']:.1f}")
                print(f"  ���֡��: {pose_quality['detected_frames']}/{pose_quality['total_frames']}")
                print(f"  ״̬: {'? ͨ��' if pose_quality['passed'] else '? δͨ��'}")
                
                # �ۺ�����
                print("\n���ۺ�������")
                overall = result['overall_assessment']
                print(f"  �ۺ�����: {overall['overall_score']:.3f}")
                print(f"  ������: {overall['detection_score']:.3f}")
                print(f"  ���Ŷȷ���: {overall['confidence_score']:.3f}")
                print(f"  �쳣�ͷ�: {overall['anomaly_penalty']:.3f}")
                print(f"  ͨ����ֵ: {overall['threshold']:.3f}")
                print(f"  ���ս��: {'? ͨ��' if overall['passed'] else '? δͨ��'}")
            else:
                # �����
                pose_quality = result['pose_quality']
                overall = result['overall_assessment']
                status = "? ͨ��" if overall['passed'] else "? δͨ��"
                print(f"���: {status} (����: {overall['overall_score']:.3f}, �����: {pose_quality['detection_rate']:.2f})")
            
            return 0 if result['overall_assessment']['passed'] else 1
            
        else:
            # ��������Ŀ¼
            print(f"\n��ʼ����������ƵĿ¼: {input_path}")
            print("="*60)
            
            results = filter.batch_evaluate(str(input_path), args.output)
            
            if not results:
                print("û���ҵ���Ƶ�ļ�����ʧ��")
                return 1
            
            # ��ʾͳ����Ϣ
            total_videos = len(results)
            passed_videos = sum(1 for r in results if r.get('overall_assessment', {}).get('passed', False))
            failed_videos = total_videos - passed_videos
            
            print(f"\n�����������:")
            print(f"  ����Ƶ��: {total_videos}")
            print(f"  ͨ����: {passed_videos}")
            print(f"  δͨ����: {failed_videos}")
            print(f"  ͨ����: {passed_videos/total_videos*100:.1f}%")
            
            if args.output:
                print(f"  ��ϸ����ѱ��浽: {args.output}")
            
            return 0 if passed_videos > 0 else 1
            
    except ImportError as e:
        print(f"�������: {e}")
        print("��ȷ���Ѱ�װ��������: pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"���д���: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)