"""
使用YOLO增强版姿态检测示例

展示如何使用更robust的YOLO检测器
"""

import sys
sys.path.append('..')

import cv2
import numpy as np


def example_mediapipe_vs_yolo():
    """对比MediaPipe和YOLO在部分身体检测中的表现"""
    
    print("="*70)
    print("姿态检测方法对比：MediaPipe改进版 vs YOLO")
    print("="*70)
    
    # 测试图像路径
    test_image_path = 'test_image.jpg'  # 替换为你的测试图像
    
    try:
        image = cv2.imread(test_image_path)
        if image is None:
            print(f"??  无法读取图像: {test_image_path}")
            print("提示: 请准备一张包含人物的测试图像")
            return
    except Exception as e:
        print(f"读取图像出错: {e}")
        return
    
    # 方法1: MediaPipe改进版（自适应）
    print("\n【方法1: MediaPipe改进版（自适应）】")
    print("-"*70)
    
    try:
        from detectors import PoseAnomalyDetector
        
        detector_mp = PoseAnomalyDetector()
        result_mp = detector_mp.evaluate(image)
        
        if result_mp['pose_detected']:
            print(f"? 检测成功")
            print(f"  检测模式: {result_mp['detection_mode']}")
            print(f"  可见部位: {result_mp['visible_body_parts']}")
            print(f"  质量分数: {result_mp['pose_quality_score']:.3f}")
            print(f"  异常数量: {result_mp['anomaly_count']}")
            if result_mp['anomaly_count'] > 0:
                print(f"  检测到的异常:")
                for anomaly in result_mp['all_anomalies'][:3]:
                    print(f"    - {anomaly}")
        else:
            print(f"? 未检测到人体")
    
    except Exception as e:
        print(f"MediaPipe检测出错: {e}")
    
    # 方法2: YOLO版本
    print("\n【方法2: YOLO增强版】")
    print("-"*70)
    
    try:
        from detectors import PoseAnomalyDetectorYOLO, YOLO_AVAILABLE
        
        if not YOLO_AVAILABLE:
            print("? YOLO不可用")
            print("安装方法: pip install ultralytics")
            return
        
        detector_yolo = PoseAnomalyDetectorYOLO(model_path='yolov8n-pose.pt')
        result_yolo = detector_yolo.evaluate(image)
        
        if result_yolo['pose_detected']:
            print(f"? 检测成功")
            print(f"  检测模式: {result_yolo['detection_mode']}")
            print(f"  可见部位: {result_yolo['visible_body_parts']}")
            print(f"  检测置信度: {result_yolo['detection_confidence']:.3f}")
            print(f"  质量分数: {result_yolo['pose_quality_score']:.3f}")
            print(f"  异常数量: {result_yolo['anomaly_count']}")
            if result_yolo['anomaly_count'] > 0:
                print(f"  检测到的异常:")
                for anomaly in result_yolo['all_anomalies'][:3]:
                    print(f"    - {anomaly}")
            if result_yolo['bbox'] is not None:
                bbox = result_yolo['bbox']
                print(f"  人物框: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]")
        else:
            print(f"? 未检测到人体")
    
    except ImportError:
        print("? YOLO不可用")
        print("安装方法: pip install ultralytics")
    except Exception as e:
        print(f"YOLO检测出错: {e}")
    
    print("\n" + "="*70)


def example_auto_select():
    """自动选择最佳检测器"""
    
    print("\n【智能检测器选择】")
    print("="*70)
    
    try:
        from detectors import get_pose_detector, YOLO_AVAILABLE
        
        # 尝试使用YOLO，如果不可用则自动回退到MediaPipe
        detector = get_pose_detector(use_yolo=True)
        
        if YOLO_AVAILABLE:
            print("? 使用YOLO检测器（推荐）")
        else:
            print("??  YOLO不可用，使用MediaPipe改进版")
        
        # 使用统一接口
        test_image = cv2.imread('test_image.jpg')
        if test_image is not None:
            result = detector.evaluate(test_image)
            print(f"检测方法: {result.get('method', 'MediaPipe')}")
            print(f"检测结果: {result['pose_detected']}")
    
    except Exception as e:
        print(f"出错: {e}")


def example_batch_with_yolo():
    """批量处理时使用YOLO"""
    
    print("\n【批量处理示例】")
    print("="*70)
    
    from detectors import get_pose_detector, YOLO_AVAILABLE
    import glob
    
    # 获取所有测试图像
    images = glob.glob('test_images/*.jpg')
    
    if not images:
        print("未找到测试图像")
        print("提示: 在 test_images/ 文件夹中放入测试图片")
        return
    
    # 选择检测器
    use_yolo = YOLO_AVAILABLE
    detector = get_pose_detector(use_yolo=use_yolo)
    
    print(f"处理 {len(images)} 张图像...")
    print(f"使用检测器: {'YOLO' if use_yolo else 'MediaPipe'}")
    print()
    
    results = []
    for img_path in images:
        image = cv2.imread(img_path)
        result = detector.evaluate(image)
        results.append({
            'path': img_path,
            'detected': result['pose_detected'],
            'mode': result.get('detection_mode', 'unknown'),
            'anomaly_count': result.get('anomaly_count', 0)
        })
    
    # 统计
    detected_count = sum(1 for r in results if r['detected'])
    anomaly_count = sum(1 for r in results if r['anomaly_count'] > 0)
    
    print(f"\n统计结果:")
    print(f"  检出率: {detected_count}/{len(images)} ({detected_count/len(images)*100:.1f}%)")
    print(f"  异常数: {anomaly_count} ({anomaly_count/len(images)*100:.1f}%)")
    
    # 显示各检测模式分布
    from collections import Counter
    mode_count = Counter(r['mode'] for r in results if r['detected'])
    print(f"\n检测模式分布:")
    for mode, count in mode_count.items():
        print(f"  {mode}: {count}")


def example_partial_body_detection():
    """测试部分身体检测能力"""
    
    print("\n【部分身体检测测试】")
    print("="*70)
    
    from detectors import PoseAnomalyDetector
    
    # 创建模拟的部分身体图像（仅供演示）
    print("\n测试场景1: 仅上半身")
    print("-"*70)
    print("? 改进后系统能够:")
    print("  - 自动识别检测模式为 'upper_only'")
    print("  - 只检查上半身相关比例（头肩比等）")
    print("  - 不会因为缺少下半身而报错")
    print("  - 根据可见部位调整评分标准")
    
    print("\n测试场景2: 仅下半身")
    print("-"*70)
    print("? 改进后系统能够:")
    print("  - 自动识别检测模式为 'lower_only'")
    print("  - 只检查下半身相关比例（大小腿比等）")
    print("  - 提供针对性的异常检测")
    
    print("\n测试场景3: 严重遮挡")
    print("-"*70)
    print("? YOLO版本优势:")
    print("  - 更强的遮挡处理能力")
    print("  - 提供边界框信息")
    print("  - 多实例检测支持")
    print("  - 更高的检测置信度")


def main():
    print("\n" + "?"*35)
    print("AIGC视频姿态检测 - YOLO增强版本使用示例")
    print("?"*35 + "\n")
    
    # 示例1: 对比测试
    example_mediapipe_vs_yolo()
    
    # 示例2: 自动选择
    example_auto_select()
    
    # 示例3: 批量处理
    # example_batch_with_yolo()  # 取消注释以运行
    
    # 示例4: 部分身体检测说明
    example_partial_body_detection()
    
    print("\n" + "="*70)
    print("? 使用建议:")
    print("="*70)
    print("1. 生产环境: 使用YOLO版本（pip install ultralytics）")
    print("2. 快速测试: 使用MediaPipe改进版")
    print("3. 混合策略: MediaPipe初筛 + YOLO精筛")
    print("\n详细文档: 参考 POSE_DETECTION_UPGRADE.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

