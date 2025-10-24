"""
自定义配置示例：根据特定需求调整检测参数
"""

import sys
sys.path.append('..')

from video_filter import VideoQualityFilter


def example_strict_mode():
    """严格模式：用于高质量视频筛选"""
    print("\n【示例1：严格模式】")
    print("="*60)
    
    # 创建严格配置
    strict_config = {
        'quality_thresholds': {
            'blur_threshold': 150.0,      # 提高清晰度要求
            'noise_threshold': 20.0,      # 降低噪声容忍度
            'overall_pass_score': 0.75    # 提高总体分数要求
        },
        'detection_modules': {
            'enable_blur_detection': True,
            'enable_noise_detection': True,
            'enable_pose_detection': True,
            'enable_temporal_consistency': True
        }
    }
    
    filter = VideoQualityFilter()
    filter.config['quality_thresholds'].update(strict_config['quality_thresholds'])
    
    print("配置: 严格模式")
    print(f"  - 模糊度阈值: {strict_config['quality_thresholds']['blur_threshold']}")
    print(f"  - 噪声阈值: {strict_config['quality_thresholds']['noise_threshold']}")
    print(f"  - 通过分数: {strict_config['quality_thresholds']['overall_pass_score']}")
    
    # 评估视频
    video_path = 'test_video.mp4'
    result = filter.evaluate_video(video_path, verbose=False)
    
    print(f"\n结果: {'? 通过' if result['overall_assessment']['passed'] else '? 未通过'}")
    print(f"分数: {result['overall_assessment']['overall_score']:.3f}")


def example_fast_mode():
    """快速模式：用于大批量初筛"""
    print("\n【示例2：快速模式】")
    print("="*60)
    
    # 创建快速配置
    fast_config = {
        'video_processing': {
            'sample_fps': 5,              # 降低采样率
            'max_frames': 15,             # 减少处理帧数
            'resize_height': 480,         # 降低分辨率
            'resize_width': 854
        },
        'detection_modules': {
            'enable_blur_detection': True,
            'enable_noise_detection': True,
            'enable_pose_detection': False,  # 关闭姿态检测以提速
            'enable_temporal_consistency': True
        }
    }
    
    filter = VideoQualityFilter()
    filter.config['video_processing'].update(fast_config['video_processing'])
    filter.config['detection_modules'].update(fast_config['detection_modules'])
    
    print("配置: 快速模式")
    print(f"  - 采样帧率: {fast_config['video_processing']['sample_fps']} fps")
    print(f"  - 最大帧数: {fast_config['video_processing']['max_frames']}")
    print(f"  - 处理分辨率: {fast_config['video_processing']['resize_width']}x{fast_config['video_processing']['resize_height']}")
    print(f"  - 姿态检测: {'开启' if fast_config['detection_modules']['enable_pose_detection'] else '关闭'}")
    
    # 评估视频
    video_path = 'test_video.mp4'
    result = filter.evaluate_video(video_path, verbose=False)
    
    print(f"\n结果: {'? 通过' if result['overall_assessment']['passed'] else '? 未通过'}")
    print(f"分数: {result['overall_assessment']['overall_score']:.3f}")


def example_pose_focused():
    """姿态聚焦模式：专注于人物视频的姿态检测"""
    print("\n【示例3：姿态聚焦模式】")
    print("="*60)
    
    filter = VideoQualityFilter()
    
    # 调整配置，只关注姿态相关问题
    filter.config['detection_modules']['enable_pose_detection'] = True
    filter.config['detection_modules']['enable_blur_detection'] = False
    filter.config['detection_modules']['enable_noise_detection'] = False
    
    print("配置: 姿态聚焦模式")
    print("  - 只启用姿态检测模块")
    print("  - 关闭图像质量检测")
    
    # 评估视频
    video_path = 'test_video.mp4'
    result = filter.evaluate_video(video_path, verbose=False)
    
    # 显示姿态相关结果
    pose_quality = result['pose_quality']
    print(f"\n姿态检测结果:")
    if pose_quality.get('has_person', False):
        print(f"  人物检出率: {pose_quality.get('person_detection_rate', 0):.1%}")
        print(f"  异常检出率: {pose_quality.get('anomaly_rate', 0):.1%}")
        print(f"  姿态质量分数: {pose_quality.get('avg_pose_quality_score', 0):.3f}")
        
        if pose_quality.get('all_detected_anomalies'):
            print(f"  检测到的异常:")
            for anomaly in pose_quality['all_detected_anomalies']:
                print(f"    - {anomaly}")
    else:
        print("  未检测到人物")


def example_custom_thresholds():
    """自定义阈值：根据具体业务需求调整"""
    print("\n【示例4：自定义阈值】")
    print("="*60)
    
    filter = VideoQualityFilter()
    
    # 根据业务需求自定义阈值
    custom_thresholds = {
        'blur_threshold': 120.0,           # 中等清晰度要求
        'noise_threshold': 25.0,           # 中等噪声容忍
        'temporal_consistency_threshold': 0.75,  # 提高时序要求
        'overall_pass_score': 0.65         # 中等通过标准
    }
    
    filter.config['quality_thresholds'].update(custom_thresholds)
    
    print("自定义阈值配置:")
    for key, value in custom_thresholds.items():
        print(f"  - {key}: {value}")
    
    # 评估视频
    video_path = 'test_video.mp4'
    result = filter.evaluate_video(video_path, verbose=False)
    
    print(f"\n结果: {'? 通过' if result['overall_assessment']['passed'] else '? 未通过'}")
    print(f"总体分数: {result['overall_assessment']['overall_score']:.3f}")
    
    # 显示各项指标是否达标
    print("\n各项指标达标情况:")
    print(f"  图像质量: {'?' if result['image_quality']['passed'] else '?'}")
    print(f"  姿态质量: {'?' if result['pose_quality']['passed'] else '?'}")
    print(f"  时序一致性: {'?' if result['temporal_consistency']['passed'] else '?'}")


def main():
    print("AIGC视频筛选 - 自定义配置示例")
    print("="*60)
    
    # 运行各个示例
    # 注意：这些示例需要有实际的测试视频文件
    
    try:
        example_strict_mode()
        example_fast_mode()
        example_pose_focused()
        example_custom_thresholds()
    except Exception as e:
        print(f"\n错误: {e}")
        print("提示: 请确保有可用的测试视频文件")
    
    print("\n" + "="*60)
    print("示例运行完成！")
    print("你可以根据这些示例创建适合自己需求的配置")
    print("="*60)


if __name__ == '__main__':
    main()

