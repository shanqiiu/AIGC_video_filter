"""
基础使用示例：评估单个视频
"""

import sys
sys.path.append('..')

from video_filter import VideoQualityFilter


def main():
    # 初始化视频质量筛选器
    print("初始化AIGC视频质量筛选器...")
    filter = VideoQualityFilter(config_path='../config.yaml')
    
    # 评估单个视频
    video_path = 'test_video.mp4'  # 替换为你的视频路径
    
    print(f"\n开始评估视频: {video_path}")
    print("="*60)
    
    result = filter.evaluate_video(video_path, verbose=True)
    
    # 详细结果展示
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
    
    # 图像质量
    print("\n【图像质量】")
    img_quality = result['image_quality']
    print(f"  模糊度分数: {img_quality['blur_score']:.2f} (>100为清晰)")
    print(f"  噪声水平: {img_quality['noise_level']:.2f} (<30为良好)")
    print(f"  锐度: {img_quality['sharpness']:.2f}")
    print(f"  对比度: {img_quality['contrast']:.2f}")
    print(f"  质量评分: {img_quality['quality_score']:.3f}")
    print(f"  状态: {'? 通过' if img_quality['passed'] else '? 未通过'}")
    
    # 人体姿态
    print("\n【人体姿态】")
    pose_quality = result['pose_quality']
    if 'note' in pose_quality:
        print(f"  说明: {pose_quality['note']}")
    else:
        print(f"  检测帧数: {pose_quality['sampled_frames']}")
        print(f"  人物检出率: {pose_quality['person_detection_rate']:.1%}")
        print(f"  异常检出率: {pose_quality['anomaly_rate']:.1%}")
        if pose_quality.get('all_detected_anomalies'):
            print(f"  检测到的异常:")
            for anomaly in pose_quality['all_detected_anomalies'][:5]:  # 显示前5个
                print(f"    - {anomaly}")
    print(f"  状态: {'? 通过' if pose_quality['passed'] else '? 未通过'}")
    
    # 时序一致性
    print("\n【时序一致性】")
    temporal = result['temporal_consistency']
    print(f"  整体分数: {temporal['temporal_consistency_score']:.3f}")
    print(f"  闪烁检测: {'有' if temporal['flicker_detection']['has_flicker'] else '无'}")
    print(f"  抖动检测: {'有' if temporal['jitter_detection']['has_jitter'] else '无'}")
    print(f"  运动平滑度: {temporal['motion_smoothness']['smoothness_score']:.3f}")
    print(f"  色彩一致性: {temporal['color_consistency']['color_consistency_score']:.3f}")
    print(f"  状态: {'? 通过' if temporal['passed'] else '? 未通过'}")
    
    # 综合评估
    print("\n" + "="*60)
    print("【综合评估】")
    print("="*60)
    overall = result['overall_assessment']
    print(f"  总体分数: {overall['overall_score']:.3f} / 1.0")
    print(f"  筛选结果: {'? 通过' if overall['passed'] else '? 未通过'}")
    
    if overall['issues']:
        print(f"\n  发现的问题:")
        for issue in overall['issues']:
            print(f"    - {issue}")
    
    print(f"\n  建议: {overall['recommendation']}")
    print("="*60)


if __name__ == '__main__':
    main()


