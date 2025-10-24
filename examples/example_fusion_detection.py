"""
物体融合与交互异常检测示例

展示如何检测：
1. 手臂与栏杆融合
2. 侧身姿势的异常
3. 人物与物体的不自然遮挡
"""

import sys
sys.path.append('..')

import cv2
import numpy as np


def example_fusion_detection():
    """检测物体融合问题"""
    
    print("="*70)
    print("物体融合与交互异常检测示例")
    print("="*70)
    
    # 测试图像路径
    test_image = 'test_fusion.jpg'  # 替换为你的测试图像
    
    try:
        image = cv2.imread(test_image)
        if image is None:
            print(f"? 无法读取图像: {test_image}")
            print("\n说明：")
            print("这个检测器专门用于检测AIGC视频中的常见问题：")
            print("1. 手臂与栏杆/桌子等物体融合")
            print("2. 身体部位与背景物体异常融合")
            print("3. 侧身、多样姿势下的异常")
            print("4. 不自然的遮挡和深度关系")
            return
    except Exception as e:
        print(f"读取图像出错: {e}")
        return
    
    print(f"\n正在分析图像: {test_image}")
    print("-"*70)
    
    # 导入检测器
    from detectors import ObjectInteractionDetector, PoseAnomalyDetector
    
    # 1. 先进行姿态检测（可选，但能提供更精确的结果）
    pose_detector = PoseAnomalyDetector()
    pose_result = pose_detector.detect_pose(image)
    
    landmarks = None
    if pose_result is not None:
        landmarks = pose_result['landmarks']
        print("? 检测到人物姿态")
    else:
        print("? 未检测到人物（将进行基于图像的分析）")
    
    # 2. 进行物体交互异常检测
    interaction_detector = ObjectInteractionDetector()
    result = interaction_detector.evaluate(image, landmarks)
    
    # 3. 显示结果
    print("\n【检测结果】")
    print("="*70)
    
    # 边缘分析
    print("\n1. 边缘异常分析:")
    edge_analysis = result['edge_analysis']
    print(f"   边缘一致性: {edge_analysis['edge_consistency']:.3f}")
    print(f"   边缘模糊度: {edge_analysis['edge_blur_score']:.3f}")
    print(f"   融合可能性: {edge_analysis['fusion_likelihood']:.3f}")
    
    if edge_analysis.get('has_fusion_anomaly', False):
        print("   ? **检测到疑似融合问题！**")
    else:
        print("   ? 边缘正常")
    
    # 遮挡分析
    print("\n2. 遮挡一致性分析:")
    occlusion_analysis = result['occlusion_analysis']
    print(f"   颜色过渡: {occlusion_analysis['color_transition_score']:.3f}")
    if 'limb_occlusion_score' in occlusion_analysis:
        print(f"   肢体遮挡: {occlusion_analysis['limb_occlusion_score']:.3f}")
    print(f"   遮挡一致: {'是' if occlusion_analysis['is_occlusion_consistent'] else '否'}")
    
    if not occlusion_analysis['is_occlusion_consistent']:
        print("   ? **检测到遮挡异常！**")
    
    # 姿势合理性
    if landmarks is not None:
        print("\n3. 姿势合理性分析:")
        pose_plausibility = result['pose_plausibility']
        print(f"   对称性分数: {pose_plausibility['symmetry_score']:.3f}")
        print(f"   连续性分数: {pose_plausibility['continuity_score']:.3f}")
        print(f"   稳定性分数: {pose_plausibility['stability_score']:.3f}")
        print(f"   合理性评分: {pose_plausibility['plausibility_score']:.3f}")
        
        if not pose_plausibility['is_plausible']:
            print("   ? **检测到姿势异常！**")
            for anomaly in pose_plausibility['anomalies']:
                print(f"     - {anomaly}")
    
    # 综合评估
    print("\n" + "="*70)
    print("【综合评估】")
    print("="*70)
    print(f"总体异常分数: {result['overall_anomaly_score']:.3f} (0-1，越高越异常)")
    print(f"判定结果: {'? 有异常' if result['has_interaction_anomaly'] else '? 正常'}")
    
    if result['issues']:
        print(f"\n检测到的问题:")
        for issue in result['issues']:
            print(f"  ? {issue}")
    
    print(f"\n建议: {result['recommendation']}")
    print("="*70)


def example_specific_scenarios():
    """针对特定场景的检测示例"""
    
    print("\n\n")
    print("="*70)
    print("特定场景检测能力说明")
    print("="*70)
    
    scenarios = {
        "手臂与栏杆融合": {
            "检测原理": [
                "分析手臂边缘的清晰度和一致性",
                "检查手臂区域的颜色过渡是否自然",
                "评估肢体连线的颜色连续性"
            ],
            "关键指标": "edge_blur_score + limb_occlusion_score"
        },
        "侧身姿势异常": {
            "检测原理": [
                "不要求正面，检查身体对称性的一般规律",
                "验证关节连续性（相邻关节合理距离）",
                "评估姿势稳定性（重心位置）"
            ],
            "关键指标": "symmetry_score + continuity_score"
        },
        "依靠姿势异常": {
            "检测原理": [
                "检查身体与支撑物的接触区域",
                "分析重心是否合理",
                "验证支撑点与重心的关系"
            ],
            "关键指标": "stability_score + edge_consistency"
        },
        "背景融合": {
            "检测原理": [
                "使用超像素分割识别物体边界",
                "分析相邻区域的颜色过渡",
                "检测边缘的异常模糊"
            ],
            "关键指标": "color_transition_score + fusion_likelihood"
        }
    }
    
    for scenario, info in scenarios.items():
        print(f"\n【场景：{scenario}】")
        print(f"检测原理：")
        for principle in info['检测原理']:
            print(f"  ? {principle}")
        print(f"关键指标：{info['关键指标']}")
        print("-"*70)


def example_integration_with_video():
    """与视频筛选系统集成的示例"""
    
    print("\n\n")
    print("="*70)
    print("集成到视频筛选系统")
    print("="*70)
    
    print("""
使用方法：

1. 在video_filter.py中集成：

```python
from detectors import ObjectInteractionDetector

class VideoQualityFilter:
    def __init__(self, config_path='config.yaml'):
        # ... 现有代码 ...
        self.interaction_detector = ObjectInteractionDetector(self.config)
    
    def evaluate_frame_interaction(self, frame, pose_landmarks=None):
        '''评估单帧的物体交互问题'''
        return self.interaction_detector.evaluate(frame, pose_landmarks)
    
    def evaluate_video(self, video_path, verbose=True):
        # ... 提取帧 ...
        
        # 对关键帧进行交互检测
        sample_indices = np.linspace(0, len(frames)-1, min(10, len(frames)), dtype=int)
        
        interaction_scores = []
        for idx in sample_indices:
            # 先获取姿态
            pose_result = self.pose_detector.evaluate(frames[idx])
            landmarks = pose_result.get('landmarks') if pose_result.get('pose_detected') else None
            
            # 检测交互异常
            interaction_result = self.evaluate_frame_interaction(frames[idx], landmarks)
            interaction_scores.append(interaction_result['overall_anomaly_score'])
        
        # 添加到总评分
        avg_interaction_score = np.mean(interaction_scores)
        results['interaction_analysis'] = {
            'avg_anomaly_score': avg_interaction_score,
            'has_interaction_issues': avg_interaction_score > 0.5
        }
```

2. 在config.yaml中添加配置：

```yaml
detection_modules:
  enable_interaction_detection: true  # 启用物体交互检测
  
quality_thresholds:
  interaction_anomaly_threshold: 0.5  # 交互异常阈值
```

3. 更新综合评分：

```python
def _calculate_overall_score(self, results):
    # 添加交互检测分数
    interaction_score = 1.0 - results.get('interaction_analysis', {}).get('avg_anomaly_score', 0)
    
    # 更新权重
    weights = {
        'image': 0.30,
        'pose': 0.25,
        'temporal': 0.30,
        'interaction': 0.15  # 新增
    }
    
    overall_score = (
        image_score * weights['image'] +
        pose_score * weights['pose'] +
        temporal_score * weights['temporal'] +
        interaction_score * weights['interaction']
    )
```
    """)


def main():
    print("\n" + "?"*70)
    print("AIGC视频物体融合与交互异常检测")
    print("?"*70 + "\n")
    
    # 示例1：基础检测
    example_fusion_detection()
    
    # 示例2：场景说明
    example_specific_scenarios()
    
    # 示例3：集成方法
    example_integration_with_video()
    
    print("\n" + "="*70)
    print("? 核心优势:")
    print("="*70)
    print("1. 针对AIGC特有问题：手臂融合、身体融合等")
    print("2. 支持多样姿势：侧身、依靠、躺卧等非标准姿势")
    print("3. 无需完整人体：部分可见也能检测")
    print("4. 基于物理规律：不限制具体姿势，检查合理性")
    print("\n详细文档：参考 OBJECT_INTERACTION_DETECTION.md")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

