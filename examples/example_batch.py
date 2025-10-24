"""
批量处理示例：评估文件夹中的所有视频
"""

import sys
sys.path.append('..')

from video_filter import VideoQualityFilter
import os


def main():
    # 初始化筛选器
    print("初始化AIGC视频质量筛选器...")
    filter = VideoQualityFilter(config_path='../config.yaml')
    
    # 设置视频文件夹路径
    video_folder = './test_videos'  # 替换为你的视频文件夹路径
    
    if not os.path.exists(video_folder):
        print(f"错误: 文件夹 {video_folder} 不存在")
        print("请创建文件夹并放入测试视频，或修改 video_folder 变量")
        return
    
    # 批量评估
    print(f"\n开始批量评估文件夹: {video_folder}")
    print("="*60)
    
    results = filter.batch_evaluate(
        video_dir=video_folder,
        output_summary='../results/batch_summary.json'
    )
    
    # 分析结果
    print("\n" + "="*60)
    print("批量评估结果分析")
    print("="*60)
    
    # 统计各类问题
    issue_stats = {}
    passed_videos = []
    failed_videos = []
    
    for result in results:
        video_name = os.path.basename(result['video_path'])
        
        if result['overall_assessment']['passed']:
            passed_videos.append(video_name)
        else:
            failed_videos.append({
                'name': video_name,
                'score': result['overall_assessment']['overall_score'],
                'issues': result['overall_assessment']['issues']
            })
            
            # 统计问题类型
            for issue in result['overall_assessment']['issues']:
                if issue in issue_stats:
                    issue_stats[issue] += 1
                else:
                    issue_stats[issue] = 1
    
    # 打印统计结果
    print(f"\n总计视频数: {len(results)}")
    print(f"通过数量: {len(passed_videos)} ({len(passed_videos)/len(results)*100:.1f}%)")
    print(f"未通过数量: {len(failed_videos)} ({len(failed_videos)/len(results)*100:.1f}%)")
    
    # 显示通过的视频
    if passed_videos:
        print(f"\n? 通过的视频 ({len(passed_videos)}个):")
        for video in passed_videos[:10]:  # 最多显示10个
            print(f"  - {video}")
        if len(passed_videos) > 10:
            print(f"  ... 还有 {len(passed_videos)-10} 个")
    
    # 显示未通过的视频及问题
    if failed_videos:
        print(f"\n? 未通过的视频 ({len(failed_videos)}个):")
        # 按分数排序
        failed_videos.sort(key=lambda x: x['score'])
        
        for video_info in failed_videos:
            print(f"\n  - {video_info['name']}")
            print(f"    分数: {video_info['score']:.3f}")
            print(f"    问题: {', '.join(video_info['issues'])}")
    
    # 问题类型统计
    if issue_stats:
        print(f"\n问题类型统计:")
        sorted_issues = sorted(issue_stats.items(), key=lambda x: x[1], reverse=True)
        for issue, count in sorted_issues:
            print(f"  - {issue}: {count}次 ({count/len(results)*100:.1f}%)")
    
    print("\n" + "="*60)
    print(f"详细报告已保存至: ../results/")
    print("="*60)


if __name__ == '__main__':
    main()


