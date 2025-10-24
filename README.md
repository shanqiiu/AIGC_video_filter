# AIGC视频质量筛选器（YOLO版本）

基于YOLO的AIGC视频质量筛选工具，专注于人体姿态异常检测。

## 功能特点

- **仅使用YOLO检测**: 简化架构，仅使用YOLO进行人体姿态检测
- **高效检测**: 基于YOLOv8-pose模型，检测速度快
- **姿态异常检测**: 检测身体比例异常、关节角度异常等
- **批量处理**: 支持批量视频质量评估
- **可视化功能**: 支持姿态可视化、异常标注、统计图表等
- **简化配置**: 精简的配置文件，易于使用

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `ultralytics`: YOLO模型
- `opencv-python`: 视频处理
- `numpy`: 数值计算
- `PyYAML`: 配置文件
- `tqdm`: 进度条

可视化依赖：
- `matplotlib`: 统计图表
- `seaborn`: 高级图表
- `plotly`: 交互式仪表板
- `Pillow`: 图像处理

## 快速开始

### 1. 单个视频评估

```bash
python video_filter.py test_video.mp4 -v
```

### 2. 批量视频评估

```bash
python video_filter.py video_directory -o results/summary.json
```

### 3. 使用自定义配置

```bash
python video_filter.py test_video.mp4 -c my_config.yaml -v
```

### 4. 可视化功能使用

```bash
# 启用所有可视化功能
python video_filter.py test_video.mp4 --enable-viz -v

# 只保存姿态检测图像
python video_filter.py test_video.mp4 --save-pose -v

# 保存异常检测图像
python video_filter.py test_video.mp4 --save-anomaly -v

# 保存对比图像
python video_filter.py test_video.mp4 --save-comparison -v

# 保存统计图表
python video_filter.py test_video.mp4 --save-charts -v
```

### 5. 编程接口使用

```python
from video_filter import VideoQualityFilter

# 初始化筛选器
filter = VideoQualityFilter(config_path='config.yaml')

# 评估视频
result = filter.evaluate_video('test_video.mp4', verbose=True)

# 可视化结果保存在 results/visualizations/ 目录
```

## 配置说明

配置文件 `config.yaml`:

```yaml
# 质量阈值设置
quality_thresholds:
  pose_confidence_threshold: 0.5  # 姿态检测置信度
  overall_pass_score: 0.6        # 总体通过分数

# 视频处理参数
video_processing:
  sample_fps: 10                  # 采样帧率
  max_frames: 30                  # 最大处理帧数
  resize_height: 720              # 处理高度
  resize_width: 1280              # 处理宽度

# 可视化设置
visualization:
  enable_visualization: false     # 是否启用可视化功能
  save_pose_images: false         # 保存姿态检测结果图像
  save_anomaly_images: false      # 保存异常检测结果图像
  save_comparison_images: false   # 保存原始帧与检测结果对比图
  save_statistics_charts: false   # 保存统计图表
  pose_skeleton_color: [0, 255, 0]  # 姿态骨架颜色 (BGR)
  anomaly_color: [0, 0, 255]     # 异常标注颜色 (BGR)
  confidence_color: [255, 0, 0]  # 置信度标注颜色 (BGR)
  line_thickness: 2              # 绘制线条粗细
  font_scale: 0.6                # 字体大小
  show_confidence: true          # 显示置信度数值
  show_keypoint_names: false     # 显示关键点名称

# 输出设置
output:
  save_report: true               # 保存详细报告
  output_dir: "./results"         # 输出目录
  visualization_dir: "./results/visualizations"  # 可视化结果目录
```

## 命令行参数

```bash
python video_filter.py <input> [options]

参数:
  input              输入视频文件或目录路径
  -o, --output       输出摘要文件路径（可选）
  -c, --config       配置文件路径（默认：config.yaml）
  -v, --verbose      显示详细信息
  --enable-viz       启用可视化功能
  --save-pose        保存姿态检测图像
  --save-anomaly     保存异常检测图像
  --save-comparison  保存对比图像
  --save-charts      保存统计图表
  -h, --help         显示帮助信息

示例:
  # 评估单个视频
  python video_filter.py test_video.mp4 -v
  
  # 批量评估目录
  python video_filter.py video_directory -o results/summary.json
  
  # 使用自定义配置
  python video_filter.py test_video.mp4 -c my_config.yaml -v
```

## 检测指标

### 人体姿态检测
- **检测率**: 成功检测到人体的帧数比例
- **置信度**: 检测的平均置信度
- **异常数量**: 检测到的姿态异常数量

### 身体比例检查
- 头肩比例检查
- 大小腿比例检查

### 关节角度检查
- 肘关节角度检查
- 膝关节角度检查

## 可视化功能

### 姿态可视化
- **骨架绘制**: 在图像上绘制人体姿态骨架
- **关键点标注**: 显示17个关键点位置
- **异常高亮**: 用不同颜色标注异常关键点
- **置信度显示**: 显示检测置信度数值

### 异常检测可视化
- **异常区域标注**: 高亮显示异常检测区域
- **异常计数**: 显示每帧的异常数量
- **异常类型**: 区分不同类型的异常

### 对比展示
- **原始vs检测**: 并排显示原始帧和检测结果
- **前后对比**: 显示处理前后的差异
- **质量对比**: 展示质量评估结果

### 统计图表
- **检测率分布**: 检测率直方图
- **置信度分布**: 置信度分布图
- **异常数量**: 异常数量趋势图
- **综合评分**: 评分变化趋势

### 交互式仪表板
- **实时数据**: 动态显示检测数据
- **交互操作**: 支持缩放、筛选等操作
- **多维度展示**: 同时显示多个指标

## 输出结果

评估结果包含：
- 视频基本信息（分辨率、帧率、时长等）
- 姿态检测结果（检测率、置信度、异常数）
- 综合评分和通过状态
- 可视化文件（图像、图表、仪表板）

## 注意事项

1. **模型下载**: 首次运行时会自动下载YOLOv8-pose模型
2. **GPU加速**: 建议使用GPU加速检测
3. **内存使用**: 处理大视频时注意内存使用情况
4. **依赖安装**: 确保正确安装ultralytics包

## 文件结构

```
AIGC_video_filter/
├── detectors/
│   ├── __init__.py
│   └── pose_anomaly_detector_yolo.py  # YOLO检测器
├── video_filter.py                   # 主视频过滤器（可直接运行）
├── visualization.py                  # 可视化模块
├── config.yaml                       # 配置文件
├── requirements.txt                  # 依赖列表
├── results/                          # 输出结果目录
│   ├── visualizations/               # 可视化结果目录
│   │   ├── *_pose_frame_*.jpg        # 姿态检测图像
│   │   ├── *_anomaly_frame_*.jpg     # 异常检测图像
│   │   ├── *_comparison_frame_*.jpg  # 对比图像
│   │   ├── *_statistics.png          # 统计图表
│   │   └── *_dashboard.html          # 交互式仪表板
│   └── summary.json                  # 评估摘要
└── README.md                         # 说明文档
```

## 简化说明

相比原版本，本版本进行了以下简化：

1. **仅使用YOLO检测**: 移除了MediaPipe、图像质量检测、时序一致性检测等模块
2. **精简配置**: 移除了不必要的配置项
3. **简化代码**: 删除了冗余的检测逻辑
4. **专注姿态**: 专注于人体姿态异常检测功能

这样的简化使得代码更加清晰，维护更容易，同时保持了核心的检测功能。