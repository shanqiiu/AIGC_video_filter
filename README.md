# AIGC视频质量筛选器（YOLO版本）

基于YOLO的AIGC视频质量筛选工具，专注于人体姿态异常检测。

## 功能特点

- **仅使用YOLO检测**: 简化架构，仅使用YOLO进行人体姿态检测
- **高效检测**: 基于YOLOv8-pose模型，检测速度快
- **姿态异常检测**: 检测身体比例异常、关节角度异常等
- **批量处理**: 支持批量视频质量评估
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

### 4. 编程接口使用

```python
from video_filter import VideoQualityFilter

# 初始化筛选器
filter = VideoQualityFilter(config_path='config.yaml')

# 评估视频
result = filter.evaluate_video('test_video.mp4', verbose=True)
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

# 输出设置
output:
  save_report: true               # 保存详细报告
  output_dir: "./results"         # 输出目录
```

## 命令行参数

```bash
python video_filter.py <input> [options]

参数:
  input              输入视频文件或目录路径
  -o, --output       输出摘要文件路径（可选）
  -c, --config       配置文件路径（默认：config.yaml）
  -v, --verbose      显示详细信息
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

## 输出结果

评估结果包含：
- 视频基本信息（分辨率、帧率、时长等）
- 姿态检测结果（检测率、置信度、异常数）
- 综合评分和通过状态

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
├── config.yaml                       # 配置文件
├── requirements.txt                  # 依赖列表
└── README.md                         # 说明文档
```

## 简化说明

相比原版本，本版本进行了以下简化：

1. **仅使用YOLO检测**: 移除了MediaPipe、图像质量检测、时序一致性检测等模块
2. **精简配置**: 移除了不必要的配置项
3. **简化代码**: 删除了冗余的检测逻辑
4. **专注姿态**: 专注于人体姿态异常检测功能

这样的简化使得代码更加清晰，维护更容易，同时保持了核心的检测功能。