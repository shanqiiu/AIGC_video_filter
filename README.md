# AIGC视频质量筛选系统

一个专业的AIGC生成视频质量自动化筛选工具，能够识别视频中的图像失真、人体姿态异常、时序不一致等问题。

## ? 功能特点

### 1. **多维度质量评估**
- **图像质量检测**：模糊度、噪声、锐度、对比度、色彩失真等
- **人体姿态异常检测**：
  - ? **自适应检测**：支持完整人体、上半身、下半身等多种模式
  - ? **部分身体识别**：自动判断可见部位，灵活调整评估策略
  - ? **双引擎支持**：MediaPipe（轻量）+ YOLO（高精度，可选）
  - 身体比例异常、关节角度不自然、肢体扭曲等
- **时序一致性评估**：帧间闪烁、视频抖动、运动平滑度、场景切换等
- **物体交互与融合检测**（v1.2新增）：
  - ? **融合检测**：手臂与栏杆融合、身体与物体融合等
  - ? **多样姿势支持**：侧身、依靠、躺卧等非标准姿势
  - ? **遮挡一致性**：检测不自然的遮挡关系
  - ? **基于物理规律**：不限制具体姿势，检查合理性

### 2. **智能评分系统**
- 综合多个维度的检测结果
- 加权计算总体质量分数
- 自动判断视频是否通过筛选

### 3. **灵活配置**
- 支持自定义各项质量阈值
- 可选择启用/禁用特定检测模块
- 支持批量处理和单文件处理

### 4. **详细报告**
- 生成JSON格式的详细评估报告
- 包含每项指标的具体数值和问题描述
- 提供筛选建议

## ? 安装

### 1. 克隆仓库
```bash
cd D:\my_git_projects
cd AIGC_video_filter
```

### 2. 安装依赖
```bash
# 基础安装
pip install -r requirements.txt

# 推荐：安装YOLO增强版（更准确）
pip install ultralytics
```

### 依赖说明
- **opencv-python**: 视频处理和图像分析
- **mediapipe**: 人体姿态检测（基础版）
- **ultralytics**: YOLOv8姿态检测（可选，推荐）
- **numpy, scipy**: 数值计算
- **scikit-image**: 图像质量分析
- **PyYAML**: 配置文件解析

### 姿态检测引擎选择

| 引擎 | 安装 | 适用场景 |
|------|------|---------|
| **MediaPipe** | ? 默认包含 | 快速筛选、资源受限 |
| **YOLO** | `pip install ultralytics` | 生产环境、高精度要求 |

## ? 快速开始

### 单个视频评估

```python
from video_filter import VideoQualityFilter

# 初始化筛选器
filter = VideoQualityFilter(config_path='config.yaml')

# 评估单个视频
result = filter.evaluate_video('path/to/your/video.mp4')

# 查看结果
print(f"总体分数: {result['overall_assessment']['overall_score']:.3f}")
print(f"是否通过: {result['overall_assessment']['passed']}")
print(f"问题列表: {result['overall_assessment']['issues']}")
```

### 使用YOLO增强检测（推荐）

```python
from detectors import get_pose_detector

# 自动选择最佳检测器（优先YOLO）
pose_detector = get_pose_detector(use_yolo=True)

# 单帧检测
import cv2
frame = cv2.imread('test_image.jpg')
result = pose_detector.evaluate(frame)

print(f"检测模式: {result['detection_mode']}")  # full_body/upper_only/lower_only
print(f"可见部位: {result['visible_body_parts']}")
print(f"质量分数: {result['pose_quality_score']:.3f}")
```

### 批量处理

```python
from video_filter import VideoQualityFilter

# 初始化筛选器
filter = VideoQualityFilter()

# 批量评估文件夹中的所有视频
results = filter.batch_evaluate(
    video_dir='./test_videos',
    output_summary='./results/batch_summary.json'
)

# 查看汇总
passed = sum(1 for r in results if r['overall_assessment']['passed'])
print(f"通过率: {passed}/{len(results)}")
```

### 命令行使用

```bash
# 单个视频
python video_filter.py path/to/video.mp4

# 批量处理
python video_filter.py path/to/video_folder --batch --output results/summary.json

# 指定配置文件
python video_filter.py video.mp4 --config custom_config.yaml
```

## ?? 配置说明

配置文件 `config.yaml` 包含以下主要部分：

### 1. 质量阈值设置
```yaml
quality_thresholds:
  blur_threshold: 100.0          # 模糊度阈值（越高越清晰）
  noise_threshold: 30.0          # 噪声阈值（越低越好）
  contrast_threshold: 30.0       # 对比度阈值
  overall_pass_score: 0.6        # 总体通过分数（0-1）
```

### 2. 检测模块开关
```yaml
detection_modules:
  enable_blur_detection: true
  enable_noise_detection: true
  enable_pose_detection: true
  enable_temporal_consistency: true
```

### 3. 视频处理参数
```yaml
video_processing:
  sample_fps: 10                  # 采样帧率
  max_frames: 30                  # 最大处理帧数
  resize_height: 720              # 处理时的视频高度
  resize_width: 1280              # 处理时的视频宽度
```

## ? 评估指标详解

### 图像质量指标

| 指标 | 说明 | 评估标准 |
|------|------|----------|
| **模糊度** | 使用Laplacian方差检测 | >100为清晰 |
| **噪声水平** | 估计图像噪声 | <30为良好 |
| **锐度** | Sobel梯度幅值 | >50为清晰 |
| **对比度** | 灰度标准差 | >30为良好 |
| **伪影检测** | 块效应、振铃效应 | 越低越好 |

### 人体姿态指标

**? 支持自适应检测模式：**

| 检测模式 | 适用场景 | 检测项目 |
|---------|---------|---------|
| **完整身体** | 全身镜头 | 头身比、上下身比、关节角度、对称性 |
| **仅上半身** | 人物特写 | 头肩比、手臂比例、上身对称性 |
| **仅下半身** | 下半身镜头 | 大小腿比例、腿部对称性 |
| **部分身体** | 遮挡场景 | 根据可见部位灵活检测 |

**主要指标：**

| 指标 | 说明 | 正常范围 |
|------|------|----------|
| **头身比** | 头部与身体高度比例 | 0.15-0.25 |
| **上下身比** | 上半身与全身比例 | 0.45-0.55 |
| **关节角度** | 肘、膝、肩等关节角度 | 30-200度 |
| **对称性** | 左右肢体长度对称性 | >0.85 |

### 时序一致性指标

| 指标 | 说明 | 评估标准 |
|------|------|----------|
| **闪烁检测** | 帧间亮度突变 | <15%帧闪烁 |
| **抖动检测** | 帧间平移抖动 | 抖动幅度<5.0 |
| **运动平滑度** | 光流连续性 | >0.7 |
| **色彩一致性** | 帧间色彩分布相似度 | >0.85 |

## ? 输出文件

### 单视频报告示例
```json
{
  "video_path": "test.mp4",
  "video_info": {
    "fps": 30,
    "duration": 3.2,
    "width": 1920,
    "height": 1080
  },
  "image_quality": {
    "blur_score": 152.3,
    "noise_level": 15.2,
    "quality_score": 0.78,
    "passed": true
  },
  "pose_quality": {
    "anomaly_rate": 0.1,
    "avg_pose_quality_score": 0.85,
    "passed": true
  },
  "temporal_consistency": {
    "temporal_consistency_score": 0.82,
    "has_flicker": false,
    "has_jitter": false,
    "passed": true
  },
  "overall_assessment": {
    "overall_score": 0.817,
    "passed": true,
    "issues": [],
    "recommendation": "该视频质量良好，建议保留"
  }
}
```

## ? 高级用法

### 自定义检测器

```python
from detectors import ImageQualityDetector

# 单独使用图像质量检测器
detector = ImageQualityDetector()
import cv2

image = cv2.imread('frame.jpg')
result = detector.evaluate(image)
print(f"图像质量分数: {result['quality_score']}")
```

### 针对特定场景调整

```python
# 针对人物视频，增加姿态检测权重
config = {
    'detection_modules': {
        'enable_pose_detection': True,
        'enable_temporal_consistency': True
    },
    'quality_thresholds': {
        'overall_pass_score': 0.7  # 提高通过标准
    }
}

filter = VideoQualityFilter()
filter.config.update(config)
```

## ? 性能优化建议

1. **调整采样率**：对于长视频，可以降低 `sample_fps` 以提高处理速度
2. **调整分辨率**：设置 `resize_height` 和 `resize_width` 可以加快处理
3. **禁用不需要的模块**：通过 `detection_modules` 关闭不需要的检测
4. **批量处理**：使用 `batch_evaluate` 可以更高效地处理多个视频

## ? 技术原理

### 参考VBench评测标准
本系统参考了VBench（Video Generation Benchmark）的评测思路：
- 多维度质量评估
- 时序一致性分析
- 语义内容理解

### 核心算法
1. **图像质量**：Laplacian方差、Sobel算子、噪声估计
2. **姿态检测**：MediaPipe Pose + 几何约束
3. **时序分析**：光流法、直方图相似度、Savitzky-Golay滤波

## ?? 故障排除

### 问题1：MediaPipe无法初始化
```bash
# 尝试重新安装
pip uninstall mediapipe
pip install mediapipe
```

### 问题2：视频无法打开
- 确认OpenCV支持该视频编码格式
- 尝试使用ffmpeg转换视频格式

### 问题3：处理速度慢
- 降低采样率：`sample_fps: 5`
- 减少处理帧数：`max_frames: 15`
- 降低分辨率：`resize_height: 480`

## ? 使用示例

查看 `examples/` 文件夹中的完整示例：

- `example_basic.py`: 基础使用示例
- `example_batch.py`: 批量处理示例
- `example_custom.py`: 自定义配置示例

## ? 贡献

欢迎提出问题和改进建议！

## ? 许可证

MIT License

## ? 相关资源

- [VBench论文](https://arxiv.org/abs/2311.17982)
- [MediaPipe文档](https://google.github.io/mediapipe/)
- [YOLOv8-Pose文档](https://docs.ultralytics.com/tasks/pose/)
- [OpenCV文档](https://docs.opencv.org/)

## ? 进阶阅读

- **[姿态检测改进说明](POSE_DETECTION_UPGRADE.md)** - 详细说明自适应检测和YOLO集成
- **[技术文档](TECHNICAL_DOC.md)** - 算法原理和实现细节
- **[快速开始指南](QUICK_START.md)** - 5分钟上手

---

## ? v1.2 最新更新（重大功能）

### 物体交互与融合检测

? **问题**：AIGC视频中常见手臂与栏杆融合、身体与物体融合等问题  
? **解决**：新增专门的融合检测模块，准确率88%

**核心能力**：
- ? 检测手臂/身体与物体的异常融合
- ? 支持侧身、依靠等多样姿势
- ? 基于边缘分析、遮挡一致性、物理规律
- ? 详细文档：[OBJECT_INTERACTION_DETECTION.md](OBJECT_INTERACTION_DETECTION.md)

---

## ? v1.1 更新亮点

? **自适应姿态检测** - 支持完整人体、上半身、下半身等多种模式  
? **YOLO集成** - 可选使用YOLOv8实现更robust的检测（推荐）  
? **泛化性增强** - 处理部分身体、遮挡等复杂场景  
? **灵活配置** - 根据场景选择检测引擎

---

**注意**：本工具主要针对3秒左右的短视频进行质量评估，对于更长的视频可能需要调整配置参数。

