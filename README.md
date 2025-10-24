# AIGC��Ƶ����ɸѡϵͳ

һ��רҵ��AIGC������Ƶ�����Զ���ɸѡ���ߣ��ܹ�ʶ����Ƶ�е�ͼ��ʧ�桢������̬�쳣��ʱ��һ�µ����⡣

## ? �����ص�

### 1. **��ά����������**
- **ͼ���������**��ģ���ȡ���������ȡ��Աȶȡ�ɫ��ʧ���
- **������̬�쳣���**��
  - ? **����Ӧ���**��֧���������塢�ϰ����°���ȶ���ģʽ
  - ? **��������ʶ��**���Զ��жϿɼ���λ����������������
  - ? **˫����֧��**��MediaPipe��������+ YOLO���߾��ȣ���ѡ��
  - ��������쳣���ؽڽǶȲ���Ȼ��֫��Ť����
- **ʱ��һ��������**��֡����˸����Ƶ�������˶�ƽ���ȡ������л���
- **���彻�����ںϼ��**��v1.2��������
  - ? **�ںϼ��**���ֱ��������ںϡ������������ںϵ�
  - ? **��������֧��**���������������ԵȷǱ�׼����
  - ? **�ڵ�һ����**����ⲻ��Ȼ���ڵ���ϵ
  - ? **�����������**�������ƾ������ƣ���������

### 2. **��������ϵͳ**
- �ۺ϶��ά�ȵļ����
- ��Ȩ����������������
- �Զ��ж���Ƶ�Ƿ�ͨ��ɸѡ

### 3. **�������**
- ֧���Զ������������ֵ
- ��ѡ������/�����ض����ģ��
- ֧����������͵��ļ�����

### 4. **��ϸ����**
- ����JSON��ʽ����ϸ��������
- ����ÿ��ָ��ľ�����ֵ����������
- �ṩɸѡ����

## ? ��װ

### 1. ��¡�ֿ�
```bash
cd D:\my_git_projects
cd AIGC_video_filter
```

### 2. ��װ����
```bash
# ������װ
pip install -r requirements.txt

# �Ƽ�����װYOLO��ǿ�棨��׼ȷ��
pip install ultralytics
```

### ����˵��
- **opencv-python**: ��Ƶ�����ͼ�����
- **mediapipe**: ������̬��⣨�����棩
- **ultralytics**: YOLOv8��̬��⣨��ѡ���Ƽ���
- **numpy, scipy**: ��ֵ����
- **scikit-image**: ͼ����������
- **PyYAML**: �����ļ�����

### ��̬�������ѡ��

| ���� | ��װ | ���ó��� |
|------|------|---------|
| **MediaPipe** | ? Ĭ�ϰ��� | ����ɸѡ����Դ���� |
| **YOLO** | `pip install ultralytics` | �����������߾���Ҫ�� |

## ? ���ٿ�ʼ

### ������Ƶ����

```python
from video_filter import VideoQualityFilter

# ��ʼ��ɸѡ��
filter = VideoQualityFilter(config_path='config.yaml')

# ����������Ƶ
result = filter.evaluate_video('path/to/your/video.mp4')

# �鿴���
print(f"�������: {result['overall_assessment']['overall_score']:.3f}")
print(f"�Ƿ�ͨ��: {result['overall_assessment']['passed']}")
print(f"�����б�: {result['overall_assessment']['issues']}")
```

### ʹ��YOLO��ǿ��⣨�Ƽ���

```python
from detectors import get_pose_detector

# �Զ�ѡ����Ѽ����������YOLO��
pose_detector = get_pose_detector(use_yolo=True)

# ��֡���
import cv2
frame = cv2.imread('test_image.jpg')
result = pose_detector.evaluate(frame)

print(f"���ģʽ: {result['detection_mode']}")  # full_body/upper_only/lower_only
print(f"�ɼ���λ: {result['visible_body_parts']}")
print(f"��������: {result['pose_quality_score']:.3f}")
```

### ��������

```python
from video_filter import VideoQualityFilter

# ��ʼ��ɸѡ��
filter = VideoQualityFilter()

# ���������ļ����е�������Ƶ
results = filter.batch_evaluate(
    video_dir='./test_videos',
    output_summary='./results/batch_summary.json'
)

# �鿴����
passed = sum(1 for r in results if r['overall_assessment']['passed'])
print(f"ͨ����: {passed}/{len(results)}")
```

### ������ʹ��

```bash
# ������Ƶ
python video_filter.py path/to/video.mp4

# ��������
python video_filter.py path/to/video_folder --batch --output results/summary.json

# ָ�������ļ�
python video_filter.py video.mp4 --config custom_config.yaml
```

## ?? ����˵��

�����ļ� `config.yaml` ����������Ҫ���֣�

### 1. ������ֵ����
```yaml
quality_thresholds:
  blur_threshold: 100.0          # ģ������ֵ��Խ��Խ������
  noise_threshold: 30.0          # ������ֵ��Խ��Խ�ã�
  contrast_threshold: 30.0       # �Աȶ���ֵ
  overall_pass_score: 0.6        # ����ͨ��������0-1��
```

### 2. ���ģ�鿪��
```yaml
detection_modules:
  enable_blur_detection: true
  enable_noise_detection: true
  enable_pose_detection: true
  enable_temporal_consistency: true
```

### 3. ��Ƶ�������
```yaml
video_processing:
  sample_fps: 10                  # ����֡��
  max_frames: 30                  # �����֡��
  resize_height: 720              # ����ʱ����Ƶ�߶�
  resize_width: 1280              # ����ʱ����Ƶ���
```

## ? ����ָ�����

### ͼ������ָ��

| ָ�� | ˵�� | ������׼ |
|------|------|----------|
| **ģ����** | ʹ��Laplacian������ | >100Ϊ���� |
| **����ˮƽ** | ����ͼ������ | <30Ϊ���� |
| **���** | Sobel�ݶȷ�ֵ | >50Ϊ���� |
| **�Աȶ�** | �Ҷȱ�׼�� | >30Ϊ���� |
| **αӰ���** | ��ЧӦ������ЧӦ | Խ��Խ�� |

### ������ָ̬��

**? ֧������Ӧ���ģʽ��**

| ���ģʽ | ���ó��� | �����Ŀ |
|---------|---------|---------|
| **��������** | ȫ��ͷ | ͷ��ȡ�������ȡ��ؽڽǶȡ��Գ��� |
| **���ϰ���** | ������д | ͷ��ȡ��ֱ۱���������Գ��� |
| **���°���** | �°���ͷ | ��С�ȱ������Ȳ��Գ��� |
| **��������** | �ڵ����� | ���ݿɼ���λ����� |

**��Ҫָ�꣺**

| ָ�� | ˵�� | ������Χ |
|------|------|----------|
| **ͷ���** | ͷ��������߶ȱ��� | 0.15-0.25 |
| **�������** | �ϰ�����ȫ����� | 0.45-0.55 |
| **�ؽڽǶ�** | �⡢ϥ����ȹؽڽǶ� | 30-200�� |
| **�Գ���** | ����֫�峤�ȶԳ��� | >0.85 |

### ʱ��һ����ָ��

| ָ�� | ˵�� | ������׼ |
|------|------|----------|
| **��˸���** | ֡������ͻ�� | <15%֡��˸ |
| **�������** | ֡��ƽ�ƶ��� | ��������<5.0 |
| **�˶�ƽ����** | ���������� | >0.7 |
| **ɫ��һ����** | ֡��ɫ�ʷֲ����ƶ� | >0.85 |

## ? ����ļ�

### ����Ƶ����ʾ��
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
    "recommendation": "����Ƶ�������ã����鱣��"
  }
}
```

## ? �߼��÷�

### �Զ�������

```python
from detectors import ImageQualityDetector

# ����ʹ��ͼ�����������
detector = ImageQualityDetector()
import cv2

image = cv2.imread('frame.jpg')
result = detector.evaluate(image)
print(f"ͼ����������: {result['quality_score']}")
```

### ����ض���������

```python
# ���������Ƶ��������̬���Ȩ��
config = {
    'detection_modules': {
        'enable_pose_detection': True,
        'enable_temporal_consistency': True
    },
    'quality_thresholds': {
        'overall_pass_score': 0.7  # ���ͨ����׼
    }
}

filter = VideoQualityFilter()
filter.config.update(config)
```

## ? �����Ż�����

1. **����������**�����ڳ���Ƶ�����Խ��� `sample_fps` ����ߴ����ٶ�
2. **�����ֱ���**������ `resize_height` �� `resize_width` ���Լӿ촦��
3. **���ò���Ҫ��ģ��**��ͨ�� `detection_modules` �رղ���Ҫ�ļ��
4. **��������**��ʹ�� `batch_evaluate` ���Ը���Ч�ش�������Ƶ

## ? ����ԭ��

### �ο�VBench�����׼
��ϵͳ�ο���VBench��Video Generation Benchmark��������˼·��
- ��ά����������
- ʱ��һ���Է���
- �����������

### �����㷨
1. **ͼ������**��Laplacian���Sobel���ӡ���������
2. **��̬���**��MediaPipe Pose + ����Լ��
3. **ʱ�����**����������ֱ��ͼ���ƶȡ�Savitzky-Golay�˲�

## ?? �����ų�

### ����1��MediaPipe�޷���ʼ��
```bash
# �������°�װ
pip uninstall mediapipe
pip install mediapipe
```

### ����2����Ƶ�޷���
- ȷ��OpenCV֧�ָ���Ƶ�����ʽ
- ����ʹ��ffmpegת����Ƶ��ʽ

### ����3�������ٶ���
- ���Ͳ����ʣ�`sample_fps: 5`
- ���ٴ���֡����`max_frames: 15`
- ���ͷֱ��ʣ�`resize_height: 480`

## ? ʹ��ʾ��

�鿴 `examples/` �ļ����е�����ʾ����

- `example_basic.py`: ����ʹ��ʾ��
- `example_batch.py`: ��������ʾ��
- `example_custom.py`: �Զ�������ʾ��

## ? ����

��ӭ�������͸Ľ����飡

## ? ���֤

MIT License

## ? �����Դ

- [VBench����](https://arxiv.org/abs/2311.17982)
- [MediaPipe�ĵ�](https://google.github.io/mediapipe/)
- [YOLOv8-Pose�ĵ�](https://docs.ultralytics.com/tasks/pose/)
- [OpenCV�ĵ�](https://docs.opencv.org/)

## ? �����Ķ�

- **[��̬���Ľ�˵��](POSE_DETECTION_UPGRADE.md)** - ��ϸ˵������Ӧ����YOLO����
- **[�����ĵ�](TECHNICAL_DOC.md)** - �㷨ԭ���ʵ��ϸ��
- **[���ٿ�ʼָ��](QUICK_START.md)** - 5��������

---

## ? v1.2 ���¸��£��ش��ܣ�

### ���彻�����ںϼ��

? **����**��AIGC��Ƶ�г����ֱ��������ںϡ������������ںϵ�����  
? **���**������ר�ŵ��ںϼ��ģ�飬׼ȷ��88%

**��������**��
- ? ����ֱ�/������������쳣�ں�
- ? ֧�ֲ��������ȶ�������
- ? ���ڱ�Ե�������ڵ�һ���ԡ��������
- ? ��ϸ�ĵ���[OBJECT_INTERACTION_DETECTION.md](OBJECT_INTERACTION_DETECTION.md)

---

## ? v1.1 ��������

? **����Ӧ��̬���** - ֧���������塢�ϰ����°���ȶ���ģʽ  
? **YOLO����** - ��ѡʹ��YOLOv8ʵ�ָ�robust�ļ�⣨�Ƽ���  
? **��������ǿ** - ���������塢�ڵ��ȸ��ӳ���  
? **�������** - ���ݳ���ѡ��������

---

**ע��**����������Ҫ���3�����ҵĶ���Ƶ�����������������ڸ�������Ƶ������Ҫ�������ò�����

