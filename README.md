# AIGC��Ƶ����ɸѡ����YOLO�汾��

����YOLO��AIGC��Ƶ����ɸѡ���ߣ�רע��������̬�쳣��⡣

## �����ص�

- **��ʹ��YOLO���**: �򻯼ܹ�����ʹ��YOLO����������̬���
- **��Ч���**: ����YOLOv8-poseģ�ͣ�����ٶȿ�
- **��̬�쳣���**: �����������쳣���ؽڽǶ��쳣��
- **��������**: ֧��������Ƶ��������
- **���ӻ�����**: ֧����̬���ӻ����쳣��ע��ͳ��ͼ���
- **������**: ����������ļ�������ʹ��

## ��װ����

```bash
pip install -r requirements.txt
```

��Ҫ������
- `ultralytics`: YOLOģ��
- `opencv-python`: ��Ƶ����
- `numpy`: ��ֵ����
- `PyYAML`: �����ļ�
- `tqdm`: ������

���ӻ�������
- `matplotlib`: ͳ��ͼ��
- `seaborn`: �߼�ͼ��
- `plotly`: ����ʽ�Ǳ��
- `Pillow`: ͼ����

## ���ٿ�ʼ

### 1. ������Ƶ����

```bash
python video_filter.py test_video.mp4 -v
```

### 2. ������Ƶ����

```bash
python video_filter.py video_directory -o results/summary.json
```

### 3. ʹ���Զ�������

```bash
python video_filter.py test_video.mp4 -c my_config.yaml -v
```

### 4. ���ӻ�����ʹ��

```bash
# �������п��ӻ�����
python video_filter.py test_video.mp4 --enable-viz -v

# ֻ������̬���ͼ��
python video_filter.py test_video.mp4 --save-pose -v

# �����쳣���ͼ��
python video_filter.py test_video.mp4 --save-anomaly -v

# ����Ա�ͼ��
python video_filter.py test_video.mp4 --save-comparison -v

# ����ͳ��ͼ��
python video_filter.py test_video.mp4 --save-charts -v
```

### 5. ��̽ӿ�ʹ��

```python
from video_filter import VideoQualityFilter

# ��ʼ��ɸѡ��
filter = VideoQualityFilter(config_path='config.yaml')

# ������Ƶ
result = filter.evaluate_video('test_video.mp4', verbose=True)

# ���ӻ���������� results/visualizations/ Ŀ¼
```

## ����˵��

�����ļ� `config.yaml`:

```yaml
# ������ֵ����
quality_thresholds:
  pose_confidence_threshold: 0.5  # ��̬������Ŷ�
  overall_pass_score: 0.6        # ����ͨ������

# ��Ƶ�������
video_processing:
  sample_fps: 10                  # ����֡��
  max_frames: 30                  # �����֡��
  resize_height: 720              # ����߶�
  resize_width: 1280              # ������

# ���ӻ�����
visualization:
  enable_visualization: false     # �Ƿ����ÿ��ӻ�����
  save_pose_images: false         # ������̬�����ͼ��
  save_anomaly_images: false      # �����쳣�����ͼ��
  save_comparison_images: false   # ����ԭʼ֡�������Ա�ͼ
  save_statistics_charts: false   # ����ͳ��ͼ��
  pose_skeleton_color: [0, 255, 0]  # ��̬�Ǽ���ɫ (BGR)
  anomaly_color: [0, 0, 255]     # �쳣��ע��ɫ (BGR)
  confidence_color: [255, 0, 0]  # ���Ŷȱ�ע��ɫ (BGR)
  line_thickness: 2              # ����������ϸ
  font_scale: 0.6                # �����С
  show_confidence: true          # ��ʾ���Ŷ���ֵ
  show_keypoint_names: false     # ��ʾ�ؼ�������

# �������
output:
  save_report: true               # ������ϸ����
  output_dir: "./results"         # ���Ŀ¼
  visualization_dir: "./results/visualizations"  # ���ӻ����Ŀ¼
```

## �����в���

```bash
python video_filter.py <input> [options]

����:
  input              ������Ƶ�ļ���Ŀ¼·��
  -o, --output       ���ժҪ�ļ�·������ѡ��
  -c, --config       �����ļ�·����Ĭ�ϣ�config.yaml��
  -v, --verbose      ��ʾ��ϸ��Ϣ
  --enable-viz       ���ÿ��ӻ�����
  --save-pose        ������̬���ͼ��
  --save-anomaly     �����쳣���ͼ��
  --save-comparison  ����Ա�ͼ��
  --save-charts      ����ͳ��ͼ��
  -h, --help         ��ʾ������Ϣ

ʾ��:
  # ����������Ƶ
  python video_filter.py test_video.mp4 -v
  
  # ��������Ŀ¼
  python video_filter.py video_directory -o results/summary.json
  
  # ʹ���Զ�������
  python video_filter.py test_video.mp4 -c my_config.yaml -v
```

## ���ָ��

### ������̬���
- **�����**: �ɹ���⵽�����֡������
- **���Ŷ�**: ����ƽ�����Ŷ�
- **�쳣����**: ��⵽����̬�쳣����

### ����������
- ͷ��������
- ��С�ȱ������

### �ؽڽǶȼ��
- ��ؽڽǶȼ��
- ϥ�ؽڽǶȼ��

## ���ӻ�����

### ��̬���ӻ�
- **�Ǽܻ���**: ��ͼ���ϻ���������̬�Ǽ�
- **�ؼ����ע**: ��ʾ17���ؼ���λ��
- **�쳣����**: �ò�ͬ��ɫ��ע�쳣�ؼ���
- **���Ŷ���ʾ**: ��ʾ������Ŷ���ֵ

### �쳣�����ӻ�
- **�쳣�����ע**: ������ʾ�쳣�������
- **�쳣����**: ��ʾÿ֡���쳣����
- **�쳣����**: ���ֲ�ͬ���͵��쳣

### �Ա�չʾ
- **ԭʼvs���**: ������ʾԭʼ֡�ͼ����
- **ǰ��Ա�**: ��ʾ����ǰ��Ĳ���
- **�����Ա�**: չʾ�����������

### ͳ��ͼ��
- **����ʷֲ�**: �����ֱ��ͼ
- **���Ŷȷֲ�**: ���Ŷȷֲ�ͼ
- **�쳣����**: �쳣��������ͼ
- **�ۺ�����**: ���ֱ仯����

### ����ʽ�Ǳ��
- **ʵʱ����**: ��̬��ʾ�������
- **��������**: ֧�����š�ɸѡ�Ȳ���
- **��ά��չʾ**: ͬʱ��ʾ���ָ��

## ������

�������������
- ��Ƶ������Ϣ���ֱ��ʡ�֡�ʡ�ʱ���ȣ�
- ��̬�����������ʡ����Ŷȡ��쳣����
- �ۺ����ֺ�ͨ��״̬
- ���ӻ��ļ���ͼ��ͼ���Ǳ�壩

## ע������

1. **ģ������**: �״�����ʱ���Զ�����YOLOv8-poseģ��
2. **GPU����**: ����ʹ��GPU���ټ��
3. **�ڴ�ʹ��**: �������Ƶʱע���ڴ�ʹ�����
4. **������װ**: ȷ����ȷ��װultralytics��

## �ļ��ṹ

```
AIGC_video_filter/
������ detectors/
��   ������ __init__.py
��   ������ pose_anomaly_detector_yolo.py  # YOLO�����
������ video_filter.py                   # ����Ƶ����������ֱ�����У�
������ visualization.py                  # ���ӻ�ģ��
������ config.yaml                       # �����ļ�
������ requirements.txt                  # �����б�
������ results/                          # ������Ŀ¼
��   ������ visualizations/               # ���ӻ����Ŀ¼
��   ��   ������ *_pose_frame_*.jpg        # ��̬���ͼ��
��   ��   ������ *_anomaly_frame_*.jpg     # �쳣���ͼ��
��   ��   ������ *_comparison_frame_*.jpg  # �Ա�ͼ��
��   ��   ������ *_statistics.png          # ͳ��ͼ��
��   ��   ������ *_dashboard.html          # ����ʽ�Ǳ��
��   ������ summary.json                  # ����ժҪ
������ README.md                         # ˵���ĵ�
```

## ��˵��

���ԭ�汾�����汾���������¼򻯣�

1. **��ʹ��YOLO���**: �Ƴ���MediaPipe��ͼ��������⡢ʱ��һ���Լ���ģ��
2. **��������**: �Ƴ��˲���Ҫ��������
3. **�򻯴���**: ɾ��������ļ���߼�
4. **רע��̬**: רע��������̬�쳣��⹦��

�����ļ�ʹ�ô������������ά�������ף�ͬʱ�����˺��ĵļ�⹦�ܡ�