# AIGC��Ƶ����ɸѡ����YOLO�汾��

����YOLO��AIGC��Ƶ����ɸѡ���ߣ�רע��������̬�쳣��⡣

## �����ص�

- **��ʹ��YOLO���**: �򻯼ܹ�����ʹ��YOLO����������̬���
- **��Ч���**: ����YOLOv8-poseģ�ͣ�����ٶȿ�
- **��̬�쳣���**: �����������쳣���ؽڽǶ��쳣��
- **��������**: ֧��������Ƶ��������
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

### 4. ��̽ӿ�ʹ��

```python
from video_filter import VideoQualityFilter

# ��ʼ��ɸѡ��
filter = VideoQualityFilter(config_path='config.yaml')

# ������Ƶ
result = filter.evaluate_video('test_video.mp4', verbose=True)
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

# �������
output:
  save_report: true               # ������ϸ����
  output_dir: "./results"         # ���Ŀ¼
```

## �����в���

```bash
python video_filter.py <input> [options]

����:
  input              ������Ƶ�ļ���Ŀ¼·��
  -o, --output       ���ժҪ�ļ�·������ѡ��
  -c, --config       �����ļ�·����Ĭ�ϣ�config.yaml��
  -v, --verbose      ��ʾ��ϸ��Ϣ
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

## ������

�������������
- ��Ƶ������Ϣ���ֱ��ʡ�֡�ʡ�ʱ���ȣ�
- ��̬�����������ʡ����Ŷȡ��쳣����
- �ۺ����ֺ�ͨ��״̬

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
������ config.yaml                       # �����ļ�
������ requirements.txt                  # �����б�
������ README.md                         # ˵���ĵ�
```

## ��˵��

���ԭ�汾�����汾���������¼򻯣�

1. **��ʹ��YOLO���**: �Ƴ���MediaPipe��ͼ��������⡢ʱ��һ���Լ���ģ��
2. **��������**: �Ƴ��˲���Ҫ��������
3. **�򻯴���**: ɾ��������ļ���߼�
4. **רע��̬**: רע��������̬�쳣��⹦��

�����ļ�ʹ�ô������������ά�������ף�ͬʱ�����˺��ĵļ�⹦�ܡ�