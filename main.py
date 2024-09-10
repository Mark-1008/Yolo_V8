import cv2
from ultralytics import YOLO
from data_processing.data_loader import load_video

# 加载YOLOv8模型
model = YOLO('yolov8n.pt')  # 使用预训练模型 yolov8n.pt

def run_detection(video_path):
    cap = load_video(video_path)
    if not cap:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 使用YOLOv8进行目标检测
        results = model(frame)

        # 绘制检测结果
        annotated_frame = results[0].plot()  # 标注检测框在图像上

        # 显示视频帧
        cv2.imshow('YOLOv8 Detection', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# 按间距中的绿色按钮以运行脚本。
if __name__ == "__main__":
    video_path = 'path_to_your_video.mp4'  # 替换为你的视频路径
    run_detection(video_path)
