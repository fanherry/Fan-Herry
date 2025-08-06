from ultralytics import YOLO
import os
import cv2

# 文件配置
MODEL_PT = '../yolov8s.pt'  # 官方格式模型
name=input("name:")
VIDEO_FILE = f'{name}.mp4'  # 待检测视频
OUTPUT_FILE = 'JJ.mp4'  # 输出视频路径


def load_yolo_model():
    """智能加载YOLO模型（与之前相同）"""
    if os.path.exists(MODEL_PT):
        print("检测到yolov8s.pt，正在加载官方模型...")
        try:
            return YOLO(MODEL_PT)
        except Exception as e:
            print(f"加载.pt文件失败: {e}")
    return YOLO('yolov8s.pt')  # 备选自动下载


def process_video(model, video_path, output_path):
    """处理视频并保存结果"""
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return

    # 获取视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"开始处理视频: {video_path} (FPS: {fps}, 分辨率: {width}x{height})")

    # 逐帧处理
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 执行检测（YOLO使用RGB格式）
        results = model(frame, verbose=False)  # verbose=False关闭控制台输出

        # 绘制检测结果
        annotated_frame = results[0].plot()  # 获取带标注的帧

        # 转换颜色空间（OpenCV需要BGR）
        annotated_frame_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # 写入输出视频
        out.write(annotated_frame_bgr)

        frame_count += 1
        if frame_count % 30 == 0:  # 每30帧打印进度
            print(f"已处理 {frame_count} 帧...")

    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频处理完成! 结果已保存到: {output_path}")


if __name__ == '__main__':
    # 1. 加载模型
    model = load_yolo_model()

    # 2. 检查视频文件
    if not os.path.exists(VIDEO_FILE):
        exit(f"错误: 视频文件 {VIDEO_FILE} 不存在")

    # 3. 处理视频
    process_video(model, VIDEO_FILE, OUTPUT_FILE)

    # 4. 可选：在窗口中实时显示（调试用）
    # play_result_video(OUTPUT_FILE)  # 可以添加这个函数