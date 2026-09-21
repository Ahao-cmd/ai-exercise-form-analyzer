import cv2

print("AI Exercise Form Analyzer started.")

video_path = "videos/test_squat.mov"

video = cv2.VideoCapture(video_path)

if video.isOpened():
    print("Video opened successfully.")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    duration = total_frames / fps

    print("Width:", width)
    print("Height:", height)
    print("FPS:", fps)
    print("Total frames:", total_frames)
    print("Duration:", round(duration, 2), "seconds")

    success, frame = video.read()

    if success:
        print("First frame read successfully.")
        print("Frame shape:", frame.shape)

        cv2.imwrite("first_frame.jpg", frame)
        print("First frame saved")
        
    else:
        print("Failed to read first frame.")

    
else:
    print("Failed to open video.")




video.release()

