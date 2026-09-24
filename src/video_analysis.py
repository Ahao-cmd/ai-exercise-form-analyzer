import cv2
import math
import mediapipe as mp
import matplotlib.pyplot as plt



def to_pixel(landmark, width, height):
    x = landmark.x * width
    y = landmark.y * height
    return (x, y)


def calculate_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]

    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return math.degrees(math.acos(cosine_angle))

video_path = "videos/test_squat.mov"
model_path = "models/pose_landmarker_full.task"

video = cv2.VideoCapture(video_path)

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

base_options = mp.tasks.BaseOptions(
    model_asset_path=model_path
)

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)

knee_angles = []
times = [] 
frame_number = 0
rep_count = 0
stage = "standing"

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = video.read()

        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        timestamp_ms = int((frame_number / fps) * 1000)

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]

            hip = to_pixel(landmarks[23], width, height)
            knee = to_pixel(landmarks[25], width, height)
            ankle = to_pixel(landmarks[27], width, height)

            knee_angle = calculate_angle(
                hip,
                knee,
                ankle
            )
            if knee_angle < 100 and stage == "standing":
                stage = "down"

            if knee_angle > 150 and stage == "down":
                stage = "standing"
                rep_count += 1
                print("Rep completed:", rep_count)  

            knee_angles.append(knee_angle)
            times.append(frame_number / fps)

        frame_number += 1

video.release()

print("Frames processed:", frame_number)
print("Pose frames detected:", len(knee_angles))

if knee_angles:
    print("Maximum knee angle:", round(max(knee_angles), 2))
    print("Minimum knee angle:", round(min(knee_angles), 2))

    plt.figure(figsize=(12, 6))

    plt.plot(times, knee_angles)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Knee Angle (degrees)")
    plt.title("Knee Angle During Squats")

    plt.grid(True)

    plt.savefig("knee_angle_plot.png")
    plt.close()

    print("Knee angle plot saved.")

print("Total squat repetitions:", rep_count)