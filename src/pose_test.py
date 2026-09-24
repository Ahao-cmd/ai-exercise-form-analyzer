import mediapipe as mp
import math
import cv2

def to_pixel(landmark, width, height):
    x = landmark.x * width
    y = landmark.y * height
    return (x, y)


def calculate_angle(a, b, c):#hip, knee, ankle三个点knee作为顶点读取角度
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]

    magnitude_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2)
    magnitude_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    cosine_angle = dot_product / (magnitude_ba * magnitude_bc)

    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    angle = math.degrees(math.acos(cosine_angle))

    return angle


image_path = "first_frame.jpg"
model_path = "models/pose_landmarker_full.task"


base_options = mp.tasks.BaseOptions(
    model_asset_path=model_path
)

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)

image = mp.Image.create_from_file(image_path)

with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(image)


if result.pose_landmarks:
    landmarks = result.pose_landmarks[0]

    print("Pose detected successfully.")
    print("Number of landmarks:", len(landmarks))

    selected_landmarks = {
        "shoulder": 11,
        "hip": 23,
        "knee": 25,
        "ankle": 27
    }

    print("Using left-side landmarks")

    image_width = image.width
    image_height = image.height

    hip = to_pixel(
        landmarks[selected_landmarks["hip"]],
        image_width,
        image_height
    )

    knee = to_pixel(
        landmarks[selected_landmarks["knee"]],
        image_width,
        image_height
    )

    ankle = to_pixel(
        landmarks[selected_landmarks["ankle"]],
        image_width,
        image_height
    )

    knee_angle = calculate_angle(hip, knee, ankle)
    frame = cv2.imread(image_path)

    hip_point = (int(hip[0]), int(hip[1]))
    knee_point = (int(knee[0]), int(knee[1]))
    ankle_point = (int(ankle[0]), int(ankle[1]))

    cv2.circle(frame, hip_point, 15, (0, 255, 0), -1)
    cv2.circle(frame, knee_point, 15, (0, 255, 0), -1)
    cv2.circle(frame, ankle_point, 15, (0, 255, 0), -1)

    cv2.line(frame, hip_point, knee_point, (255, 0, 0), 8)
    cv2.line(frame, knee_point, ankle_point, (255, 0, 0), 8)

    cv2.putText(
    frame,
    f"{knee_angle:.1f} deg",
    knee_point,
    cv2.FONT_HERSHEY_SIMPLEX,
    2,
    (0, 0, 255),
    5
    )

    cv2.imwrite("pose_result.jpg", frame)

    print("Pose result image saved.")
    

    print("Knee angle:", round(knee_angle, 2), "degrees")

    for name, index in selected_landmarks.items():
        landmark = landmarks[index]

        print(
            name,
            "x =", round(landmark.x, 3),
            "y =", round(landmark.y, 3),
            "visibility =", round(landmark.visibility, 3)
        )

else:
    print("No pose detected.")