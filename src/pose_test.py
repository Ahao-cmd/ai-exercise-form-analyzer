import mediapipe as mp


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
    left_landmarks = {
    "shoulder": 11,
    "hip": 23,
    "knee": 25,
    "ankle": 27
}

right_landmarks = {
    "shoulder": 12,
    "hip": 24,
    "knee": 26,
    "ankle": 28
}

left_visibility = (
    landmarks[11].visibility
    + landmarks[23].visibility
    + landmarks[25].visibility
    + landmarks[27].visibility
) / 4

right_visibility = (
    landmarks[12].visibility
    + landmarks[24].visibility
    + landmarks[26].visibility
    + landmarks[28].visibility
) / 4

if left_visibility > right_visibility:
    selected_side = "left"
    selected_landmarks = left_landmarks
else:
    selected_side = "right"
    selected_landmarks = right_landmarks

print("Selected side:", selected_side)
print("Left visibility:", round(left_visibility, 3))
print("Right visibility:", round(right_visibility, 3))

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