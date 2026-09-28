import cv2
import math
import mediapipe as mp
import matplotlib.pyplot as plt
import statistics


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

def calculate_torso_lean(shoulder, hip):
    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]

    angle = math.degrees(
        math.atan2(abs(dx), abs(dy))
    )

    return angle

video_path = "videos/test_squat.mov"
model_path = "models/pose_landmarker_full.task"

video = cv2.VideoCapture(video_path)

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_width = 1080
output_height = 1920

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output_video = cv2.VideoWriter(
    "output_squat_analysis.mp4",
    fourcc,
    fps,
    (output_width, output_height)
)

base_options = mp.tasks.BaseOptions(
    model_asset_path=model_path
)

options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO
)

knee_angles = []
times = [] 
hip_angles = []
rep_summaries = []

rep_in_progress = False
reached_bottom = False
rep_start_time = None

current_rep_knee_angles = []
current_rep_hip_angles = []
current_rep_torso_leans = []
torso_leans = []
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

            shoulder = to_pixel(landmarks[11], width, height)
            hip = to_pixel(landmarks[23], width, height)
            knee = to_pixel(landmarks[25], width, height)
            ankle = to_pixel(landmarks[27], width, height)

            knee_angle = calculate_angle(
                hip,
                knee,
                ankle
            )

            hip_angle = calculate_angle(
                shoulder,
                hip,
                knee
            )

            torso_lean = calculate_torso_lean(
                shoulder,
                hip
            )

            current_time = frame_number / fps


            # Start a new repetition
            if not rep_in_progress and knee_angle < 150:
                rep_in_progress = True
                reached_bottom = False
                rep_start_time = current_time
                stage = "down"

                current_rep_knee_angles = []
                current_rep_hip_angles = []
                current_rep_torso_leans = []


            # Record data during the repetition
            if rep_in_progress:
                current_rep_knee_angles.append(knee_angle)
                current_rep_hip_angles.append(hip_angle)
                current_rep_torso_leans.append(torso_lean)

                # Confirm that the squat reached the bottom
                if knee_angle < 100:
                    reached_bottom = True

                # Complete the repetition after standing back up
                if reached_bottom and knee_angle > 150:
                    rep_count += 1
                    stage = "standing"

                    rep_end_time = current_time
                    rep_duration = rep_end_time - rep_start_time

                    rep_summary = {
                        "rep": rep_count,
                        "duration": rep_duration,
                        "min_knee_angle": min(current_rep_knee_angles),
                        "min_hip_angle": min(current_rep_hip_angles),
                        "max_torso_lean": max(current_rep_torso_leans)
                    }

                    # Cancel an incomplete repetition
                    if rep_in_progress and not reached_bottom and knee_angle > 150:
                        rep_in_progress = False
                        rep_start_time = None

                        current_rep_knee_angles = []
                        current_rep_hip_angles = []
                        current_rep_torso_leans = []

                        stage = "standing"

                    rep_summaries.append(rep_summary)

                    print("Rep completed:", rep_count)

                    rep_in_progress = False
                    reached_bottom = False
                    rep_start_time = None


            # Draw skeleton
            shoulder_point = (int(shoulder[0]), int(shoulder[1]))
            hip_point = (int(hip[0]), int(hip[1]))
            knee_point = (int(knee[0]), int(knee[1]))
            ankle_point = (int(ankle[0]), int(ankle[1]))

            cv2.line(frame, shoulder_point, hip_point, (255, 0, 0), 12)
            cv2.line(frame, hip_point, knee_point, (255, 0, 0), 12)
            cv2.line(frame, knee_point, ankle_point, (255, 0, 0), 12)

            cv2.circle(frame, shoulder_point, 20, (0, 255, 0), -1)
            cv2.circle(frame, hip_point, 20, (0, 255, 0), -1)
            cv2.circle(frame, knee_point, 20, (0, 255, 0), -1)
            cv2.circle(frame, ankle_point, 20, (0, 255, 0), -1)


            # Display analysis
            cv2.putText(
                frame,
                f"Knee Angle: {knee_angle:.1f}",
                (80, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (0, 0, 255),
                6
            )

            cv2.putText(
                frame,
                f"Hip Angle: {hip_angle:.1f}",
                (80, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (255, 0, 255),
                6
            )

            cv2.putText(
                frame,
                f"Torso Lean: {torso_lean:.1f}",
                (80, 450),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (255, 255, 0),
                6
            )

            cv2.putText(
                frame,
                f"Reps: {rep_count}",
                (80, 650),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (0, 255, 0),
                6
            )

            cv2.putText(
                frame,
                f"Stage: {stage.upper()}",
                (80, 800),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (0, 255, 255),
                6
            )


            # Save frame data
            knee_angles.append(knee_angle)
            hip_angles.append(hip_angle)
            torso_leans.append(torso_lean)
            times.append(current_time)


        # Always write the frame to the output video
        output_frame = cv2.resize(
            frame,
            (output_width, output_height)
        )
    
        output_video.write(output_frame)

        frame_number += 1

video.release()
output_video.release()


print("Annotated video saved as output_squat_analysis.mp4")
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

print("\nPer-rep analysis:")


print("\nPer-rep analysis:")

if rep_summaries:
    durations = [rep["duration"] for rep in rep_summaries]
    min_knees = [rep["min_knee_angle"] for rep in rep_summaries]
    min_hips = [rep["min_hip_angle"] for rep in rep_summaries]
    max_torso_leans = [rep["max_torso_lean"] for rep in rep_summaries]

    avg_duration = statistics.mean(durations)
    avg_knee = statistics.mean(min_knees)
    avg_hip = statistics.mean(min_hips)
    avg_torso = statistics.mean(max_torso_leans)

    if len(rep_summaries) >= 2:
        duration_std = statistics.stdev(durations)
        knee_std = statistics.stdev(min_knees)
        hip_std = statistics.stdev(min_hips)
        torso_std = statistics.stdev(max_torso_leans)
    else:
        duration_std = 0
        knee_std = 0
        hip_std = 0
        torso_std = 0

duration_cv = (duration_std / avg_duration) * 100
knee_cv = (knee_std / avg_knee) * 100
hip_cv = (hip_std / avg_hip) * 100
torso_cv = (torso_std / avg_torso) * 100

most_different_duration_rep = max(
    rep_summaries,
    key=lambda rep: abs(rep["duration"] - avg_duration)
)

most_different_knee_rep = max(
    rep_summaries,
    key=lambda rep: abs(rep["min_knee_angle"] - avg_knee)
)

print(
    "\nMost different tempo:",
    f"Rep {most_different_duration_rep['rep']}"
)

print(
    "Most different depth:",
    f"Rep {most_different_knee_rep['rep']}"
)

print("\nConsistency percentages:")
print("Tempo CV:", round(duration_cv, 1), "%")
print("Knee depth CV:", round(knee_cv, 1), "%")
print("Hip angle CV:", round(hip_cv, 1), "%")
print("Torso lean CV:", round(torso_cv, 1), "%")

print("\nSet summary:")
print("Average duration:", round(avg_duration, 2), "seconds")
print("Average minimum knee angle:", round(avg_knee, 1), "degrees")
print("Average minimum hip angle:", round(avg_hip, 1), "degrees")
print("Average maximum torso lean:", round(avg_torso, 1), "degrees")

print("\nConsistency:")
print("Duration standard deviation:", round(duration_std, 2))
print("Knee angle standard deviation:", round(knee_std, 2))
print("Hip angle standard deviation:", round(hip_std, 2))
print("Torso lean standard deviation:", round(torso_std, 2))

for rep in rep_summaries:
    print(
        f"Rep {rep['rep']}: "
        f"Duration = {rep['duration']:.2f}s, "
        f"Min Knee = {rep['min_knee_angle']:.1f}°, "
        f"Min Hip = {rep['min_hip_angle']:.1f}°, "
        f"Max Torso Lean = {rep['max_torso_lean']:.1f}°"
    )

else:
    print("No completed repetitions available for analysis.")