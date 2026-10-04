import cv2
import math
import mediapipe as mp
import matplotlib.pyplot as plt
import statistics
import json
import csv
import os
import argparse
from pathlib import Path

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

# Command-line arguments
parser = argparse.ArgumentParser(
    description="AI Exercise Form Analyzer"
)

parser.add_argument(
    "--video",
    type=str,
    default="videos/test_squat.mov",
    help="Path to the input exercise video"
)

args = parser.parse_args()

video_path = args.video
model_path = "models/pose_landmarker_full.task"
# Minimum required landmark visibility
MIN_VISIBILITY = 0.5
# Landmark position jump detection
ANKLE_JUMP_THRESHOLD = 0.18

# Create a separate output folder for each video
video_name = Path(video_path).stem

output_dir = os.path.join(
    "outputs",
    video_name
)

os.makedirs(output_dir, exist_ok=True)

output_video_path = os.path.join(
    output_dir,
    "squat_analysis.mp4"
)

plot_path = os.path.join(
    output_dir,
    "knee_angle_plot.png"
)

json_report_path = os.path.join(
    output_dir,
    "analysis_report.json"
)

csv_report_path = os.path.join(
    output_dir,
    "rep_analysis.csv"
)

video = cv2.VideoCapture(video_path)
if not video.isOpened():
    raise FileNotFoundError(
        f"Cannot open input video: {video_path}"
    )

fps = video.get(cv2.CAP_PROP_FPS)
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_width = 1080
output_height = 1920

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output_video = cv2.VideoWriter(
    output_video_path,
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
# Previous frame ankle information
previous_ankle_point = None
previous_lower_leg_length = None

# Number of suspicious ankle jumps
ankle_jump_frames = 0
# Store ankle jump measurements for evaluation
ankle_jump_records = []
low_visibility_frames = 0
missing_pose_frames = 0
previous_knee_angle = None
current_min_knee_angle = None
bottom_time = None
# Automatically selected body side
selected_side = None
# Require a confirmed standing position before starting a rep
ready_for_rep = False
standing_frames = 0

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

            # Automatically select the more visible body side
            if selected_side is None:

                left_indices = [11, 23, 25, 27]
                right_indices = [12, 24, 26, 28]

                # Calculate average visibility
                left_visibility = sum(
                    landmarks[i].visibility
                    for i in left_indices
                ) / 4

                right_visibility = sum(
                    landmarks[i].visibility
                    for i in right_indices
                ) / 4

                # Select the more visible side
                if left_visibility >= right_visibility:
                    selected_side = "left"
                else:
                    selected_side = "right"

                print("\nAutomatic side selection:")
                print("Left visibility:", round(left_visibility, 3))
                print("Right visibility:", round(right_visibility, 3))
                print("Selected side:", selected_side.upper())

                # Warn when the first-frame comparison is ambiguous
                if (
                    abs(left_visibility - right_visibility) < 0.1
                    or max(left_visibility, right_visibility) < 0.5
                ):
                    print("Warning: Side selection may be unreliable.")

            # Use the selected side for the entire video
            if selected_side == "left":
                shoulder_id = 11
                hip_id = 23
                knee_id = 25
                ankle_id = 27

            else:
                shoulder_id = 12
                hip_id = 24
                knee_id = 26
                ankle_id = 28

            # Check landmark visibility
            required_indices = [
                shoulder_id,
                hip_id,
                knee_id,
                ankle_id
            ]

            visibilities = [
                landmarks[i].visibility
                for i in required_indices
            ]

            if min(visibilities) < MIN_VISIBILITY:
                low_visibility_frames += 1
                                # Identify which landmarks have low visibility
                joint_names = [
                    "Shoulder",
                    "Hip",
                    "Knee",
                    "Ankle"
                ]

                low_joints = []

                for name, visibility in zip(joint_names, visibilities):
                    if visibility < MIN_VISIBILITY:
                        low_joints.append(
                            f"{name} = {visibility:.2f}"
                        )

                print(
                    f"\nTracking lost at {frame_number / fps:.2f}s"
                )

                print("Selected side:", selected_side.upper())
                print("Low visibility joints:", ", ".join(low_joints))

                if rep_in_progress:
                    print("Incomplete rep discarded: low visibility")

                # Reset the current repetition
                rep_in_progress = False
                reached_bottom = False
                rep_start_time = None
                current_min_knee_angle = None
                bottom_time = None
                previous_knee_angle = None
                previous_ankle_point = None
                previous_lower_leg_length = None
                ready_for_rep = False
                standing_frames = 0

                current_rep_knee_angles = []
                current_rep_hip_angles = []
                current_rep_torso_leans = []

                stage = "tracking_lost"

                cv2.putText(
                    frame,
                    "Tracking lost: low visibility",
                    (80, 950),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    (0, 0, 255),
                    5
                )

                output_frame = cv2.resize(
                    frame,
                    (output_width, output_height)
                )

                output_video.write(output_frame)

                frame_number += 1
                continue

            # Tracking recovered
            if stage == "tracking_lost":
                stage = "standing"

            shoulder = to_pixel(landmarks[shoulder_id], width, height)
            hip = to_pixel(landmarks[hip_id], width, height)
            knee = to_pixel(landmarks[knee_id], width, height)
            ankle = to_pixel(landmarks[ankle_id], width, height)

            # Calculate current lower-leg length
            current_lower_leg_length = math.dist(
                knee,
                ankle
            )

            # Detect suspicious ankle position jumps
            if (
                previous_ankle_point is not None
                and previous_lower_leg_length is not None
            ):

                # Distance moved by ankle between two frames
                ankle_movement = math.dist(
                    ankle,
                    previous_ankle_point
                )

                # Normalize by previous lower-leg length
                ankle_jump_ratio = ankle_movement / max(
                    previous_lower_leg_length,
                    1.0
                )
                ankle_jump_records.append({
                    "time": frame_number / fps,
                    "ratio": ankle_jump_ratio,
                    "visibility": landmarks[ankle_id].visibility
                })

                # Check whether movement exceeds the threshold
                if ankle_jump_ratio > ANKLE_JUMP_THRESHOLD:

                    ankle_jump_frames += 1

                    print(
                        f"\nWarning: Suspicious ankle jump "
                        f"at {frame_number / fps:.2f}s"
                    )

                    print(
                        "Selected side:",
                        selected_side.upper()
                    )

                    print(
                        "Ankle jump ratio:",
                        round(ankle_jump_ratio, 3)
                    )

                    print(
                        "Ankle visibility:",
                        round(landmarks[ankle_id].visibility, 3)
                    )

            # Save current ankle information for the next frame
            previous_ankle_point = ankle
            previous_lower_leg_length = current_lower_leg_length

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


                        # Calculate knee angle change from the previous frame
            angle_change = 0.0

            if previous_knee_angle is not None:
                angle_change = knee_angle - previous_knee_angle

            # Confirm standing position before allowing a new rep
            if not rep_in_progress:
                if knee_angle >= 155:
                    standing_frames += 1

                    if standing_frames >= 3:
                        ready_for_rep = True
                else:
                    standing_frames = 0

            # Start a new repetition
            if ready_for_rep and not rep_in_progress and knee_angle < 150:                
                rep_in_progress = True
                reached_bottom = False
                ready_for_rep = False
                standing_frames = 0
                rep_start_time = current_time
                stage = "descending"

                current_min_knee_angle = knee_angle
                bottom_time = current_time

                current_rep_knee_angles = []
                current_rep_hip_angles = []
                current_rep_torso_leans = []


            # Record data during the repetition
            if rep_in_progress:
                current_rep_knee_angles.append(knee_angle)
                current_rep_hip_angles.append(hip_angle)
                current_rep_torso_leans.append(torso_lean)

                # Track the lowest knee angle and its time
                if knee_angle < current_min_knee_angle:
                    current_min_knee_angle = knee_angle
                    bottom_time = current_time


                # Confirm sufficient squat depth
                if knee_angle < 100:
                    reached_bottom = True


                # Determine movement stage
                if not reached_bottom:
                    if angle_change < -0.3:
                        stage = "descending"

                else:
                    if angle_change > 0.3:
                        stage = "ascending"

                    elif abs(angle_change) <= 0.3 and knee_angle < 110:
                        stage = "bottom"


                # Complete the repetition
                if reached_bottom and knee_angle > 150:
                    rep_count += 1
                    stage = "standing"

                    rep_end_time = current_time

                    rep_duration = rep_end_time - rep_start_time
                    descent_duration = bottom_time - rep_start_time
                    ascent_duration = rep_end_time - bottom_time

                    if ascent_duration > 0:
                        tempo_ratio = descent_duration / ascent_duration
                    else:
                        tempo_ratio = 0

                    rep_summary = {
                        "rep": rep_count,
                        "duration": rep_duration,
                        "descent_duration": descent_duration,
                        "ascent_duration": ascent_duration,
                        "tempo_ratio": tempo_ratio,
                        "min_knee_angle": min(current_rep_knee_angles),
                        "min_hip_angle": min(current_rep_hip_angles),
                        "max_torso_lean": max(current_rep_torso_leans)
                    }

                    rep_summaries.append(rep_summary)

                    print("Rep completed:", rep_count)

                    rep_in_progress = False
                    reached_bottom = False
                    rep_start_time = None
                    current_min_knee_angle = None
                    bottom_time = None


                # Cancel an incomplete repetition
                elif not reached_bottom and knee_angle > 150:
                    rep_in_progress = False
                    rep_start_time = None
                    current_min_knee_angle = None
                    bottom_time = None
                    stage = "standing"

                    current_rep_knee_angles = []
                    current_rep_hip_angles = []
                    current_rep_torso_leans = []


            # Save current angle for the next frame
            previous_knee_angle = knee_angle


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

        else:
            missing_pose_frames += 1

            if rep_in_progress:
                print("Incomplete rep discarded: pose missing")

            rep_in_progress = False
            reached_bottom = False
            rep_start_time = None
            current_min_knee_angle = None
            bottom_time = None
            previous_knee_angle = None
            previous_ankle_point = None
            previous_lower_leg_length = None
            current_rep_knee_angles = []
            current_rep_hip_angles = []
            current_rep_torso_leans = []

            stage = "tracking_lost"

        # Always write the frame to the output video
        output_frame = cv2.resize(
            frame,
            (output_width, output_height)
        )
    
        output_video.write(output_frame)

        frame_number += 1

video.release()
output_video.release()


print("Annotated video saved to:", output_video_path)
print("Frames processed:", frame_number)
print("Valid analyzed frames:", len(knee_angles))
print("Pose detected frames:", frame_number - missing_pose_frames)

if knee_angles:
    print("Maximum knee angle:", round(max(knee_angles), 2))
    print("Minimum knee angle:", round(min(knee_angles), 2))

    plt.figure(figsize=(12, 6))

    plt.plot(times, knee_angles)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Knee Angle (degrees)")
    plt.title("Knee Angle During Squats")

    plt.grid(True)

    plt.savefig(plot_path)
    plt.close()

    print("Knee angle plot saved to:", plot_path)

print("Total squat repetitions:", rep_count)
print("\nTracking validation:")
print("Low visibility frames:", low_visibility_frames)
print("Missing pose frames:", missing_pose_frames)
print("Suspicious ankle jump frames:", ankle_jump_frames)

# Display the five largest observed ankle jumps
if ankle_jump_records:

    top_jumps = sorted(
        ankle_jump_records,
        key=lambda item: item["ratio"],
        reverse=True
    )[:5]

    print("\nTop 5 ankle position changes:")

    for jump in top_jumps:
        print(
            f"Time = {jump['time']:.2f}s, "
            f"Jump Ratio = {jump['ratio']:.3f}, "
            f"Visibility = {jump['visibility']:.3f}"
        )

print("\nPer-rep analysis:")

if rep_summaries:
    durations = [rep["duration"] for rep in rep_summaries]
    min_knees = [rep["min_knee_angle"] for rep in rep_summaries]
    min_hips = [rep["min_hip_angle"] for rep in rep_summaries]
    max_torso_leans = [rep["max_torso_lean"] for rep in rep_summaries]

    descent_durations = [
        rep["descent_duration"]
        for rep in rep_summaries
    ]

    ascent_durations = [
        rep["ascent_duration"]
        for rep in rep_summaries
    ]

    tempo_ratios = [
        rep["tempo_ratio"]
        for rep in rep_summaries
    ]

    # Calculate averages
    avg_descent = statistics.mean(descent_durations)
    avg_ascent = statistics.mean(ascent_durations)
    avg_tempo_ratio = statistics.mean(tempo_ratios)
    avg_duration = statistics.mean(durations)
    avg_knee = statistics.mean(min_knees)
    avg_hip = statistics.mean(min_hips)
    avg_torso = statistics.mean(max_torso_leans)

    # Calculate standard deviations
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

    # Calculate coefficient of variation
    duration_cv = (duration_std / avg_duration) * 100
    knee_cv = (knee_std / avg_knee) * 100
    hip_cv = (hip_std / avg_hip) * 100
    torso_cv = (torso_std / avg_torso) * 100

    # Find repetitions that differ most from the average
    most_different_duration_rep = max(
        rep_summaries,
        key=lambda rep: abs(rep["duration"] - avg_duration)
    )

    most_different_knee_rep = max(
        rep_summaries,
        key=lambda rep: abs(rep["min_knee_angle"] - avg_knee)
    )

    # Print per-rep results
    for rep in rep_summaries:
        print(
            f"Rep {rep['rep']}: "
            f"Duration = {rep['duration']:.2f}s, "
            f"Descent = {rep['descent_duration']:.2f}s, "
            f"Ascent = {rep['ascent_duration']:.2f}s, "
            f"Tempo Ratio = {rep['tempo_ratio']:.2f}:1, "
            f"Min Knee = {rep['min_knee_angle']:.1f}°, "
            f"Min Hip = {rep['min_hip_angle']:.1f}°, "
            f"Max Torso Lean = {rep['max_torso_lean']:.1f}°"
        )

    # Print set summary
    print("\nSet summary:")
    print("Average duration:", round(avg_duration, 2), "seconds")
    print("Average minimum knee angle:", round(avg_knee, 1), "degrees")
    print("Average minimum hip angle:", round(avg_hip, 1), "degrees")
    print("Average maximum torso lean:", round(avg_torso, 1), "degrees")

    # Print consistency statistics
    print("\nConsistency:")
    print("Duration standard deviation:", round(duration_std, 2))
    print("Knee angle standard deviation:", round(knee_std, 2))
    print("Hip angle standard deviation:", round(hip_std, 2))
    print("Torso lean standard deviation:", round(torso_std, 2))

    print("\nConsistency percentages:")
    print("Tempo CV:", round(duration_cv, 1), "%")
    print("Knee depth CV:", round(knee_cv, 1), "%")
    print("Hip angle CV:", round(hip_cv, 1), "%")
    print("Torso lean CV:", round(torso_cv, 1), "%")

    print(
        "\nMost different tempo:",
        f"Rep {most_different_duration_rep['rep']}"
    )

    print(
        "Most different depth:",
        f"Rep {most_different_knee_rep['rep']}"
    )

    print(
        "Average descent duration:",
        round(avg_descent, 2),
        "seconds"
    )

    print(
        "Average ascent duration:",
        round(avg_ascent, 2),
        "seconds"
    )

    print(
        "Average tempo ratio:",
        round(avg_tempo_ratio, 2),
        ":1"
    )

    # Create structured JSON report
    analysis_report = {
        "total_reps": rep_count,

        "set_summary": {
            "average_duration": round(avg_duration, 2),
            "average_min_knee_angle": round(avg_knee, 1),
            "average_min_hip_angle": round(avg_hip, 1),
            "average_max_torso_lean": round(avg_torso, 1),
            "average_descent_duration": round(avg_descent, 2),
            "average_ascent_duration": round(avg_ascent, 2),
            "average_tempo_ratio": round(avg_tempo_ratio, 2),
        },

        "consistency": {
            "tempo_cv_percent": round(duration_cv, 1),
            "knee_depth_cv_percent": round(knee_cv, 1),
            "hip_angle_cv_percent": round(hip_cv, 1),
            "torso_lean_cv_percent": round(torso_cv, 1)
        },

        "most_different_reps": {
            "tempo": most_different_duration_rep["rep"],
            "depth": most_different_knee_rep["rep"]
        },

        "repetitions": rep_summaries
    }

    # Save JSON report
    with open(json_report_path, "w") as file:
        json.dump(
            analysis_report,
            file,
            indent=4
        )

    print("\nAnalysis report saved to:", json_report_path)
    with open(csv_report_path, "w", newline="") as csv_file:
        fieldnames = [
            "rep",
            "duration",
            "descent_duration",
            "ascent_duration",
            "tempo_ratio",
            "min_knee_angle",
            "min_hip_angle",
            "max_torso_lean"
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for rep in rep_summaries:
            writer.writerow({
                "rep": rep["rep"],
                "duration": round(rep["duration"], 2),
                "descent_duration": round(rep["descent_duration"], 2),
                "ascent_duration": round(rep["ascent_duration"], 2),
                "tempo_ratio": round(rep["tempo_ratio"], 2),
                "min_knee_angle": round(rep["min_knee_angle"], 1),
                "min_hip_angle": round(rep["min_hip_angle"], 1),
                "max_torso_lean": round(rep["max_torso_lean"], 1)
            })

    print("CSV report saved to:", csv_report_path)
else:
    print("No completed repetitions available for analysis.")