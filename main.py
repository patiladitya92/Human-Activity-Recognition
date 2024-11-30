import cv2
import numpy as np
import mediapipe as mp

# Function to extract motion-based features
def extract_motion_features(frame, prev_frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_diff = cv2.absdiff(gray, prev_gray)
    _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    motion_intensity = np.count_nonzero(motion_mask)
    return motion_intensity

# Function to process live camera feed or uploaded video file
def process_feed(mode='live', file_path=None):
    if mode == 'live':
        cap = cv2.VideoCapture(0)
    elif mode == 'video':
        cap = cv2.VideoCapture(file_path)
    else:
        print("Invalid mode. Please choose 'live' or 'video'.")
        return

    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    prev_frame = None
    activities = {1: 'Walking', 2: 'Running', 3: 'Crawling', 4: 'Sitting', 5: 'Standing', 6: 'Eating', 7: 'Dancing', 8: 'Jumping', 9: 'Stretching'}

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if prev_frame is None:
            prev_frame = frame
            continue
        
        motion_intensity = extract_motion_features(frame, prev_frame)

        if motion_intensity < 1500:
            activity_label = 5  # Sitting
        elif motion_intensity < 5000:
            activity_label = 3  # Crawling
        elif motion_intensity < 10000:
            activity_label = 1  # Walking
        elif motion_intensity < 20000:
            activity_label = 5  # Standing
        elif motion_intensity < 30000:
            activity_label = 6  # Eating
        elif motion_intensity < 50000:
            activity_label = 8  # Jumping
        else:
            activity_label = 7  # Dancing

        print("Detected Activity:", activities[activity_label])

        # Pose estimation
        img = cv2.resize(frame, (700, 600))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)

        # Display pose estimation without background
        pose_img = np.zeros_like(img)
        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                pose_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_draw.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_draw.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            cv2.imshow('Pose Estimation', pose_img)

        # Display live video
        cv2.imshow('Live Video', img)

        prev_frame = frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main function for notebook usage
def main():
    process_feed()

if __name__ == '__main__':
    main()
