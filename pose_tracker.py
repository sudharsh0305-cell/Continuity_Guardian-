import numpy as np
import math
import cv2

def calculate_angle_3pt(p1, p2, p3):
    """
    Calculates the angle at point p2 formed by points p1, p2, p3.
    Returns angle in degrees.
    """
    if p1 is None or p2 is None or p3 is None:
        return 0.0
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (norm1 * norm2)
    # Clip to avoid float precision issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    angle = np.arccos(cos_angle)
    return float(np.degrees(angle))

def calculate_line_angle(p1, p2, relative_to="horizontal"):
    """
    Calculates the angle of the line connecting p1 and p2.
    relative_to: "horizontal" or "vertical"
    """
    if p1 is None or p2 is None:
        return 0.0
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    
    if relative_to == "vertical":
        # Angle relative to vertical centerline
        return float(math.degrees(math.atan2(dx, dy)))
    else:
        # Angle relative to horizontal centerline
        return float(math.degrees(math.atan2(dy, dx)))

def extract_pose_features(keypoints, confs=None, min_conf=0.4):
    """
    Extracts elbow angles, shoulder tilt, and torso stance from 17 keypoints.
    keypoints: np.ndarray of shape (17, 2)
    confs: np.ndarray of shape (17,)
    """
    def get_pt(idx):
        if confs is not None and confs[idx] < min_conf:
            return None
        return (float(keypoints[idx][0]), float(keypoints[idx][1]))
        
    l_shoulder = get_pt(5)
    r_shoulder = get_pt(6)
    l_elbow = get_pt(7)
    r_elbow = get_pt(8)
    l_wrist = get_pt(9)
    r_wrist = get_pt(10)
    l_hip = get_pt(11)
    r_hip = get_pt(12)
    
    l_elbow_bend = calculate_angle_3pt(l_shoulder, l_elbow, l_wrist)
    r_elbow_bend = calculate_angle_3pt(r_shoulder, r_elbow, r_wrist)
    
    shoulder_tilt = 0.0
    if l_shoulder is not None and r_shoulder is not None:
        shoulder_tilt = calculate_line_angle(l_shoulder, r_shoulder, "horizontal")
        
    torso_stance = 0.0
    if l_shoulder is not None and r_shoulder is not None and l_hip is not None and r_hip is not None:
        mid_shoulder = ((l_shoulder[0] + r_shoulder[0])/2.0, (l_shoulder[1] + r_shoulder[1])/2.0)
        mid_hip = ((l_hip[0] + r_hip[0])/2.0, (l_hip[1] + r_hip[1])/2.0)
        torso_stance = calculate_line_angle(mid_shoulder, mid_hip, "vertical")
        
    return {
        "left_elbow": round(l_elbow_bend, 1),
        "right_elbow": round(r_elbow_bend, 1),
        "shoulder_tilt": round(shoulder_tilt, 1),
        "torso_stance": round(torso_stance, 1)
    }

def draw_skeleton(frame, keypoints, confs=None, min_conf=0.4):
    """
    Draws pose skeleton lines connecting joint keypoints on the image frame.
    """
    # Keypoint connections mapping
    connections = [
        (5, 6), # Shoulder to shoulder
        (5, 7), (7, 9), # Left arm
        (6, 8), (8, 10), # Right arm
        (5, 11), (6, 12), # Torso sides
        (11, 12) # Hip to hip
    ]
    
    # Draw skeletal lines
    for start_idx, end_idx in connections:
        if confs is not None and (confs[start_idx] < min_conf or confs[end_idx] < min_conf):
            continue
        p1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
        p2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))
        cv2.line(frame, p1, p2, (255, 0, 255), 2)
        
    # Draw keypoint joint joints
    for idx in [5, 6, 7, 8, 9, 10, 11, 12]:
        if confs is not None and confs[idx] < min_conf:
            continue
        pt = (int(keypoints[idx][0]), int(keypoints[idx][1]))
        cv2.circle(frame, pt, 4, (0, 255, 255), -1)
