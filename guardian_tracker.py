import cv2
import math
import json
import os
import time
import argparse
import queue
import threading
import numpy as np
from ultralytics import YOLO
from reid_engine import extract_appearance_vector, cosine_similarity, load_registry, save_registry
from audio_monitor import analyze_audio, detect_audio_anomalies
from pose_tracker import extract_pose_features, draw_skeleton
from depth_tracker import MonocularDepthEstimator, get_box_depth

def detect_shot_boundaries(video_path, threshold=27.0):
    """
    Detects shot boundaries/camera cuts in the video file using PySceneDetect.
    Returns:
        list of cut frame indices.
    """
    try:
        from scenedetect import detect, ContentDetector
        print(f"[Scene Detect] Analyzing shot boundaries for: {video_path}...")
        scene_list = detect(video_path, ContentDetector(threshold=threshold))
        cuts = []
        for i, scene in enumerate(scene_list):
            if i > 0: # Cuts happen between scenes
                cuts.append(scene[0].frame_num)
        return cuts
    except Exception as e:
        print(f"[Scene Detect] Failed to run PySceneDetect: {e}. Falling back to empty boundaries.")
        return []

class ThreadedVideoQueue:
    """
    Threaded frame reader to retrieve video frames asynchronously in a separate thread.
    Helps maximize pipeline throughput by decoupling decoding from processing.
    """
    def __init__(self, video_path, queue_size=128, sampled_frames=None):
        self.cap = cv2.VideoCapture(video_path)
        self.Q = queue.Queue(maxsize=queue_size)
        self.stopped = False
        self.sampled_frames = sampled_frames
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        
    def start(self):
        self.thread.start()
        return self
        
    def update(self):
        if self.sampled_frames is not None:
            for frame_idx in self.sampled_frames:
                if self.stopped:
                    return
                # Seek directly to the target frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = self.cap.read()
                if not success:
                    break
                # Wait if Q is full
                while not self.stopped and self.Q.full():
                    time.sleep(0.001)
                if self.stopped:
                    return
                self.Q.put((frame_idx, frame))
            self.stop()
            return

        while True:
            if self.stopped:
                return
            if not self.Q.full():
                success, frame = self.cap.read()
                if not success:
                    self.stop()
                    return
                self.Q.put(frame)
            else:
                # Sleep briefly if queue is full to reduce CPU load
                time.sleep(0.001)
                
    def read(self):
        if self.Q.empty() and self.stopped:
            return False, None
        try:
            # Retrieve next frame with a bounded timeout in case of slowdowns
            val = self.Q.get(timeout=1.0)
            return True, val
        except queue.Empty:
            return False, None
            
    def isOpened(self):
        return self.cap.isOpened() and not self.stopped
        
    def release(self):
        self.stop()
        self.cap.release()
        
    def stop(self):
        self.stopped = True
        # Drain the queue to unblock the background thread if it is blocked on Q.put
        while not self.Q.empty():
            try:
                self.Q.get_nowait()
            except queue.Empty:
                break

def main():
    parser = argparse.ArgumentParser(description="Continuity Guardian Batch Processing Pipeline")
    parser.add_argument("--video", type=str, default=None, help="Specific video file to process (defaults to all)")
    parser.add_argument("--input-dir", type=str, default="input_videos", help="Input directory containing video files (default: input_videos)")
    parser.add_argument("--frame-skip", type=int, default=3, help="Process inference every N-th frame (default: 3)")
    parser.add_argument("--edge-margin", type=float, default=0.05, help="Edge margin fraction for screen exit check (default: 0.05)")
    parser.add_argument("--jump-threshold", type=float, default=50.0, help="Spatial jump distance threshold in pixels (default: 50.0)")
    parser.add_argument("--vanishing-grace", type=int, default=5, help="Grace period frames for vanishing checking (default: 5)")
    parser.add_argument("--no-show", action="store_true", help="Disable GUI display window (recommended for streamlit/server runs)")
    parser.add_argument("--fast-sample", action="store_true", help="Enable ContentDetector keyframe sampling for fast scanning")
    
    args = parser.parse_args()
    
    input_folder = args.input_dir
    logs_folder = "logs"
    output_folder = "output_videos"
    snapshots_folder = "anomaly_snapshots"
    clips_folder = "anomaly_clips"
    master_log_path = "master_continuity_log.json"
    
    FRAME_SKIP = args.frame_skip
    EDGE_MARGIN_PERCENT = args.edge_margin
    JUMP_THRESHOLD = args.jump_threshold
    VANISHING_GRACE = args.vanishing_grace
    NO_SHOW = args.no_show
    
    # Create directories if they do not exist
    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(snapshots_folder, exist_ok=True)
    os.makedirs(clips_folder, exist_ok=True)
    
    # Scan for video files with matching extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    if args.video:
        video_files = [args.video]
    else:
        video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)]
    
    if not video_files:
        print(f"No video files found in '{input_folder}/'. Please add video files to process.")
        return
        
    print(f"Discovered video(s) to process: {video_files}")
    
    # Structure master logs to hold both event logs and execution metrics
    master_log_data = {
        "metrics": {},
        "events": []
    }
    
    # Load Multi-Shot Re-ID appearance registry
    reid_registry = load_registry()
    
    last_known_postures = {}
    last_known_depths = {}
    abort_batch = False

    for video_file in sorted(video_files):
        if abort_batch:
            break
            
        video_path = os.path.join(input_folder, video_file)
        video_name = video_file
        video_basename = os.path.splitext(video_file)[0]
        
        print("\n" + "="*60)
        print(f"PROCESSING VIDEO (BATCH RUN): {video_name}")
        print("="*60)
        
        # Load YOLO models (re-instantiated for each video to isolate tracker state)
        print("Loading YOLO models...")
        model = YOLO("yolov8n.pt")
        pose_model = YOLO("yolov8n-pose.pt")
        depth_estimator = MonocularDepthEstimator()
        
        sampled_frames = None
        if args.fast_sample:
            temp_cap = cv2.VideoCapture(video_path)
            total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            temp_cap.release()
            
            cuts = detect_shot_boundaries(video_path)
            print(f"[Fast Sample] Detected {len(cuts)} shot boundaries in {video_name}.")
            
            sampled_set = set()
            # Sample starting keyframes
            for f in range(min(15, total_frames)):
                sampled_set.add(f)
            # Sample keyframes around detected camera cuts
            for cut in cuts:
                for f in range(max(0, cut - 5), min(total_frames, cut + 5)):
                    sampled_set.add(f)
            # Sample ending keyframes
            for f in range(max(0, total_frames - 5), total_frames):
                sampled_set.add(f)
                
            sampled_frames = sorted(list(sampled_set))
            print(f"[Fast Sample] Keyframe reduction: {total_frames} frames -> {len(sampled_frames)} frames.")
            
        print(f"Starting ThreadedVideoQueue for: {video_path}")
        reader = ThreadedVideoQueue(video_path, sampled_frames=sampled_frames).start()

        if not reader.isOpened():
            print(f"Error: Could not open video source '{video_path}'. Skipping.")
            continue

        print(f"YOLO tracking started (FRAME_SKIP={FRAME_SKIP}). Press 'q' to abort.")
        
        # Get frame properties and initialize VideoWriter
        fps = reader.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(reader.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(reader.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_video_path = os.path.join(output_folder, f"{video_basename}_annotated.mp4")
        depth_video_path = os.path.join(output_folder, f"{video_basename}_depth.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        depth_writer = cv2.VideoWriter(depth_video_path, fourcc, fps, (width, height))
        print(f"Saving annotated video to: {output_video_path}")
        print(f"Saving colorized depth video to: {depth_video_path}")
        
        # Parse audio stream and detect anomalies
        print(f"Extracting and analyzing audio track for: {video_path}...")
        audio_db_timeline = analyze_audio(video_path)
        audio_anomalies = detect_audio_anomalies(audio_db_timeline)
        print(f"Audio Analysis: Decoded {len(audio_db_timeline)} frames. Detected {len(audio_anomalies)} audio events/anomalies.")

        # Tracking dictionaries & counters
        previous_centers = {}
        missing_counters = {}
        last_known_boxes = {}
        centroid_history = {} # track_id -> list of (cx, cy)
        video_logs = []
        reid_discontinuities_logged = set()
        object_movement_timeline = []
        posture_timeline = {}
        depth_timeline = {}
        detected_classes = set()
        
        # Sliding window buffer (last 3 annotated frames) for pre-anomaly snapshot stitching
        recent_frames = []
        # Active snapshots tracking list (anomalies currently capturing post-frames)
        active_snapshots = []
        
        # Isolated anomaly sub-clips variables
        pre_anomaly_frames_limit = int(2.5 * fps)
        post_anomaly_frames_limit = int(2.5 * fps)
        frame_history = []
        active_clips = []
        
        # Alert banner overlay state
        alert_overlay_text = ""
        alert_overlay_timer = 0
        
        # Performance logging metrics initialization
        total_inference_time = 0.0
        inference_frames_count = 0
        frame_count = 0
        last_inference_results = None
        
        start_total_time = time.time()

        # Loop through the video frames
        while reader.isOpened():
            success, val = reader.read()

            if success:
                if args.fast_sample:
                    frame_idx, frame = val
                    frame_count = frame_idx + 1
                else:
                    frame = val
                    frame_count += 1
                draw_skeleton_queue = []
                
                # Compute per-frame normalized depth map (near=0.0, far=1.0)
                depth_map = depth_estimator.estimate_depth(frame)
                
                # Export to colorized depth heatmap video
                depth_uint8 = (depth_map * 255).astype(np.uint8)
                colorized_depth = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
                depth_writer.write(colorized_depth)
                
                # Retrieve frame shape and calculate edge ROI boundaries
                height, width = frame.shape[:2]
                x_min = width * EDGE_MARGIN_PERCENT
                x_max = width * (1.0 - EDGE_MARGIN_PERCENT)
                y_min = height * EDGE_MARGIN_PERCENT
                y_max = height * (1.0 - EDGE_MARGIN_PERCENT)
                
                # Determine if this is an inference frame
                is_inference_frame = (frame_count % FRAME_SKIP == 1)
                
                if is_inference_frame:
                    # Run YOLO tracking and measure model execution speed
                    start_inf = time.time()
                    results = model.track(frame, persist=True, tracker="custom_tracker.yaml", conf=0.05)
                    pose_results = pose_model(frame, verbose=False)
                    total_inference_time += (time.time() - start_inf)
                    inference_frames_count += 1
                    
                    last_inference_results = results
                    if results[0].boxes is not None:
                        names = results[0].names
                        for c in results[0].boxes.cls.int().cpu().tolist():
                            detected_classes.add(names[c])
                    
                    # Extract pose keypoints and boxes
                    pose_keypoints = []
                    pose_boxes = []
                    if pose_results[0].keypoints is not None and len(pose_results[0].keypoints) > 0:
                        kpts_xy = pose_results[0].keypoints.xy.cpu().numpy()
                        kpts_conf = pose_results[0].keypoints.conf.cpu().numpy() if pose_results[0].keypoints.conf is not None else None
                        if pose_results[0].boxes is not None:
                            p_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                            for i in range(len(kpts_xy)):
                                confs = kpts_conf[i] if kpts_conf is not None else None
                                pose_keypoints.append((kpts_xy[i], confs))
                                pose_boxes.append(p_boxes[i])

                    # Get track IDs present in the current frame
                    frame_displacements = []
                    current_ids = set()
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        current_ids = set(results[0].boxes.id.int().cpu().tolist())

                    # Find missing IDs (present in previous_centers but not in current frame)
                    missing_ids = set(previous_centers.keys()) - current_ids

                    # Calculate centers and check for drastic shifts
                    if results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.int().cpu().tolist()
                        for box, track_id in zip(boxes, track_ids):
                            x1, y1, x2, y2 = box
                            cx = (x1 + x2) / 2
                            cy = (y1 + y2) / 2
                            last_known_boxes[track_id] = box.tolist()

                            # 1. Multi-Shot Re-ID appearance validation
                            try:
                                ix1, iy1, ix2, iy2 = map(int, [max(0, x1), max(0, y1), min(width, x2), min(height, y2)])
                                crop_img = frame[iy1:iy2, ix1:ix2]
                                v_curr = extract_appearance_vector(crop_img)
                                if v_curr:
                                    track_id_str = str(track_id)
                                    
                                    # Save a thumbnail profile for the actor/prop
                                    thumbnail_path = f"anomaly_snapshots/thumbnail_id_{track_id}.png"
                                    if not os.path.exists(thumbnail_path):
                                        cv2.imwrite(thumbnail_path, crop_img)
                                        
                                    if track_id_str in reid_registry:
                                        v_reg = reid_registry[track_id_str]
                                        sim = cosine_similarity(v_curr, v_reg)
                                        if sim < 0.75 and track_id_str not in reid_discontinuities_logged:
                                            message = f"CRITICAL: Object ID {track_id} has an appearance discontinuity (similarity: {sim:.2f})!"
                                            print(f"[{video_name} - Frame {frame_count}] {message}")
                                            
                                            # Trigger alert overlay banner
                                            alert_overlay_text = f"RE-ID ERROR: ID {track_id}"
                                            alert_overlay_timer = 25
                                            
                                            video_logs.append({
                                                "video_name": video_name,
                                                "frame": frame_count,
                                                "log_type": "appearance_discontinuity",
                                                "object_id": track_id,
                                                "box": box.tolist(),
                                                "message": message
                                            })
                                            reid_discontinuities_logged.add(track_id_str)
                                        elif sim >= 0.75:
                                            # Smooth update
                                            v_new = [0.9 * a + 0.1 * b for a, b in zip(v_reg, v_curr)]
                                            reid_registry[track_id_str] = v_new
                                    else:
                                        # Register new tracking profile
                                        reid_registry[track_id_str] = v_curr
                            except Exception as e:
                                print(f"Error processing Re-ID for object {track_id}: {e}")

                            # 2. Skeletal Pose & Posture Continuity Tracking
                            try:
                                matched_pose_idx = -1
                                min_dist = float('inf')
                                for p_idx, p_box in enumerate(pose_boxes):
                                    px1, py1, px2, py2 = p_box[:4]
                                    pcx = (px1 + px2) / 2
                                    pcy = (py1 + py2) / 2
                                    dist = math.sqrt((cx - pcx)**2 + (cy - pcy)**2)
                                    if dist < min_dist and dist < 120:
                                        min_dist = dist
                                        matched_pose_idx = p_idx
                                        
                                if matched_pose_idx != -1:
                                    kpts, confs = pose_keypoints[matched_pose_idx]
                                    draw_skeleton_queue.append((kpts, confs))
                                    current_pose = extract_pose_features(kpts, confs)
                                    
                                    if current_pose:
                                        track_id_str = str(track_id)
                                        is_reappearing = (track_id in missing_counters and missing_counters[track_id] > 0)
                                        
                                        if track_id_str in last_known_postures:
                                            prev_pose = last_known_postures[track_id_str]
                                            devs = []
                                            for k in ["left_elbow", "right_elbow", "shoulder_tilt", "torso_stance"]:
                                                diff = abs(current_pose[k] - prev_pose[k])
                                                if diff > 180.0:
                                                    diff = 360.0 - diff
                                                devs.append(diff)
                                            max_dev = max(devs, default=0.0)
                                            
                                            # Alert if posture shifts drastically (>35 degrees)
                                            # across transitions or reappearance
                                            if max_dev > 35.0 and (is_reappearing or frame_count <= 5):
                                                message = f"CRITICAL: Object ID {track_id} has a posture discontinuity (deviation: {max_dev:.1f} degrees)!"
                                                print(f"[{video_name} - Frame {frame_count}] {message}")
                                                
                                                alert_overlay_text = f"POSE DEV: {max_dev:.0f} DEG"
                                                alert_overlay_timer = 20
                                                
                                                video_logs.append({
                                                    "video_name": video_name,
                                                    "frame": frame_count,
                                                    "log_type": "posture_discontinuity",
                                                    "object_id": track_id,
                                                    "box": box.tolist(),
                                                    "message": message
                                                })
                                                
                                        # Update last known pose state
                                        last_known_postures[track_id_str] = current_pose
                                        
                                        # Append to timeline history
                                        if track_id_str not in posture_timeline:
                                            posture_timeline[track_id_str] = []
                                        posture_timeline[track_id_str].append({
                                            "frame": frame_count,
                                            "left_elbow": current_pose["left_elbow"],
                                            "right_elbow": current_pose["right_elbow"],
                                            "shoulder_tilt": current_pose["shoulder_tilt"],
                                            "torso_stance": current_pose["torso_stance"]
                                        })
                            except Exception as e:
                                print(f"Error processing pose for object {track_id}: {e}")

                            # 3. Monocular 3D Spatial Depth Solver
                            try:
                                z = get_box_depth(depth_map, box)
                                track_id_str = str(track_id)
                                is_reappearing = (track_id in missing_counters and missing_counters[track_id] > 0)
                                
                                if track_id_str in last_known_depths:
                                    prev_z = last_known_depths[track_id_str]
                                    z_diff = abs(z - prev_z)
                                    
                                    # Log critical 3D_SPATIAL_JUMP alert if Z shifts drastically (>0.25)
                                    # across transitions or reappearance
                                    if z_diff > 0.25 and (is_reappearing or frame_count <= 5):
                                        message = f"CRITICAL: Object ID {track_id} has a 3D depth shift (deviation: {z_diff:.2f})!"
                                        print(f"[{video_name} - Frame {frame_count}] {message}")
                                        
                                        alert_overlay_text = f"3D JUMP: ID {track_id}"
                                        alert_overlay_timer = 20
                                        
                                        video_logs.append({
                                            "video_name": video_name,
                                            "frame": frame_count,
                                            "log_type": "3d_spatial_jump",
                                            "object_id": track_id,
                                            "box": box.tolist(),
                                            "message": message
                                        })
                                        
                                # Update last known depth state
                                last_known_depths[track_id_str] = z
                                
                                # Append to depth timeline
                                if track_id_str not in depth_timeline:
                                    depth_timeline[track_id_str] = []
                                depth_timeline[track_id_str].append({
                                    "frame": frame_count,
                                    "z": round(z, 3)
                                })
                            except Exception as e:
                                print(f"Error processing depth for object {track_id}: {e}")

                            if track_id in previous_centers:
                                prev_cx, prev_cy = previous_centers[track_id]
                                distance = math.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
                                frame_displacements.append(distance)
                                
                                if distance > JUMP_THRESHOLD:
                                    # Check audio anomaly correlation (within 3 frames)
                                    is_av_correlated = False
                                    for f_idx in range(frame_count - 3, frame_count + 4):
                                        if f_idx in audio_anomalies:
                                            is_av_correlated = True
                                            break
                                            
                                    if is_av_correlated:
                                        log_type = "av_jump_cut"
                                        message = f"AV JUMP CUT ALERT: Object ID {track_id} shifted position drastically correlated with audio sync break!"
                                    else:
                                        log_type = "spatial_jump"
                                        message = f"ALERT: Object ID {track_id} shifted position drastically!"
                                        
                                    print(f"[{video_name} - Frame {frame_count}] {message}")
                                    # Set overlay alert banner text & timer
                                    alert_overlay_text = f"AV JUMP CUT" if is_av_correlated else f"SPATIAL JUMP: ID {track_id}"
                                    alert_overlay_timer = 15
                                    
                                    # Trigger snapshot capture
                                    anomaly_id = f"{video_basename}_{log_type}_{track_id}_frame{frame_count}"
                                    active_snapshots.append({
                                        "anomaly_id": anomaly_id,
                                        "pre_frames": list(recent_frames),
                                        "post_frames": []
                                    })
                                    
                                    # Trigger clip capture (2.5s before, 2.5s after)
                                    clip_name = f"clip_{video_basename}_{log_type}_frame{frame_count}.mp4"
                                    clip_path = os.path.join(clips_folder, clip_name)
                                    active_clips.append({
                                        "output_path": clip_path,
                                        "frames": list(frame_history),
                                        "remaining": post_anomaly_frames_limit
                                    })
                                    
                                    video_logs.append({
                                        "video_name": video_name,
                                        "frame": frame_count,
                                        "log_type": log_type,
                                        "object_id": track_id,
                                        "box": box.tolist(),
                                        "message": message
                                    })
                            
                            # Store / update the center coordinates for the track ID
                            previous_centers[track_id] = (cx, cy)
                            
                            # Update centroid history (up to 20 past frames)
                            if track_id not in centroid_history:
                                centroid_history[track_id] = []
                            centroid_history[track_id].append((cx, cy))
                            if len(centroid_history[track_id]) > 20:
                                centroid_history[track_id].pop(0)
                                
                            # Clear missing counter since it reappeared
                            missing_counters.pop(track_id, None)

                    # Handle missing IDs with grace period of frames
                    expired_ids = []
                    for track_id in missing_ids:
                        missing_counters[track_id] = missing_counters.get(track_id, 0) + 1
                        if missing_counters[track_id] > VANISHING_GRACE:
                            cx, cy = previous_centers[track_id]
                            # Check if the last known centroid was in the central ROI
                            is_in_roi = (x_min <= cx <= x_max) and (y_min <= cy <= y_max)
                            
                            if is_in_roi:
                                # Check audio anomaly correlation (within 3 frames)
                                is_av_correlated = False
                                for f_idx in range(frame_count - 3, frame_count + 4):
                                    if f_idx in audio_anomalies:
                                        is_av_correlated = True
                                        break
                                        
                                if is_av_correlated:
                                    log_type = "av_jump_cut"
                                    message = f"AV JUMP CUT ALERT: Object ID {track_id} vanished from the scene correlated with audio sync break!"
                                else:
                                    log_type = "vanishing"
                                    message = f"CRITICAL ALERT: Object ID {track_id} has vanished from the scene!"
                                    
                                print(f"[{video_name} - Frame {frame_count}] {message}")
                                # Set overlay alert banner text & timer
                                alert_overlay_text = f"AV JUMP CUT" if is_av_correlated else f"VANISHED: ID {track_id}"
                                alert_overlay_timer = 20
                                
                                # Trigger snapshot capture
                                anomaly_id = f"{video_basename}_{log_type}_{track_id}_frame{frame_count}"
                                active_snapshots.append({
                                    "anomaly_id": anomaly_id,
                                    "pre_frames": list(recent_frames),
                                    "post_frames": []
                                })
                                
                                # Trigger clip capture (2.5s before, 2.5s after)
                                clip_name = f"clip_{video_basename}_{log_type}_frame{frame_count}.mp4"
                                clip_path = os.path.join(clips_folder, clip_name)
                                active_clips.append({
                                    "output_path": clip_path,
                                    "frames": list(frame_history),
                                    "remaining": post_anomaly_frames_limit
                                })
                                
                                video_logs.append({
                                    "video_name": video_name,
                                    "frame": frame_count,
                                    "log_type": log_type,
                                    "object_id": track_id,
                                    "box": last_known_boxes.get(track_id, [0, 0, width, height]),
                                    "message": message
                                })
                            else:
                                print(f"[{video_name} - Frame {frame_count}] Note: Object ID {track_id} exited screen naturally at ({cx:.1f}, {cy:.1f}).")
                                
                            expired_ids.append(track_id)

                    # Remove vanished objects from memory
                    for track_id in expired_ids:
                        previous_centers.pop(track_id, None)
                        missing_counters.pop(track_id, None)
                        centroid_history.pop(track_id, None)

                    # Visualize the active inference tracking results
                    annotated_frame = results[0].plot()
                    object_movement_timeline.append(max(frame_displacements, default=0.0))
                else:
                    # Skipped frame: Interpolate track visualization using the last known boxes
                    if last_inference_results is not None and last_inference_results[0].boxes is not None:
                        boxes = last_inference_results[0].boxes.xyxy.cpu().numpy()
                        if last_inference_results[0].boxes.id is not None:
                            track_ids = last_inference_results[0].boxes.id.int().cpu().tolist()
                        else:
                            track_ids = [None] * len(boxes)
                            
                        # Draw last known bounding boxes on the skipped frame to preserve visual continuity
                        for box, track_id in zip(boxes, track_ids):
                            x1, y1, x2, y2 = map(int, box[:4])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label = f"ID: {track_id}" if track_id is not None else "Detecting..."
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        annotated_frame = frame
                    else:
                        annotated_frame = frame
                        
                    last_move = object_movement_timeline[-1] if object_movement_timeline else 0.0
                    object_movement_timeline.append(last_move)

                # Draw skeleton overlays on annotated_frame
                for kpts, confs in draw_skeleton_queue:
                    draw_skeleton(annotated_frame, kpts, confs)

                # Draw fading trajectories on annotated_frame
                for o_id, pts in list(centroid_history.items()):
                    if o_id in previous_centers and len(pts) > 1:
                        for i in range(len(pts) - 1):
                            alpha = (i + 1) / len(pts)
                            color = (0, int(255 * alpha), 0)
                            thickness = int(1 + 2 * alpha)
                            cv2.line(annotated_frame, (int(pts[i][0]), int(pts[i][1])), (int(pts[i+1][0]), int(pts[i+1][1])), color, thickness)

                # Draw semi-transparent ROI boundary rectangle (5% margin)
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
                cv2.addWeighted(overlay, 0.25, annotated_frame, 0.75, 0, annotated_frame)
                
                 # Draw bold red alert banner at the top if active
                if alert_overlay_timer > 0:
                    cv2.rectangle(annotated_frame, (0, 0), (width, 45), (0, 0, 255), -1)
                    # Text shadow (black)
                    cv2.putText(annotated_frame, alert_overlay_text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
                    # Text foreground (white)
                    cv2.putText(annotated_frame, alert_overlay_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    alert_overlay_timer -= 1

                # Update recent frames sliding window (keep last 3 annotated frames)
                recent_frames.append(annotated_frame.copy())
                if len(recent_frames) > 3:
                    recent_frames.pop(0)

                # Capture post-anomaly frames and stitch grids
                finished_snapshots = []
                for entry in active_snapshots:
                    entry["post_frames"].append(annotated_frame.copy())
                    if len(entry["post_frames"]) == 3:
                        sub_w = 320
                        sub_h = 192
                        pre_resized = [cv2.resize(f, (sub_w, sub_h)) for f in entry["pre_frames"]]
                        post_resized = [cv2.resize(f, (sub_w, sub_h)) for f in entry["post_frames"]]
                        
                        # Pad lists to guarantee exactly 3 frames
                        while len(pre_resized) < 3:
                            pre_resized.append(pre_resized[-1] if pre_resized else np.zeros((sub_h, sub_w, 3), dtype=np.uint8))
                        while len(post_resized) < 3:
                            post_resized.append(post_resized[-1] if post_resized else np.zeros((sub_h, sub_w, 3), dtype=np.uint8))
                            
                        top_row = cv2.hconcat(pre_resized)
                        bottom_row = cv2.hconcat(post_resized)
                        grid = cv2.vconcat([top_row, bottom_row])
                        
                        # Add banner labels
                        cv2.putText(grid, "PRE-ANOMALY (1-3)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(grid, "POST-ANOMALY (1-3)", (10, sub_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        snapshot_filename = f"anomaly_frame_{entry['anomaly_id']}.png"
                        snapshot_filepath = os.path.join(snapshots_folder, snapshot_filename)
                        cv2.imwrite(snapshot_filepath, grid)
                        print(f"Stitched anomaly snapshot saved to: {snapshot_filepath}")
                        finished_snapshots.append(entry)
                        
                for entry in finished_snapshots:
                    active_snapshots.remove(entry)

                # Update frame history window (keep last 2.5 seconds of frames)
                frame_history.append(annotated_frame.copy())
                if len(frame_history) > pre_anomaly_frames_limit:
                    frame_history.pop(0)

                # Capture post-anomaly frames and export sub-clips
                finished_clips = []
                for clip in active_clips:
                    clip["frames"].append(annotated_frame.copy())
                    clip["remaining"] -= 1
                    if clip["remaining"] == 0:
                        try:
                            clip_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            clip_writer = cv2.VideoWriter(clip["output_path"], clip_fourcc, fps, (width, height))
                            for f in clip["frames"]:
                                clip_writer.write(f)
                            clip_writer.release()
                            print(f"Isolated anomaly clip saved to: {clip['output_path']}")
                        except Exception as e:
                            print(f"Error writing isolated clip to {clip['output_path']}: {e}")
                        finished_clips.append(clip)
                        
                for clip in finished_clips:
                    active_clips.remove(clip)

                # Save the annotated frame to output video file
                writer.write(annotated_frame)

                # Show GUI display if allowed
                if not NO_SHOW:
                    cv2.imshow("YOLO Object Tracking & Guardian", annotated_frame)
                    # Break the loop if 'q' is pressed (aborts the entire batch run)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("User aborted the batch run.")
                        abort_batch = True
                        break
            else:
                # Video ended or frame queue empty
                break

        # Stop threaded frame queue reader and video writer
        reader.release()
        writer.release()
        depth_writer.release()
        if not NO_SHOW:
            cv2.destroyAllWindows()
        print(f"Released video source '{video_name}'")
        
        # Calculate execution performance metrics
        end_total_time = time.time()
        total_process_time = end_total_time - start_total_time
        
        raw_inference_fps = inference_frames_count / total_inference_time if total_inference_time > 0 else 0
        overall_pipeline_fps = frame_count / total_process_time if total_process_time > 0 else 0
        
        video_metrics = {
            "total_process_time_sec": round(total_process_time, 2),
            "raw_inference_fps": round(raw_inference_fps, 1),
            "overall_pipeline_fps": round(overall_pipeline_fps, 1),
            "audio_db_timeline": audio_db_timeline,
            "object_movement_timeline": object_movement_timeline,
            "posture_timeline": posture_timeline,
            "depth_timeline": depth_timeline,
            "detected_classes": list(detected_classes)
        }
        
        print("\n" + "-"*40)
        print(f" PERFORMANCE METRICS - {video_name}")
        print(f" • Total frames:         {frame_count}")
        print(f" • Inference runs:       {inference_frames_count}")
        print(f" • Total process time:   {video_metrics['total_process_time_sec']} sec")
        print(f" • Raw Inference Speed:  {video_metrics['raw_inference_fps']} FPS")
        print(f" • Overall Pipeline Speed: {video_metrics['overall_pipeline_fps']} FPS")
        print("-"*40 + "\n")
        
        # Save individual logs for this video
        individual_log_path = os.path.join(logs_folder, f"{video_basename}_log.json")
        try:
            with open(individual_log_path, "w") as f:
                json.dump(video_logs, f, indent=4)
            print(f"Individual log saved to {individual_log_path}")
        except Exception as e:
            print(f"Error saving individual log for '{video_name}': {e}")
            
        # Append data to unified master structures
        master_log_data["events"].extend(video_logs)
        master_log_data["metrics"][video_name] = video_metrics

    # Save the consolidated master log file and Re-ID registry
    try:
        save_registry(reid_registry)
        
        with open(master_log_path, "w") as f:
            json.dump(master_log_data, f, indent=4)
        print("\n" + "="*60)
        print(f"Unified master log saved to '{master_log_path}'")
        print(f" • Total anomalies logged: {len(master_log_data['events'])}")
        print(f" • Metrics logged for:     {list(master_log_data['metrics'].keys())}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Error saving master log file: {e}")

if __name__ == "__main__":
    main()
