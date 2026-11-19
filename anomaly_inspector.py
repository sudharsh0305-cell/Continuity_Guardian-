import sys
import os
import requests
import json
import base64
import cv2

def inspect_anomaly(image_path):
    """
    Sends the 2x3 snapshot grid to a local vision model via Ollama,
    falling back to an intelligent heuristic script parser if Ollama is not running.
    """
    ollama_url = "http://localhost:11434/api/generate"
    ollama_running = False
    try:
        r = requests.get("http://localhost:11434/", timeout=1)
        if r.status_code == 200:
            ollama_running = True
    except Exception:
        pass
        
    if ollama_running:
        try:
            with open(image_path, "rb") as image_file:
                img_str = base64.b64encode(image_file.read()).decode('utf-8')
                
            payload = {
                "model": "llava",  # standard vision model in Ollama
                "prompt": (
                    "Analyze this 2x3 frame grid representing a video tracking anomaly. "
                    "The top row shows the 3 frames before the anomaly. The bottom row shows "
                    "the 3 frames during/after the anomaly starts.\n\n"
                    "Please identify the discontinuity (e.g., did an object suddenly disappear, "
                    "or did it experience an abrupt spatial jump?) and write a formal "
                    "Director's Note suggesting what the director/editor should look at on "
                    "their timeline to resolve this error."
                ),
                "images": [img_str],
                "stream": False
            }
            resp = requests.post(ollama_url, json=payload, timeout=30)
            if resp.status_code == 200:
                response_text = resp.json().get("response", "").strip()
                if response_text:
                    return f"🎬 **DIRECTOR'S NOTES (AI-VISION GENERATED)**\n\n{response_text}"
        except Exception as e:
            pass

    # Heuristic fallback parsing the filename details
    filename = os.path.basename(image_path)
    parts = filename.replace("anomaly_frame_", "").replace(".png", "").split("_")
    
    # Defaults
    video_name = "Unknown Video"
    log_type = "anomaly"
    object_id = "Unknown"
    frame_num = "Unknown"
    
    if len(parts) >= 4:
        video_name = parts[0]
        log_type = parts[1]
        object_id = parts[2]
        frame_num = parts[3].replace("frame", "")
        
    if log_type == "vanishing":
        notes = (
            f"🎬 **DIRECTOR'S NOTES (SCRIPT SUPERVISOR RULE-BASE)**\n\n"
            f"**Scene:** `{video_name}` | **Timecode:** Frame {frame_num} | **Subject:** Object ID {object_id}\n\n"
            f"⚠️ **Anomaly Detected:** Critical Vanishing (Object Disappeared from Central Screen)\n\n"
            f"**Observations:** Object ID {object_id} was successfully tracked inside the central region of interest (ROI) "
            f"during the pre-anomaly sequence (top row). Between frame {frame_num} and the next, it vanished abruptly from the screen "
            f"without exiting through the borders.\n\n"
            f"**Correction Action Plan for Editor/Director:**\n"
            f"1. **Check Timeline Visibility:** Inspect the editing layers to verify if the element's visibility toggle, scale, or opacity "
            f"keyframe was deleted or shifted.\n"
            f"2. **Verify Occlusion Masks:** Ensure that a background/foreground mask layer is not prematurely overlaying and hiding the subject.\n"
            f"3. **Camera Check:** If this was an intentional jump-cut or scene switch, verify if the tracking engines were instructed to reset "
            f"to prevent the tracker from inheriting the ID of a missing object."
        )
    elif log_type == "spatial":
        if len(parts) >= 5:
            object_id = parts[3]
            frame_num = parts[4].replace("frame", "")
            
        notes = (
            f"🎬 **DIRECTOR'S NOTES (SCRIPT SUPERVISOR RULE-BASE)**\n\n"
            f"**Scene:** `{video_name}` | **Timecode:** Frame {frame_num} | **Subject:** Object ID {object_id}\n\n"
            f"⚠️ **Anomaly Detected:** Spatial Discontinuity (Drastic Object Jump)\n\n"
            f"**Observations:** Object ID {object_id} experienced a spatial shift of more than 50 pixels in consecutive frames, "
            f"as shown by the sudden offset of the bounding boxes between the top and bottom rows.\n\n"
            f"**Correction Action Plan for Editor/Director:**\n"
            f"1. **Audit Frame Drops:** Check the rendering pipeline for dropped frames or corrupt video packets that could skip motion segments.\n"
            f"2. **Transition Check:** If this represents a fast pan, consider using a camera motion blur effect or cross-dissolve to smooth "
            f"the abrupt position shift.\n"
            f"3. **Layout Alignment:** Confirm if the object's parent container or coordinate grid was shifted by a layout script error."
        )
    else:
        notes = (
            f"🎬 **DIRECTOR'S NOTES (SCRIPT SUPERVISOR RULE-BASE)**\n\n"
            f"**Scene:** `{video_name}` | **Timecode:** Frame {frame_num} | **Subject:** Object ID {object_id}\n\n"
            f"⚠️ **Anomaly Detected:** Layout / Geometry Discontinuity\n\n"
            f"**Observations:** The object's tracking characteristics shifted unexpectedly inside the active frame ROI.\n\n"
            f"**Correction Action Plan for Editor/Director:** Inspect the frames on the editing table for visual matching and check layout constraints."
        )
    return notes

def inspect_all_anomalies(master_log_path="master_continuity_log.json"):
    """
    Scans the master logs, extracts target frames from videos, runs the VFX
    inpainting generator on the bounding boxes, and saves side-by-side previews.
    """
    if not os.path.exists(master_log_path):
        print(f"Error: Log file not found at {master_log_path}")
        return
        
    try:
        with open(master_log_path, "r") as f:
            log_data = json.load(f)
    except Exception as e:
        print(f"Error loading master log: {e}")
        return
        
    events = log_data.get("events", [])
    if not events:
        print("No anomaly events logged to inspect.")
        return
        
    from inpaint_engine import VFXPatchGenerator
    patcher = VFXPatchGenerator()
    
    out_dir = "output_clips/vfx_patches"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 Running inpainting engine on {len(events)} logged anomalies...")
    
    for idx, ev in enumerate(events, 1):
        video_name = ev.get("video_name")
        frame_num = ev.get("frame")
        log_type = ev.get("log_type")
        object_id = ev.get("object_id")
        box = ev.get("box")
        
        if not box:
            continue
            
        video_path = os.path.join("input_videos", video_name)
        if not os.path.exists(video_path):
            continue
            
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            continue
            
        video_base = os.path.splitext(video_name)[0]
        
        # Save original frame
        orig_name = f"{video_base}_{log_type}_id{object_id}_frame{frame_num}_original.png"
        orig_path = os.path.join(out_dir, orig_name)
        cv2.imwrite(orig_path, frame)
        
        # Generate patched clean backplate frame
        patched_frame = patcher.generate_patch(frame, box)
        patch_name = f"{video_base}_{log_type}_id{object_id}_frame{frame_num}_patched.png"
        patch_path = os.path.join(out_dir, patch_name)
        cv2.imwrite(patch_path, patched_frame)
        
        print(f"✅ Generated VFX Patch preview: {patch_name}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        inspect_all_anomalies()
    else:
        if len(sys.argv) < 2:
            print("Usage:")
            print("  python3 anomaly_inspector.py --batch")
            print("  python3 anomaly_inspector.py <path_to_snapshot>")
            sys.exit(1)
        img_path = sys.argv[1]
        if not os.path.exists(img_path):
            print(f"Error: {img_path} not found.")
            sys.exit(1)
        print(inspect_anomaly(img_path))
