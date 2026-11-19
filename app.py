import streamlit as st
import os
import json
import subprocess
import pandas as pd
from anomaly_inspector import inspect_anomaly
from reid_engine import cosine_similarity

# Set page configuration for wide layout and premium title
st.set_page_config(
    page_title="Continuity Guardian Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling via CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    h1 {
        color: #00F0FF;
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        margin-bottom: 5px;
    }
    h3 {
        color: #FF5E3A;
    }
    .metric-card {
        background-color: #1A1C23;
        border: 1px solid #2D3139;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛡️ Continuity Guardian Dashboard")
st.markdown("### Real-time Video Layout & Continuity Error Detection System")

# Setup directories
input_folder = "input_videos"
output_folder = "output_videos"
logs_folder = "logs"
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(logs_folder, exist_ok=True)

# Scan for video files
video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
videos = [f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)]

# Sidebar configuration
st.sidebar.header("⚙️ Pipeline Configuration")

if not videos:
    st.sidebar.warning("No videos found in `input_videos/` directory. Please upload or copy files to proceed.")
    selected_video = None
else:
    selected_video = st.sidebar.selectbox("Select Target Video", sorted(videos))

frame_skip = st.sidebar.slider("Frame Skip (Inference Gap)", min_value=1, max_value=10, value=3)
jump_threshold = st.sidebar.slider("Spatial Jump Threshold (px)", min_value=10.0, max_value=200.0, value=50.0, step=5.0)
vanishing_grace = st.sidebar.slider("Vanishing Grace Period (frames)", min_value=1, max_value=20, value=5)
edge_margin = st.sidebar.slider("Edge Margin Ratio", min_value=0.0, max_value=0.2, value=0.05, step=0.01)

run_button = st.sidebar.button("🚀 Trigger Tracking Pipeline", type="primary", disabled=(selected_video is None))

# Main logic
if run_button and selected_video:
    with st.spinner(f"Processing '{selected_video}' with selected hyper-parameters..."):
        # Build command arguments
        cmd = [
            "python3", "guardian_tracker.py",
            "--video", selected_video,
            "--frame-skip", str(frame_skip),
            "--edge-margin", str(edge_margin),
            "--jump-threshold", str(jump_threshold),
            "--vanishing-grace", str(vanishing_grace),
            "--no-show"
        ]
        
        # Run subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            st.sidebar.success("Pipeline executed successfully!")
            # Trigger report generator
            subprocess.run(["python3", "generate_report.py"])
        else:
            st.sidebar.error("Pipeline run failed.")
            st.sidebar.text_area("Console Output:", result.stderr)

# Display content if log exists
master_log_path = "master_continuity_log.json"
if os.path.exists(master_log_path):
    try:
        with open(master_log_path, "r") as f:
            log_data = json.load(f)
    except Exception:
        log_data = {}
else:
    log_data = {}

# Layout division
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
    "📊 Analytics & Metrics", 
    "🎥 Annotated Video Output", 
    "📋 Anomaly Log Details", 
    "🎬 Director's Desk", 
    "🎞️ Anomaly Snippets", 
    "👥 Multi-Shot Re-ID", 
    "🔊 Audio-Visual Sync",
    "🧍 Posture Continuity",
    "🧊 3D Depth & Spatial Volume",
    "🎨 Generative VFX Fixes",
    "📜 Script vs. Screen"
])

with tab1:
    st.subheader("Pipeline Telemetry & Execution Summary")
    if log_data and "metrics" in log_data:
        metrics = log_data["metrics"]
        if selected_video and selected_video in metrics:
            m = metrics[selected_video]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Process Time (sec)", f"{m.get('total_process_time_sec', 0.0)} s")
            with col2:
                st.metric("Raw Inference Speed (YOLO)", f"{m.get('raw_inference_fps', 0.0)} FPS")
            with col3:
                st.metric("Overall Pipeline Speed", f"{m.get('overall_pipeline_fps', 0.0)} FPS")
        else:
            st.info("Please run the pipeline for the selected video to generate telemetry.")
    else:
        st.info("No metrics logged yet. Please trigger the pipeline.")

    st.markdown("---")
    st.subheader("Continuity Anomaly Distribution Chart")
    report_img = "batch_continuity_report.png"
    if os.path.exists(report_img):
        st.image(report_img, use_container_width=True)
    else:
        st.info("No summary plot generated yet.")

with tab2:
    st.subheader("Annotated Video Playback")
    if selected_video:
        video_base = os.path.splitext(selected_video)[0]
        annotated_video = os.path.join(output_folder, f"{video_base}_annotated.mp4")
        if os.path.exists(annotated_video):
            st.video(annotated_video)
        else:
            st.warning(f"Annotated video output '{annotated_video}' not found. Please run the tracking pipeline first.")
    else:
        st.info("Please select a video file in the sidebar.")

with tab3:
    st.subheader("Structured Anomaly Logs")
    
    # Add NLE download buttons
    from nle_exporter import export_edl, export_fcpxml
    col_edl, col_xml = st.columns(2)
    with col_edl:
        edl_data = export_edl()
        st.download_button(
            label="🎬 Export EDL (DaVinci/Premiere)",
            data=edl_data,
            file_name="continuity_markers.edl",
            mime="text/plain",
            key="download_edl"
        )
    with col_xml:
        fcpxml_data = export_fcpxml()
        st.download_button(
            label="🎬 Export FCPXML (Final Cut Pro)",
            data=fcpxml_data,
            file_name="continuity_markers.fcpxml",
            mime="text/xml",
            key="download_fcpxml"
        )
    st.markdown("---")
    
    if selected_video:
        video_base = os.path.splitext(selected_video)[0]
        individual_log = os.path.join(logs_folder, f"{video_base}_log.json")
        if os.path.exists(individual_log):
            try:
                with open(individual_log, "r") as f:
                    events = json.load(f)
                if events:
                    df = pd.DataFrame(events)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.success("No anomalies detected! Video complies with all spatial and temporal limits.")
            except Exception as e:
                st.error(f"Error loading logs: {e}")
        else:
            st.warning("No log file found for this video. Run the pipeline to populate.")
    else:
        st.info("Select a video from the sidebar to inspect anomalies.")

with tab4:
    st.subheader("🎬 AI script Supervisor's Inspection Desk")
    snapshots_dir = "anomaly_snapshots"
    os.makedirs(snapshots_dir, exist_ok=True)
    
    if selected_video:
        video_base = os.path.splitext(selected_video)[0]
        # Find all snapshots matching the selected video basename
        snapshots = [
            f for f in os.listdir(snapshots_dir) 
            if f.startswith(f"anomaly_frame_{video_base}_") and f.endswith(".png")
        ]
        
        if not snapshots:
            st.info(f"No anomaly snapshots captured yet for '{selected_video}'. Run the tracking pipeline to record anomalies.")
        else:
            st.markdown(f"Found **{len(snapshots)}** continuity anomaly snapshot(s) for the selected video:")
            for snapshot_file in sorted(snapshots):
                snapshot_path = os.path.join(snapshots_dir, snapshot_file)
                notes_path = snapshot_path + "_notes.txt"
                
                # Retrieve or generate script notes
                if os.path.exists(notes_path):
                    try:
                        with open(notes_path, "r") as f:
                            notes = f.read()
                    except Exception:
                        notes = "Failed to load cached notes."
                else:
                    with st.spinner(f"Analyzing snapshot: {snapshot_file}..."):
                        notes = inspect_anomaly(snapshot_path)
                        try:
                            with open(notes_path, "w") as f:
                                f.write(notes)
                        except Exception as e:
                            st.warning(f"Could not cache notes file: {e}")
                
                # Display snapshot image and script notes side-by-side
                col_img, col_txt = st.columns([3, 2])
                with col_img:
                    st.image(snapshot_path, caption=f"2x3 Anomaly Context Grid: {snapshot_file}", use_container_width=True)
                with col_txt:
                    st.markdown(notes)
                    
                st.markdown("---")
    else:
        st.info("Please select a video file in the sidebar to review scene snapshots.")

with tab5:
    st.subheader("Isolated Anomaly Sub-clips Gallery")
    clips_dir = "anomaly_clips"
    os.makedirs(clips_dir, exist_ok=True)
    
    if selected_video:
        video_base = os.path.splitext(selected_video)[0]
        # Find clips matching the video basename
        clips = [
            f for f in os.listdir(clips_dir)
            if f.startswith(f"clip_{video_base}_") and f.endswith(".mp4")
        ]
        
        if not clips:
            st.info(f"No isolated anomaly sub-clips captured yet for '{selected_video}'. Run the pipeline to record clips.")
        else:
            st.markdown(f"Discovered **{len(clips)}** isolated sub-clip(s) for the selected video:")
            
            # Zip archive generation of all matching clips
            import zipfile
            import io
            
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for c_file in sorted(clips):
                    fp = os.path.join(clips_dir, c_file)
                    zip_file.write(fp, c_file)
            
            st.download_button(
                label="📥 Download All Anomaly Clips as ZIP",
                data=zip_buffer.getvalue(),
                file_name=f"{video_base}_anomalies.zip",
                mime="application/zip",
                type="secondary"
            )
            st.markdown("---")
            
            # Display clips in a structured grid/gallery
            for c_file in sorted(clips):
                c_path = os.path.join(clips_dir, c_file)
                st.markdown(f"##### 🎞️ Anomaly Clip: `{c_file}`")
                
                col_vid, col_dl = st.columns([3, 1])
                with col_vid:
                    st.video(c_path)
                with col_dl:
                    try:
                        with open(c_path, "rb") as f:
                            clip_bytes = f.read()
                        st.download_button(
                            label="💾 Download Clip",
                            data=clip_bytes,
                            file_name=c_file,
                            mime="video/mp4",
                            key=c_file
                        )
                    except Exception as e:
                        st.error(f"Error loading clip file: {e}")
                st.markdown("---")
    else:
        st.info("Select a video from the sidebar to inspect isolated anomaly snippets.")

with tab6:
    st.subheader("👥 Multi-Shot Re-ID Appearance Registry")
    registry_file = "reid_registry.json"
    
    if os.path.exists(registry_file):
        try:
            with open(registry_file, "r") as f:
                registry = json.load(f)
        except Exception as e:
            registry = {}
            st.error(f"Error loading Re-ID registry: {e}")
    else:
        registry = {}

    if not registry:
        st.info("No registered object/actor profiles found. Please run the tracking pipeline first.")
    else:
        st.markdown(f"Currently tracking **{len(registry)}** unique actor/prop profile(s) across scene transitions:")
        
        # Display each tracked ID profile
        for track_id, embedding in sorted(registry.items(), key=lambda x: int(x[0])):
            st.markdown(f"#### Profile: Object ID `{track_id}`")
            
            col_profile, col_meta = st.columns([1, 3])
            
            with col_profile:
                # Render registered crop profile image if exists
                thumbnail_path = f"anomaly_snapshots/thumbnail_id_{track_id}.png"
                if os.path.exists(thumbnail_path):
                    st.image(thumbnail_path, caption=f"Appearance Profile ID {track_id}", use_container_width=True)
                else:
                    st.warning("No thumbnail captured yet.")
                    
            with col_meta:
                st.markdown(f"**Appearance Signature Length:** `{len(embedding)}` dimensions")
                
                # Calculate similarity scores to other profiles
                similarities = []
                for other_id, other_embedding in registry.items():
                    if other_id != track_id:
                        sim_score = cosine_similarity(embedding, other_embedding)
                        similarities.append((other_id, sim_score))
                
                # Sort by similarity score descending
                similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
                if similarities:
                    st.markdown("**Appearance Similarity to Other Registered Profiles:**")
                    sim_list = ", ".join([f"ID {oid}: `{score:.2f}`" for oid, score in similarities[:4]])
                    st.markdown(sim_list)
                    
                # Look for events corresponding to this track ID in master logs to count discontinuities
                discontinuity_events = []
                if log_data and "events" in log_data:
                    for ev in log_data["events"]:
                        if str(ev.get("object_id")) == str(track_id) and ev.get("log_type") == "appearance_discontinuity":
                            discontinuity_events.append(ev)
                
                if discontinuity_events:
                    st.error(f"⚠️ Detected **{len(discontinuity_events)}** appearance discontinuity error(s) across shots!")
                    # Display the discontinuity messages
                    for dev in discontinuity_events:
                        st.markdown(f"- **Scene:** `{dev.get('video_name')}` (Frame {dev.get('frame')}) : *{dev.get('message')}*")
                else:
                    st.success("✅ Continuity verified: Stable appearance matches across all scene occurrences.")
            
            st.markdown("---")

with tab7:
    st.subheader("🔊 Audio-Visual Continuity & Jump Cut Monitor")
    
    if selected_video:
        video_base = os.path.splitext(selected_video)[0]
        
        # Load audio-visual metrics for the selected video
        if log_data and "metrics" in log_data and selected_video in log_data["metrics"]:
            m = log_data["metrics"][selected_video]
            audio_db_timeline = m.get("audio_db_timeline", [])
            object_movement_timeline = m.get("object_movement_timeline", [])
            
            if audio_db_timeline and object_movement_timeline:
                st.write("Below is the frame-by-frame mapping of maximum object displacement (motion magnitude) and audio RMS decibel levels:")
                
                # Align lengths
                min_len = min(len(audio_db_timeline), len(object_movement_timeline))
                frames = list(range(1, min_len + 1))
                
                # Build pandas dataframe
                df_av = pd.DataFrame({
                    "Frame": frames,
                    "Max Object Displacement (px)": object_movement_timeline[:min_len],
                    "Audio Energy (dB)": audio_db_timeline[:min_len]
                }).set_index("Frame")
                
                # Motion timeline chart
                st.markdown("##### 🏃‍♂️ Object Movement Magnitude (px/frame)")
                st.line_chart(df_av["Max Object Displacement (px)"], use_container_width=True)
                
                # Audio timeline chart
                st.markdown("##### 🔊 Audio RMS Energy levels (dB)")
                st.line_chart(df_av["Audio Energy (dB)"], use_container_width=True)
                
                st.markdown("---")
                st.subheader("🔗 Correlated Audio-Visual Sync Anomaly Logs")
                
                # Scan for AV jump cut events in logs
                av_events = []
                if log_data and "events" in log_data:
                    for ev in log_data["events"]:
                        if ev.get("video_name") == selected_video and ev.get("log_type") == "av_jump_cut":
                            av_events.append(ev)
                            
                if av_events:
                    st.error(f"Detected **{len(av_events)}** correlated AV Jump Cut anomalies!")
                    df_av_events = pd.DataFrame(av_events)
                    st.dataframe(df_av_events, use_container_width=True)
                else:
                    st.success("✅ No correlated audio-visual jump cut anomalies detected. Motion jumps match sound track cues perfectly.")
            else:
                st.info("Timeline metrics are missing. Please re-run the pipeline to compute audio-visual telemetry.")
        else:
            st.info("Telemetry data not found for this video. Run the pipeline to process audio-visual streams.")
    else:
        st.info("Select a video from the sidebar to review audio-visual alignment.")

with tab8:
    st.subheader("🧍 Skeletal Posture Continuity Tracking")
    
    if selected_video:
        if log_data and "metrics" in log_data and selected_video in log_data["metrics"]:
            m = log_data["metrics"][selected_video]
            posture_timeline = m.get("posture_timeline", {})
            
            if posture_timeline:
                st.write("Track and monitor skeletal configurations of human actors over time. Select a Tracked ID to visualize its joint angle profiles:")
                
                # Dropdown to select Tracked ID
                available_ids = list(posture_timeline.keys())
                selected_id = st.selectbox("Select Target ID to Profile", sorted(available_ids), key="pose_id_select")
                
                if selected_id:
                    history = posture_timeline[selected_id]
                    df_pose = pd.DataFrame(history)
                    
                    if not df_pose.empty:
                        # Set index to Frame
                        df_pose = df_pose.set_index("frame")
                        
                        # Plot Joint angles
                        st.markdown(f"##### 📈 Joint Angle Time Series for Actor ID {selected_id}")
                        st.line_chart(df_pose, use_container_width=True)
                        
                        # Show raw timeline data
                        with st.expander("Show Raw Skeletal Angles Table"):
                            st.dataframe(df_pose, use_container_width=True)
                            
                        # Show posture discontinuity events for this ID
                        posture_events = []
                        if log_data and "events" in log_data:
                            for ev in log_data["events"]:
                                if ev.get("video_name") == selected_video and ev.get("log_type") == "posture_discontinuity" and str(ev.get("object_id")) == str(selected_id):
                                    posture_events.append(ev)
                                    
                        if posture_events:
                            st.error(f"⚠️ Detected **{len(posture_events)}** Posture Discontinuity alert(s) for Actor ID {selected_id}!")
                            for pe in posture_events:
                                st.markdown(f"- **Frame {pe.get('frame')}:** {pe.get('message')}")
                        else:
                            st.success(f"✅ Actor ID {selected_id} displays consistent posture profiles across tracking gaps.")
                    else:
                        st.info("No posture tracking frames found for this ID.")
            else:
                st.info("No posture tracking data found for this scene. Note: Posture tracking targets human actors/props.")
        else:
            st.info("Telemetry data not found for this video. Run the pipeline to populate.")
    else:
        st.info("Select a video from the sidebar to review posture continuity profiles.")

with tab9:
    st.subheader("🧊 Monocular 3D Spatial Depth & Volume Engine")
    
    if selected_video:
        video_base = os.path.splitext(selected_video)[0]
        
        # Display colorized depth video if it exists!
        depth_video_path = f"output_videos/{video_base}_depth.mp4"
        if os.path.exists(depth_video_path):
            st.markdown("##### 🎥 Colorized 3D Depth Map Video Playback")
            st.video(depth_video_path)
        else:
            st.info("Colorized depth heatmap video not found. Please run the tracking pipeline first.")
            
        st.markdown("---")
        
        # Load depth timelines for selected video
        if log_data and "metrics" in log_data and selected_video in log_data["metrics"]:
            m = log_data["metrics"][selected_video]
            depth_timeline = m.get("depth_timeline", {})
            
            if depth_timeline:
                st.write("Monitor Z-axis distance (depth) trajectories of tracked entities over time (0.0 represents Near/Foreground, 1.0 represents Far/Background):")
                
                # Dropdown to select Tracked ID
                available_ids = list(depth_timeline.keys())
                selected_id = st.selectbox("Select Target ID to Profile", sorted(available_ids), key="depth_id_select")
                
                if selected_id:
                    history = depth_timeline[selected_id]
                    df_depth = pd.DataFrame(history)
                    
                    if not df_depth.empty:
                        # Set index to Frame
                        df_depth = df_depth.set_index("frame")
                        
                        # Plot Z-depth
                        st.markdown(f"##### 📈 Z-Depth (Disparity Distance) for Object ID {selected_id}")
                        st.line_chart(df_depth["z"], use_container_width=True)
                        
                        # Show raw timeline data
                        with st.expander("Show Raw Z-Distance Table"):
                            st.dataframe(df_depth, use_container_width=True)
                            
                        # Show 3D spatial jump events for this ID
                        depth_events = []
                        if log_data and "events" in log_data:
                            for ev in log_data["events"]:
                                if ev.get("video_name") == selected_video and ev.get("log_type") == "3d_spatial_jump" and str(ev.get("object_id")) == str(selected_id):
                                    depth_events.append(ev)
                                    
                        if depth_events:
                            st.error(f"⚠️ Detected **{len(depth_events)}** 3D Spatial Jump alert(s) for Object ID {selected_id}!")
                            for de in depth_events:
                                st.markdown(f"- **Frame {de.get('frame')}:** {de.get('message')}")
                        else:
                            st.success(f"✅ Object ID {selected_id} displays stable Z-depth trajectories across frames.")
                    else:
                        st.info("No depth frames found for this ID.")
            else:
                st.info("No depth tracking data found for this scene.")
        else:
            st.info("Telemetry data not found for this video. Run the pipeline to populate.")
    else:
        st.info("Select a video from the sidebar to review depth maps.")

with tab10:
    st.subheader("🎨 Generative VFX Fixes & Inpainted Backplates")
    st.markdown("This tab displays side-by-side interactive image comparisons of flagged continuity anomalies before and after applying zero-dependency local inpainting.")
    
    vfx_dir = "output_clips/vfx_patches"
    if os.path.exists(vfx_dir):
        # Scan for original files matching the selected video base name
        video_base = os.path.splitext(selected_video)[0] if selected_video else ""
        all_files = os.listdir(vfx_dir)
        
        # Filter files that end with _original.png and match selected video base name
        original_files = sorted([f for f in all_files if f.endswith("_original.png") and f.startswith(video_base)])
        
        if original_files:
            st.markdown(f"Detected **{len(original_files)}** VFX patch candidates for `{selected_video}`:")
            for orig_file in original_files:
                patched_file = orig_file.replace("_original.png", "_patched.png")
                patched_path = os.path.join(vfx_dir, patched_file)
                orig_path = os.path.join(vfx_dir, orig_file)
                
                if os.path.exists(patched_path):
                    # Extract meta details from filename
                    # Format: {video_base}_{log_type}_id{object_id}_frame{frame_num}_original.png
                    meta = orig_file.replace("_original.png", "").replace(f"{video_base}_", "").split("_")
                    log_type = "Anomaly"
                    obj_id = "N/A"
                    frame_num = "N/A"
                    if len(meta) >= 3:
                        log_type = meta[0].replace("-", " ").title()
                        obj_id = meta[1].replace("id", "")
                        frame_num = meta[2].replace("frame", "")
                        
                    with st.expander(f"🔍 {log_type} Fix - ID {obj_id} (Frame {frame_num})", expanded=True):
                        col_orig, col_patched = st.columns(2)
                        with col_orig:
                            st.markdown("**Original Anomaly Frame**")
                            st.image(orig_path, use_container_width=True)
                        with col_patched:
                            st.markdown("**Generative VFX Clean Frame**")
                            st.image(patched_path, use_container_width=True)
                            
                        # Retrieve director's notes for this frame/object
                        anomaly_id = f"{video_base}_{meta[0]}_{obj_id}_frame{frame_num}"
                        snapshot_name = f"anomaly_frame_{video_base}_{meta[0]}_{obj_id}_frame{frame_num}.png"
                        snapshot_path = os.path.join("anomaly_snapshots", snapshot_name)
                        if os.path.exists(snapshot_path):
                            if st.button(f"🎬 Generate Director Notes for Frame {frame_num}", key=f"notes_{orig_file}"):
                                notes = inspect_anomaly(snapshot_path)
                                st.info(notes)
                else:
                    st.warning(f"Patched frame not found for {orig_file}")
        else:
            st.info(f"No VFX patches generated for `{selected_video}`. Run `python3 guardian.py inspect` to run the inpainting engine.")
    else:
        st.info("VFX patch folder output_clips/vfx_patches does not exist yet. Run `python3 guardian.py inspect` first.")

with tab11:
    st.subheader("📜 Script-to-Screen Compliance Verification")
    st.markdown("Cross-reference screenplay directions against actual object classes and people detected in the video sequences.")
    
    script_file = "sample_script.txt"
    if os.path.exists(script_file):
        from script_parser import ScriptParser
        parser = ScriptParser()
        scenes = parser.parse_script_file(script_file)
        
        # Determine which scene corresponds to selected video
        matching_scene = None
        for scene in scenes:
            slugline = scene["slugline"].upper()
            if "HORSE" in slugline or "FIELD" in slugline:
                if "scene2" in selected_video or "test" in selected_video:
                    matching_scene = scene
            elif "LIVING" in slugline or "ROOM" in slugline:
                if "living" in selected_video:
                    matching_scene = scene
                    
        # Fallbacks for demonstration
        if not matching_scene and scenes:
            if "test" in selected_video:
                matching_scene = scenes[0]
            else:
                matching_scene = scenes[1] if len(scenes) > 1 else scenes[0]
                
        if matching_scene:
            st.markdown(f"#### Screenplay Scene {matching_scene['scene_number']}: `{matching_scene['slugline']}`")
            
            col_script, col_compliance = st.columns([1, 1])
            
            with col_script:
                st.markdown("##### 📝 Screenplay Excerpt")
                script_excerpt = "\n\n".join(matching_scene["content"])
                st.info(script_excerpt)
                
            with col_compliance:
                st.markdown("##### 🔍 Compliance Checklist")
                # Get detected classes for this video
                detected_classes = []
                if log_data and "metrics" in log_data and selected_video in log_data["metrics"]:
                    detected_classes = [c.lower() for c in log_data["metrics"][selected_video].get("detected_classes", [])]
                
                # Map COCO classes to script synonyms
                class_synonyms = {
                    "phone": ["cell phone", "phone"],
                    "glass": ["cup", "glass", "bottle", "wine glass"],
                    "bench": ["bench", "chair", "sofa"],
                    "keys": ["keys", "key"]
                }
                
                st.markdown("**Expected Props:**")
                for prop in matching_scene["props"]:
                    prop_lower = prop.lower()
                    matched = False
                    if prop_lower in detected_classes:
                        matched = True
                    else:
                        for s in class_synonyms.get(prop_lower, []):
                            if s in detected_classes:
                                matched = True
                                break
                    if matched:
                        st.success(f"✅ Prop `{prop}`: DETECTED")
                    else:
                        st.error(f"❌ Prop `{prop}`: MISSING")
                        
                # Check characters
                st.markdown("**Expected Characters:**")
                for char in matching_scene["characters"]:
                    if "person" in detected_classes:
                        st.success(f"✅ Character `{char}`: PRESENCE CONFIRMED")
                    else:
                        st.error(f"❌ Character `{char}`: MISSING (No person detected)")
                        
            # Show script deviation warnings if any are logged in events
            deviations = []
            if log_data and "events" in log_data:
                for ev in log_data["events"]:
                    if ev.get("video_name") == selected_video and ev.get("log_type") == "script_deviation":
                        deviations.append(ev)
                        
            if deviations:
                st.warning(f"⚠️ Flagged **{len(deviations)}** Script Compliance Deviations:")
                for dev in deviations:
                    st.error(dev.get("message"))
            else:
                st.success("✅ Complete script-to-screen compliance matching successfully verified.")
        else:
            st.info("No corresponding script scene found for this video.")
    else:
        st.info("No script file `sample_script.txt` found in the root directory. Run script verification first.")



