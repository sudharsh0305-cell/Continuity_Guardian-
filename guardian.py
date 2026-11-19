import streamlit as st
import json
import os
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Continuity Guardian Dashboard",
    page_icon="🎬",
    layout="wide"
)

MASTER_LOG_PATH = "master_continuity_log.json"
SNAPSHOTS_FOLDER = "anomaly_snapshots"
CLIPS_FOLDER = "anomaly_clips"

def load_master_data():
    if os.path.exists(MASTER_LOG_PATH):
        try:
            with open(MASTER_LOG_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading master log: {e}")
    return {"metrics": {}, "events": []}

def format_bbox(box):
    """Formats bounding box array into human-readable coordinates."""
    if isinstance(box, list) and len(box) == 4:
        return f"[{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]"
    return "N/A"

def generate_dynamic_notes(event):
    """
    Generates granular, context-aware Director's Notes tailored to specific frame data.
    Eliminates static template text completely.
    """
    log_type = event.get("log_type", "spatial_jump")
    obj_id = event.get("object_id", "N/A")
    frame = event.get("frame", 0)
    box = event.get("box", None)
    
    pixel_shift = event.get("pixel_shift", None)
    posture_dev = event.get("posture_deviation", None)
    depth_delta = event.get("depth_delta", None)
    sim_score = event.get("similarity_score", None)
    last_center = event.get("last_known_center", None)

    # 1. Critical Vanishing Events
    if log_type == "vanishing":
        box_str = format_bbox(box)
        loc_str = f"at central coordinates ({last_center[0]}, {last_center[1]})" if last_center else f"within bounding box {box_str}"
        
        obs = (
            f"Object ID #{obj_id} was actively tracked {loc_str} during the pre-anomaly window. "
            f"At Frame {frame}, the tracking bounding box dissolved immediately without approaching screen margins."
        )
        action = (
            f"Inspect frame sequence around Frame {frame} for prop removal, heavy foreground occlusion, or lighting drops. "
            f"If Object #{obj_id} was obscured by an actor, add a manual tracking anchor or adjust occlusion bounds."
        )

    # 2. Posture & Pose Discontinuities
    elif log_type == "posture_discontinuity":
        dev_str = f"{posture_dev:.1f}°" if posture_dev is not None else "exceeding threshold"
        obs = (
            f"Subject ID #{obj_id} experienced an abrupt skeletal pose orientation flip of {dev_str} "
            f"between adjacent keyframes near Frame {frame}."
        )
        action = (
            f"Review actor stance and limb angles at Frame {frame}. "
            f"Match shoulder/torso alignment across the cut boundary to prevent visual pop."
        )

    # 3. 2D Spatial Shifts & AV Jump Cuts
    elif log_type in ["spatial_jump", "av_jump_cut"]:
        shift_str = f"{pixel_shift:.1f}px" if pixel_shift is not None else "significant"
        is_av = (log_type == "av_jump_cut")
        
        av_note = " (Synchronized with an audio decibel break)" if is_av else ""
        obs = (
            f"Object ID #{obj_id} underwent a spatial jump of {shift_str}{av_note} "
            f"across frame transition at Frame {frame}."
        )
        if is_av:
            action = (
                f"Audio break detected concurrently at Frame {frame}. "
                f"Perform a hard splice trim or align audio L-cut/J-cut transition to conceal the position jump."
            )
        else:
            action = (
                f"Review spatial blocking at Frame {frame}. Apply a cross-dissolve, frame trim, "
                f"or crop alignment to smooth the {shift_str} spatial jump."
            )

    # 4. 3D Spatial Depth Shifts
    elif log_type == "3d_spatial_jump":
        depth_str = f"{depth_delta:.3f} Z-units" if depth_delta is not None else "exceeding confidence bounds"
        obs = (
            f"Object ID #{obj_id} exhibited a sudden monocular depth shift of {depth_str} "
            f"at Frame {frame} in relative 3D camera space."
        )
        action = (
            f"Check camera focal length or lens distance variations around Frame {frame}. "
            f"Confirm physical depth positioning between subject and camera."
        )

    # 5. Appearance & Wardrobe/Lighting Discontinuities
    elif log_type == "appearance_discontinuity":
        sim_str = f"{sim_score:.2f}" if sim_score is not None else "< 0.75"
        obs = (
            f"Object ID #{obj_id} feature extraction returned an appearance vector similarity score of {sim_str} "
            f"at Frame {frame}, indicating costume, prop alteration, or major lighting shift."
        )
        action = (
            f"Compare color grade, illumination contrast, and costume details for Object #{obj_id} "
            f"before and after Frame {frame}."
        )

    # 6. Fallback for unclassified events
    else:
        obs = f"Object ID #{obj_id} exhibited tracking anomalies at Frame {frame}."
        action = f"Inspect keyframes around Frame {frame} for visual matching."

    return obs, action

def main():
    st.title("🎬 Continuity Guardian Dashboard")
    st.caption("Automated Continuity & Spatial Jump Analysis Pipeline")

    data = load_master_data()
    events = data.get("events", [])
    metrics = data.get("metrics", {})

    # Sidebar Pipeline Configuration
    st.sidebar.header("⚙️ Pipeline Configuration")
    video_list = list(metrics.keys()) if metrics else ["No processed videos found"]
    selected_video = st.sidebar.selectbox("Select Target Video", video_list)

    # Main Dashboard Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Telemetry Overview", 
        "🎬 Director's Desk", 
        "📋 Anomaly Event Logs"
    ])

    with tab1:
        st.subheader("Pipeline Telemetry & Execution Summary")
        if events:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Flagged Anomalies", len(events))
            
            if selected_video in metrics:
                vm = metrics[selected_video]
                col2.metric("Pipeline Speed", f"{vm.get('overall_pipeline_fps', 0)} FPS")
                col3.metric("Raw Inference Speed", f"{vm.get('raw_inference_fps', 0)} FPS")
            
            df_events = pd.DataFrame(events)
            if "log_type" in df_events.columns:
                st.write("### Anomaly Distribution")
                st.bar_chart(df_events["log_type"].value_counts())
        else:
            st.info("No telemetry logs found. Run tracker pipeline to generate logs.")

    with tab2:
        st.subheader("🎬 AI Script Supervisor's Inspection Desk")
        
        if not events:
            st.warning("No anomalies logged in master record.")
        else:
            # Filter events for selected video if available
            video_events = [e for e in events if e.get("video_name") == selected_video] if selected_video in metrics else events
            
            st.write(f"Displaying **{len(video_events)}** context-analyzed keyframe anomalies:")
            
            for idx, event in enumerate(video_events):
                st.markdown("---")
                col_img, col_notes = st.columns([1.2, 1])
                
                log_type = event.get("log_type", "spatial_jump")
                obj_id = event.get("object_id", "N/A")
                frame_idx = event.get("frame", 0)
                video_name = event.get("video_name", "N/A")
                video_base = os.path.splitext(video_name)[0]
                
                snapshot_name = f"anomaly_frame_{video_base}_{log_type}_{obj_id}_frame{frame_idx}.png"
                snapshot_path = os.path.join(SNAPSHOTS_FOLDER, snapshot_name)
                
                with col_img:
                    if os.path.exists(snapshot_path):
                        st.image(snapshot_path, use_column_width=True, caption=f"2x3 Anomaly Context Grid: {snapshot_name}")
                    else:
                        st.info(f"🖼️ Keyframe Grid: Object #{obj_id} @ Frame {frame_idx}")

                with col_notes:
                    st.markdown("### 📝 DIRECTOR'S NOTES")
                    st.write(f"**Scene**: `{video_name}` | **Timecode**: Frame {frame_idx}")
                    st.write(f"**Subject**: Object ID #{obj_id}")
                    
                    st.warning(f"⚠️ **Anomaly Detected**: {log_type.upper().replace('_', ' ')}")
                    
                    obs, action = generate_dynamic_notes(event)
                    
                    st.markdown(f"**Observations**: {obs}")
                    st.markdown(f"**Correction Action Plan**: {action}")

    with tab3:
        st.subheader("Detailed Anomaly Log Table")
        if events:
            st.dataframe(pd.DataFrame(events), use_container_width=True)
        else:
            st.info("No logs available.")

if __name__ == "__main__":
    main()