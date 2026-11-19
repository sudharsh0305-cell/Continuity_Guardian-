import json
import os
import math
import xml.etree.ElementTree as ET
from xml.dom import minidom

def frame_to_timecode(frame_num, fps=30.0):
    """
    Converts a frame count into a standard timecode string (HH:MM:SS:FF).
    """
    total_seconds = frame_num / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frames = int(round((total_seconds - int(total_seconds)) * fps))
    
    # Normalize values
    if frames >= int(fps):
        frames = 0
        seconds += 1
    if seconds >= 60:
        seconds = 0
        minutes += 1
    if minutes >= 60:
        minutes = 0
        hours += 1
        
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

def get_anomaly_color(log_type):
    """
    Maps anomaly types to standard markers color codes for Resolve/Premiere.
    """
    colors = {
        "vanishing": "RED",
        "spatial_jump": "ORANGE",
        "av_jump_cut": "BLUE",
        "appearance_discontinuity": "PURPLE"
    }
    return colors.get(log_type.lower(), "GREEN")

def export_edl(master_log_path="master_continuity_log.json"):
    """
    Parses unified logs and outputs marker details in CMX3600 Edit Decision List format.
    """
    if not os.path.exists(master_log_path):
        return "Error: master_continuity_log.json not found."
        
    try:
        with open(master_log_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error loading log: {e}"
        
    events = data.get("events", [])
    
    edl_lines = [
        "TITLE: Continuity Guardian Timeline Markers",
        "FCM: NON-DROP FRAME",
        ""
    ]
    
    for idx, ev in enumerate(events, 1):
        video_name = ev.get("video_name", "Unknown")
        frame = ev.get("frame", 1)
        log_type = ev.get("log_type", "anomaly")
        object_id = ev.get("object_id", "Unknown")
        message = ev.get("message", "")
        
        # Estimate FPS if logged in metrics
        fps = 30.0
        if "metrics" in data and video_name in data["metrics"]:
            # If we had overall frame rate we would read it, else default to 30.0
            pass
            
        tc_in = frame_to_timecode(frame, fps)
        tc_out = frame_to_timecode(frame + 1, fps)
        
        color = get_anomaly_color(log_type)
        name = f"ID {object_id} {log_type.upper()}"
        
        edl_lines.append(f"{idx:03d}  AX       V     C        {tc_in} {tc_out} {tc_in} {tc_out}")
        edl_lines.append(f"* FROM CLIP: {video_name}")
        edl_lines.append(f"* KEYFRAME MARKER: COLOR={color} NAME={name} NOTE={message}")
        edl_lines.append("")
        
    return "\n".join(edl_lines)

def export_fcpxml(master_log_path="master_continuity_log.json"):
    """
    Converts unified log events into a Final Cut Pro XML (FCPXML v1.9) file format.
    """
    if not os.path.exists(master_log_path):
        return "Error: master_continuity_log.json not found."
        
    try:
        with open(master_log_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error loading log: {e}"
        
    events = data.get("events", [])
    
    fcpxml = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(fcpxml, "resources")
    ET.SubElement(resources, "format", id="r1", name="FFVideoFormat1080p2997", frameDuration="1001/30000s")
    
    library = ET.SubElement(fcpxml, "library")
    event_el = ET.SubElement(library, "event", name="Continuity Guardian Markers")
    project = ET.SubElement(event_el, "project", name="Continuity Logs")
    sequence = ET.SubElement(project, "sequence", duration="600s", format="r1", tcStart="0s")
    spine = ET.SubElement(sequence, "spine")
    gap = ET.SubElement(spine, "gap", name="Timeline", offset="0s", duration="600s")
    
    for ev in events:
        video_name = ev.get("video_name", "Unknown")
        frame = ev.get("frame", 1)
        log_type = ev.get("log_type", "anomaly")
        object_id = ev.get("object_id", "Unknown")
        message = ev.get("message", "")
        
        fps = 30.0
        start_seconds = frame / fps
        
        marker = ET.SubElement(gap, "marker")
        marker.set("start", f"{start_seconds:.3f}s")
        marker.set("duration", "1s")
        marker.set("value", f"ID {object_id} {log_type.upper()}")
        marker.set("note", f"Clip: {video_name} | {message}")
        
    # Reformat with minidom for pretty printing
    xml_str = ET.tostring(fcpxml, encoding="utf-8")
    parsed_xml = minidom.parseString(xml_str)
    return parsed_xml.toprettyxml(indent="    ")
