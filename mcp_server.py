from fastmcp import FastMCP
import os
import json
import subprocess
import sys

# Import local pipeline modules
from nle_exporter import export_edl, export_fcpxml
from anomaly_inspector import inspect_anomaly

# Initialize the MCP server
mcp = FastMCP("Continuity Guardian")

@mcp.tool()
def scan_video(video_path: str) -> str:
    """
    Runs the YOLO tracking and continuity pipeline on a target video file.
    
    Args:
        video_path: Path to the target video file (e.g. 'input_videos/test_scene.mp4')
    """
    if not os.path.exists(video_path):
        return f"Error: Video file not found at path: {video_path}"
        
    input_dir = os.path.dirname(video_path) or "input_videos"
    video_name = os.path.basename(video_path)
    
    cmd = ["python3", "guardian_tracker.py", "--input-dir", input_dir, "--video", video_name, "--no-show"]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return f"Scan completed successfully for {video_name}!\n\nStdout:\n{res.stdout}"
    except subprocess.CalledProcessError as e:
        return f"Scan failed with error:\n{e.stderr}\n\nStdout:\n{e.stdout}"
    except Exception as ex:
        return f"Unexpected error while running scan: {ex}"

@mcp.tool()
def get_anomaly_log(video_name: str) -> str:
    """
    Retrieves the list of continuity anomalies detected for a specific video.
    
    Args:
        video_name: Name of the video file (e.g. 'test_scene.mp4')
    """
    log_path = "master_continuity_log.json"
    if not os.path.exists(log_path):
        return "No anomaly log file found. Run a scan first."
        
    try:
        with open(log_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return f"Error loading logs: {e}"
        
    events = data.get("events", [])
    filtered = [ev for ev in events if ev.get("video_name") == video_name or video_name in ev.get("video_name", "")]
    
    if not filtered:
        return f"No anomalies detected for video: '{video_name}'"
        
    return json.dumps(filtered, indent=2)

@mcp.tool()
def export_nle_markers(format: str) -> str:
    """
    Generates NLE marker timelines in EDL (Premiere/DaVinci) or FCPXML (Final Cut) format.
    
    Args:
        format: Format style, either 'edl' or 'fcpxml'
    """
    fmt = format.lower().strip()
    if fmt == "edl":
        return export_edl()
    elif fmt == "fcpxml":
        return export_fcpxml()
    else:
        return "Error: Unsupported format. Choose 'edl' or 'fcpxml'."

@mcp.tool()
def get_director_notes() -> str:
    """
    Compiles AI-vision and rule-based Director's Notes for all detected anomalies.
    """
    snapshots_dir = "anomaly_snapshots"
    if not os.path.exists(snapshots_dir):
        return "No anomaly snapshots found. Run a scan first."
        
    try:
        snapshots = [f for f in os.listdir(snapshots_dir) if f.startswith("anomaly_frame_") and f.endswith(".png")]
    except Exception as e:
        return f"Error listing snapshots directory: {e}"
        
    if not snapshots:
        return "No anomaly snapshot images found to generate notes."
        
    notes_list = []
    for snap in sorted(snapshots):
        path = os.path.join(snapshots_dir, snap)
        notes = inspect_anomaly(path)
        notes_list.append(notes)
        
    return "\n\n---\n\n".join(notes_list)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        print("Running tool definitions test pass...")
        print("• scan_video tool description:", scan_video.__doc__.split('\n')[1])
        print("• get_anomaly_log tool description:", get_anomaly_log.__doc__.split('\n')[1])
        print("• export_nle_markers tool description:", export_nle_markers.__doc__.split('\n')[1])
        print("• get_director_notes tool description:", get_director_notes.__doc__.split('\n')[1])
        print("MCP test pass successful!")
    else:
        # Run stdio transport server
        mcp.run()
