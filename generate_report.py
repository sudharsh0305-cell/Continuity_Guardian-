import json
import os
import matplotlib.pyplot as plt
import numpy as np

def main():
    log_file = "master_continuity_log.json"
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found. Please run guardian_tracker.py first.")
        return
        
    with open(log_file, "r") as f:
        data = json.load(f)
        
    # Handle dictionary structure (containing both metrics and events) or raw list structure
    if isinstance(data, dict) and "events" in data:
        events = data["events"]
        metrics = data.get("metrics", {})
    else:
        events = data
        metrics = {}
        
    total_events = len(events)
    
    print("\n" + "="*70)
    print("           BATCH CONTINUITY GUARD SUMMARY REPORT")
    print("="*70)
    
    if total_events == 0:
        print("No events logged across any videos.")
        print("="*70 + "\n")
    else:
        # Aggregate data by video name and log type
        video_data = {}
        for entry in events:
            video = entry.get("video_name", "unknown_video")
            log_type = entry.get("log_type", "unknown_type")
            if video not in video_data:
                video_data[video] = {"spatial_jump": 0, "vanishing": 0}
            if log_type in video_data[video]:
                video_data[video][log_type] += 1
                
        print(f"Total Videos Processed: {len(video_data) if video_data else len(metrics)}")
        print(f"Total Logged Events:    {total_events}\n")
        print(f"{'Video Name':<35} | {'Spatial Jumps':<13} | {'Vanished Objects':<16}")
        print("-"*72)
        for v in sorted(video_data.keys()):
            jumps = video_data[v]["spatial_jump"]
            vanished = video_data[v]["vanishing"]
            print(f"{v:<35} | {jumps:<13} | {vanished:<16}")
        print("="*70 + "\n")
        
    if metrics:
        print("           EXECUTION PERFORMANCE METRICS")
        print("="*70)
        print(f"{'Video Name':<35} | {'Proc Time (s)':<13} | {'Raw Inf (FPS)':<13} | {'Pipeline (FPS)':<14}")
        print("-"*72)
        for v in sorted(metrics.keys()):
            m = metrics[v]
            pt = m.get("total_process_time_sec", 0.0)
            rif = m.get("raw_inference_fps", 0.0)
            opf = m.get("overall_pipeline_fps", 0.0)
            print(f"{v:<35} | {pt:<13} | {rif:<13} | {opf:<14}")
        print("="*70 + "\n")

    if total_events == 0:
        return
        
    # Re-aggregate video_data for plotting in case total_events > 0
    video_data = {}
    for entry in events:
        video = entry.get("video_name", "unknown_video")
        log_type = entry.get("log_type", "unknown_type")
        if video not in video_data:
            video_data[video] = {"spatial_jump": 0, "vanishing": 0}
        if log_type in video_data[video]:
            video_data[video][log_type] += 1

    # Plotting using premium dark aesthetics
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    
    videos = sorted(list(video_data.keys()))
    x = np.arange(len(videos))
    width = 0.35
    
    jumps = [video_data[v]["spatial_jump"] for v in videos]
    vanished = [video_data[v]["vanishing"] for v in videos]
    
    # Plot bars
    bar1 = ax.bar(x - width/2, jumps, width, label='Spatial Jump (>50px)', color='#FF5E3A', alpha=0.9, edgecolor='#FFFFFF', linewidth=0.5)
    bar2 = ax.bar(x + width/2, vanished, width, label='Vanished Object (>5 frames)', color='#00F0FF', alpha=0.9, edgecolor='#FFFFFF', linewidth=0.5)
    
    # Adding text labels on top of the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f'{height}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, color='#FFFFFF', fontweight='bold')

    autolabel(bar1)
    autolabel(bar2)
    
    # Styling and labeling
    ax.set_title("Batch Continuity Guardian - Anomaly Summary", fontsize=16, fontweight='bold', pad=20, color='#FFFFFF')
    ax.set_xlabel("Processed Videos", fontsize=12, labelpad=10, color='#BBBBBB')
    ax.set_ylabel("Number of Anomalies", fontsize=12, labelpad=10, color='#BBBBBB')
    ax.set_xticks(x)
    ax.set_xticklabels(videos, rotation=15, ha='right', fontsize=10, color='#E0E0E0')
    
    # Grid lines
    ax.grid(True, linestyle=':', alpha=0.3, color='#666666', axis='y')
    
    # Set y limit slightly higher to fit labels
    max_height = max(max(jumps) if jumps else 0, max(vanished) if vanished else 0)
    ax.set_ylim(0, max_height * 1.15 if max_height > 0 else 10)
    
    # Legend
    ax.legend(loc='upper right', frameon=True, facecolor='#222222', edgecolor='#555555', fontsize=10)
    
    # Borders
    for spine in ax.spines.values():
        spine.set_color('#444444')
        spine.set_linewidth(1.2)
        
    plt.tight_layout()
    output_image = "batch_continuity_report.png"
    plt.savefig(output_image, bbox_inches='tight')
    plt.close()
    
    print(f"Success: Aggregate report chart saved as '{output_image}'\n")

if __name__ == "__main__":
    main()
