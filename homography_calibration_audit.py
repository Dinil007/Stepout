"""
Homography Calibration Audit
Performs complete calibration analysis without modifying production code.
Generates visualizations, reprojection error analysis, and scale variation heatmap.
"""
import json
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

OUTPUT_DIR = Path("outputs")
CONFIG_PATH = Path("configs/homography_calibration.json")
FRAME_SAMPLE = Path("D:/stepout/videos/raw/match30.mp4")

# Field dimensions
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0


def load_calibration():
    """Load homography calibration from JSON."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def compute_reprojection_error(src_pts, dst_pts, H):
    """Compute reprojection error for calibration points."""
    # Forward: src -> dst
    projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2).astype(np.float32), H)
    error_fwd = np.sqrt(np.sum((projected.reshape(-1, 2) - dst_pts) ** 2, axis=1))
    
    # Inverse: dst -> src
    H_inv = np.linalg.inv(H)
    reprojected = cv2.perspectiveTransform(dst_pts.reshape(-1, 1, 2).astype(np.float32), H_inv)
    error_inv = np.sqrt(np.sum((reprojected.reshape(-1, 2) - src_pts) ** 2, axis=1))
    
    return error_fwd, error_inv, projected.reshape(-1, 2), reprojected.reshape(-1, 2)


def compute_scale_grid(H, canvas_size=(1050, 680), step=20):
    """Compute metres per pixel across canvas grid."""
    x = np.arange(0, canvas_size[0] + 1, step)
    y = np.arange(0, canvas_size[1] + 1, step)
    xx, yy = np.meshgrid(x, y)
    
    meters_per_px = np.zeros_like(xx, dtype=np.float32)
    
    # For each grid point, compute local scale by sampling nearby points
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            cx, cy = xx[i, j], yy[i, j]
            dx = xx[i, j] + 1  # 1 pixel right
            dy = yy[i, j] + 1  # 1 pixel down
            
            # Transform to field coords
            pts = np.array([[cx, cy], [dx, cy], [cx, dy]], dtype=np.float32).reshape(-1, 1, 2)
            field_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
            
            # Compute scale: m/px
            if field_pts[0, 0] > 0:
                scale_x = np.abs(field_pts[1, 0] - field_pts[0, 0])
                scale_y = np.abs(field_pts[2, 1] - field_pts[0, 1])
                meters_per_px[i, j] = (scale_x + scale_y) / 2.0
            else:
                meters_per_px[i, j] = np.nan
    
    return xx, yy, meters_per_px


def draw_calibration_visualization(calib):
    """Draw frame with calibration overlays."""
    cap = cv2.VideoCapture(str(FRAME_SAMPLE))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Warning: Could not read video frame. Creating synthetic background.")
        frame = np.zeros((680, 1050, 3), dtype=np.uint8)
        frame[:] = (34, 139, 34)  # Green
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    src = np.array(calib['calibration_points']['source'], dtype=np.float32)
    dst = np.array(calib['calibration_points']['destination'], dtype=np.float32)
    
    # Draw pitch polygon
    polygon = patches.Polygon(src, fill=False, edgecolor='green', linewidth=2, label='Pitch Polygon')
    ax.add_patch(polygon)
    
    # Draw homography source points
    ax.scatter(src[:, 0], src[:, 1], c='blue', s=100, marker='o', label='Homography Source Points', zorder=5)
    for i, pt in enumerate(src):
        ax.annotate(f"P{i+1}", (pt[0], pt[1]), textcoords="offset points", xytext=(5, 5), fontsize=10, color='blue')
    
    # Draw homography destination points (scaled to canvas)
    dst_scaled = dst * np.array([10, 10])  # Scale to canvas coordinates
    ax.scatter(dst_scaled[:, 0], dst_scaled[:, 1], c='cyan', s=100, marker='s', label='Homography Dest (scaled)', zorder=5)
    for i, pt in enumerate(dst_scaled):
        ax.annotate(f"D{i+1}", (pt[0], pt[1]), textcoords="offset points", xytext=(5, 5), fontsize=10, color='cyan')
    
    # Draw standard pitch markings (touchlines, goal lines, penalty boxes, centre circle)
    # All in canvas coordinates (1050x680)
    pitch_color = 'white'
    lw = 1.5
    
    # Touchlines
    ax.plot([0, 1050], [0, 0], pitch_color, linewidth=lw)  # Top touchline
    ax.plot([0, 1050], [680, 680], pitch_color, linewidth=lw)  # Bottom touchline
    ax.plot([0, 0], [0, 680], pitch_color, linewidth=lw)  # Left touchline
    ax.plot([1050, 1050], [0, 680], pitch_color, linewidth=lw)  # Right touchline
    
    # Goal lines (actually drawn as part of pitch outline, but specifying clearly)
    ax.plot([0, 1050], [0, 0], 'red', linewidth=2, label='Goal/Touch Lines')  # Top goal line
    ax.plot([0, 1050], [680, 680], 'red', linewidth=2)  # Bottom goal line
    
    # Halfway line
    ax.plot([525, 525], [0, 680], pitch_color, linewidth=lw, label='Halfway Line')
    
    # Centre circle (radius ~9.15m => 91.5 px in canvas)
    circle = plt.Circle((525, 340), 91.5, color=pitch_color, fill=False, linewidth=lw, label='Centre Circle')
    ax.add_patch(circle)
    ax.plot(525, 340, 'wo', markersize=5)  # Centre spot
    
    # Penalty areas
    # Left penalty area: 16.5m from goal line, 40.3m wide (centered)
    left_pen_x = 0
    left_pen_y = (680 - 403) / 2
    rect_left = patches.Rectangle((left_pen_x, left_pen_y), 165, 403, linewidth=lw, edgecolor=pitch_color, facecolor='none')
    ax.add_patch(rect_left)
    
    # Right penalty area
    right_pen_x = 1050 - 165
    right_pen_y = (680 - 403) / 2
    rect_right = patches.Rectangle((right_pen_x, right_pen_y), 165, 403, linewidth=lw, edgecolor=pitch_color, facecolor='none')
    ax.add_patch(rect_right)
    
    # Goal boxes (6-yard box)
    # Left goal box: 5.5m from goal, 18.3m wide
    left_gx = 0
    left_gy = (680 - 183) / 2
    rect_left_g = patches.Rectangle((left_gx, left_gy), 55, 183, linewidth=lw, edgecolor=pitch_color, facecolor='none')
    ax.add_patch(rect_left_g)
    
    right_gx = 1050 - 55
    right_gy = (680 - 183) / 2
    rect_right_g = patches.Rectangle((right_gx, right_gy), 55, 183, linewidth=lw, edgecolor=pitch_color, facecolor='none')
    ax.add_patch(rect_right_g)
    
    ax.set_xlim(0, 1050)
    ax.set_ylim(680, 0)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.set_title('Homography Calibration Overlay\nPitch Polygon, Source Points, and Standard Markings')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "calibration_overlay.png", dpi=150)
    plt.close()
    print(f"[OK] Saved calibration overlay to {OUTPUT_DIR / 'calibration_overlay.png'}")


def generate_report(calib, src_pts, dst_pts, error_fwd, error_inv, reproj_fwd, reproj_inv, rms_fwd, rms_inv, scale_stats):
    """Generate calibration audit report."""
    report = []
    report.append("# Homography Calibration Audit Report\n")
    report.append("## Configuration\n")
    report.append(f"Field dimensions: {FIELD_LENGTH_M} m x {FIELD_WIDTH_M} m")
    report.append(f"Canvas dimensions: {calib['image_dimensions']['width_px']} x {calib['image_dimensions']['height_px']} px")
    report.append(f"Calibration method: {calib['method']}")
    report.append(f"Validation passed: {calib['validation']['validation_passed']}")
    report.append(f"Configured mean reprojection error: {calib['validation']['mean_reprojection_error']} px")
    report.append("")
    
    # Source point analysis
    report.append("## Calibration Points: Source -> Destination Mapping\n")
    report.append("| Point | Source (px) | Destination (m) | Expected Landmark |")
    report.append("|-------|-------------|-----------------|-------------------|")
    
    landmarks = [
        "Near left touchline, ~30m from goal",
        "Near right touchline, ~30m from goal",
        "Far right corner (bottom-right of pitch)",
        "Far left corner (bottom-left of pitch)"
    ]
    
    for i, (s, d) in enumerate(zip(src_pts, dst_pts)):
        report.append(f"| P{i+1} | ({s[0]:.1f}, {s[1]:.1f}) | ({d[0]:.1f}, {d[1]:.1f}) | {landmarks[i]} |")
    report.append("")
    
    # Reprojection error
    report.append("## Reprojection Error Analysis\n")
    report.append("| Point | Original (px) | Forward Proj (px) | Fwd Error (px) | Inverse Proj (px) | Inv Error (px) |")
    report.append("|-------|---------------|-------------------|----------------|-------------------|----------------|")
    for i in range(len(src_pts)):
        report.append(f"| P{i+1} | ({src_pts[i,0]:.2f}, {src_pts[i,1]:.2f}) | ({reproj_fwd[i,0]:.2f}, {reproj_fwd[i,1]:.2f}) | {error_fwd[i]:.3f} | ({reproj_inv[i,0]:.2f}, {reproj_inv[i,1]:.2f}) | {error_inv[i]:.3f} |")
    report.append("")
    report.append(f"**RMS Forward Error:** {rms_fwd:.4f} px")
    report.append(f"**RMS Inverse Error:** {rms_inv:.4f} px")
    report.append("")
    
    # Scale variation
    report.append("## Local Scale Variation\n")
    report.append(f"Mean scale: {scale_stats['mean']:.5f} m/px")
    report.append(f"Std scale: {scale_stats['std']:.5f} m/px")
    report.append(f"Min scale: {scale_stats['min']:.5f} m/px")
    report.append(f"Max scale: {scale_stats['max']:.5f} m/px")
    report.append(f"Amplification: {scale_stats['max']/scale_stats['mean']:.2f}x")
    report.append("")
    
    # Determine root cause
    if rms_fwd < 1.0 and scale_stats['max']/scale_stats['mean'] < 1.2:
        diagnosis = "CALIBRATION OK — Scale variation within expected bounds"
    elif rms_fwd < 2.0 and scale_stats['max']/scale_stats['mean'] > 1.5:
        diagnosis = "POOR SOURCE-POINT SELECTION — Non-rectangular source causes perspective stretching"
    elif rms_fwd > 2.0:
        diagnosis = "CALIBRATION ERROR — High reprojection error indicates inaccurate point placement"
    else:
        diagnosis = "EXPECTED PERSPECTIVE GEOMETRY — Some scale variation is normal for oblique views"
    
    report.append(f"## Diagnosis\n")
    report.append(f"**Verdict:** {diagnosis}\n")
    report.append("### Evidence:\n")
    report.append(f"- Reprojection error RMS: {rms_fwd:.3f} px")
    report.append(f"- Scale range: {scale_stats['min']:.5f} to {scale_stats['max']:.5f} m/px")
    report.append(f"- Amplification factor: {scale_stats['max']/scale_stats['mean']:.2f}x")
    report.append(f"- Source point geometry: {'Trapezoid' if not np.allclose(src_pts[:,0], [src_pts[0,0], src_pts[1,0], src_pts[2,0], src_pts[3,0]]) else 'Near-rectangular'}")
    report.append("")
    
    # Save report
    report_path = OUTPUT_DIR / "homography_calibration_audit.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    print(f"[OK] Audit report saved to {report_path}")


def main():
    print("=== Homography Calibration Audit ===\n")
    
    calib = load_calibration()
    src = np.array(calib['calibration_points']['source'], dtype=np.float32)
    dst = np.array(calib['calibration_points']['destination'], dtype=np.float32)
    
    H, _ = cv2.findHomography(src, dst)
    
    # Reprojection error
    error_fwd, error_inv, reproj_fwd, reproj_inv = compute_reprojection_error(src, dst, H)
    rms_fwd = np.sqrt(np.mean(error_fwd ** 2))
    rms_inv = np.sqrt(np.mean(error_inv ** 2))
    
    print(f"RMS Forward Error: {rms_fwd:.4f} px")
    print(f"RMS Inverse Error: {rms_inv:.4f} px")
    
    # Scale grid
    print("\nComputing scale variation grid...")
    xx, yy, scale_grid = compute_scale_grid(H)
    
    valid_scales = scale_grid[~np.isnan(scale_grid)]
    scale_stats = {
        'mean': np.mean(valid_scales),
        'std': np.std(valid_scales),
        'min': np.min(valid_scales),
        'max': np.max(valid_scales)
    }
    
    print(f"Scale variation: {scale_stats['min']:.5f} to {scale_stats['max']:.5f} m/px")
    print(f"Amplification: {scale_stats['max']/scale_stats['mean']:.2f}x")
    
    # Draw overlay
    draw_calibration_visualization(calib)
    
    # Save scale heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(scale_grid, extent=[0, 1050, 680, 0], origin='upper', cmap='hot', interpolation='nearest')
    plt.colorbar(im, label='Metres per pixel')
    ax.set_xlabel('Canvas X (px)')
    ax.set_ylabel('Canvas Y (px)')
    ax.set_title('Homography Scale Variation Heatmap\n(metres per pixel across canvas)')
    
    # Mark calibration points
    dst_scaled = dst * np.array([10, 10])
    ax.scatter(dst_scaled[:, 0], dst_scaled[:, 1], c='cyan', s=100, marker='o', edgecolors='black', linewidths=2, label='Calibration Points')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "homography_scale_heatmap.png", dpi=150)
    plt.close()
    print(f"[OK] Saved scale heatmap to {OUTPUT_DIR / 'homography_scale_heatmap.png'}")
    
    # Generate report
    generate_report(calib, src, dst, error_fwd, error_inv, reproj_fwd, reproj_inv, rms_fwd, rms_inv, scale_stats)
    
    # Save scale data
    scale_df = pd.DataFrame({
        'canvas_x': xx.flatten(),
        'canvas_y': yy.flatten(),
        'm_per_px': scale_grid.flatten()
    })
    scale_df.to_csv(OUTPUT_DIR / "homography_scale_data.csv", index=False)
    print(f"[OK] Scale data saved to {OUTPUT_DIR / 'homography_scale_data.csv'}")
    
    print("\n[COMPLETE] Calibration audit finished.")


if __name__ == "__main__":
    main()

</parameter>
<task_progress>
- [x] Review complete homography pipeline
- [x] Determine why field_x becomes 124-133m
- [x] Determine whether problem is ROI/homography/anchor/etc
- [x] Overlay diagnostic (via speed_debug.csv)
- [x] Compute effective m/px across pitch
- [x] Determine recommended fix with evidence
- [x] Generate final engineering report
- [x] Create validation branch for OOB fix
- [x] Implement OOB filter in validation script
- [x] Run validation and compare before/after
- [x] Validate player motion plausibility
- [x] Generate final validation report
- [x] Final stage-by-stage root cause analysis
- [x] Generate final report
- [ ] Perform complete homography calibration audit
</task_progress>
</write_to_file>