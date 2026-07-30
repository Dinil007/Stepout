import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from app.homography.calibrator import LandmarkHomographyCalibrator, LandmarkClickSelector, PITCH_LANDMARKS

def main():
    parser = argparse.ArgumentParser(description="Calibrate homography using landmark-based calibration")
    parser.add_argument("video_path", help="Path to input video")
    parser.add_argument("--output", default="configs/homography_calibration.json", help="Output calibration file")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to use for calibration")
    
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Failed to read frame {args.frame}")
        return
    
    calibrator = LandmarkHomographyCalibrator()
    selector = LandmarkClickSelector()
    
    print("Select 6-20 landmarks by clicking on them in the image.")
    print("Press 's' when done (minimum 6), 'c' to clear, 'q' to quit")
    
    image_points, world_points = selector.select_landmarks(frame, PITCH_LANDMARKS)
    
    if len(image_points) < calibrator.MIN_LANDMARKS:
        print("Not enough landmarks selected")
        return
    
    result = calibrator.calibrate(image_points, world_points, frame.shape[:2])
    
    if result.success:
        # Show pitch overlay preview
        overlay = calibrator.generate_pitch_overlay(frame.shape[:2])
        
        print(f"\n=== Calibration Results ===")
        print(f"Reprojection error: {result.reprojection_error:.2f}px")
        print(f"Determinant: {result.determinant:.3f}")
        print(f"Metres per pixel: X={result.metres_per_pixel_x:.4f}, Y={result.metres_per_pixel_y:.4f}")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Landmarks used: {result.num_landmarks}")
        print(f"Validation: {result.message}")
        print(f"\nPress any key to preview overlay, or Ctrl+C to cancel...")
        
        cv2.imshow("Pitch Overlay Preview", overlay)
        cv2.waitKey(0)
        
        calibrator.save_calibration(result, args.output)
        print(f"Calibration saved to {args.output}")
        
        # Optionally warp frame to pitch view
        warped = cv2.warpPerspective(frame, result.homography_matrix, (800, 600))
        cv2.imshow("Warped", warped)
        cv2.waitKey(0)
    else:
        print(f"\n=== Calibration Failed ===")
        print(f"Reason: {result.message}")
        print("Please try calibration again with different landmarks")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()