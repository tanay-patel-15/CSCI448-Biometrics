#!/usr/bin/env python3
"""
========================================================
HAND BIOMETRICS ASSIGNMENT - COMPLETE SOLUTION
========================================================

INSTALLATION:
    pip install numpy opencv-python matplotlib

USAGE:
    1. Place 5 hand images in ./hand_images/ folder
    2. Run: python hand_biometrics.py
    3. For each image, click landmarks as prompted
    4. Results saved to ./outputs/

CONTROLS DURING ANNOTATION:
    - Left click: Place landmark point
    - 'u': Undo last point
    - 'r': Restart current image
    - Enter/Space: Confirm and proceed to next image
    - 'q': Quit

LANDMARK SCHEMA (12 points total per image):
    F1 (Thumb):  2 points - start, end along thumb centerline
    F2 (Index):  2 points - start, end along index finger centerline
    F3 (Middle): 2 points - start, end along middle finger centerline
    F4 (Ring):   2 points - start, end along ring finger centerline
    F5 (Pinky):  2 points - start, end along pinky finger centerline
    F6 (Palm):   2 points - start, end for palm reference line

FEATURE VECTOR (16 dimensions):
    - F2-F5: 3 width measurements each = 12 values
    - F6: 1 width measurement = 1 value
    - F1: 3 profile statistics (mean, std, length) = 3 values
    Total: 16 features per image

========================================================
"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv


# ========================================================
# CONFIGURATION
# ========================================================

INPUT_FOLDER = "./hand_images"
OUTPUT_FOLDER = "./outputs"
NUM_IMAGES = 5
NUM_POINTS_PER_IMAGE = 12  # 6 lines × 2 points each
SAMPLE_POINTS = 200  # Points to sample along each axis
WIDTH_AXIS_HALF_LENGTH = 80  # Pixels for perpendicular width axes
NUM_WIDTH_POSITIONS = 3  # Positions along each finger for width measurement

# Finger line definitions
FINGER_LINES = {
    'F1': {'name': 'Thumb', 'indices': (0, 1), 'type': 'along'},
    'F2': {'name': 'Index', 'indices': (2, 3), 'type': 'width'},
    'F3': {'name': 'Middle', 'indices': (4, 5), 'type': 'width'},
    'F4': {'name': 'Ring', 'indices': (6, 7), 'type': 'width'},
    'F5': {'name': 'Pinky', 'indices': (8, 9), 'type': 'width'},
    'F6': {'name': 'Palm', 'indices': (10, 11), 'type': 'width'},
}


# ========================================================
# UTILITY FUNCTIONS
# ========================================================

def ensure_output_folder():
    """Create output folder if it doesn't exist."""
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


def load_images(folder: str, num_images: int = 5) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load images from folder, sort by filename, and return first num_images.
    
    Returns:
        images: List of RGB images (numpy arrays)
        filenames: List of corresponding filenames
    """
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    
    # Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(folder_path.glob(ext))
    
    if len(image_files) == 0:
        raise FileNotFoundError(f"No images found in {folder}")
    
    # Sort by filename
    image_files = sorted(image_files)[:num_images]
    
    if len(image_files) < num_images:
        print(f"Warning: Only found {len(image_files)} images, expected {num_images}")
    
    # Load images
    images = []
    filenames = []
    for img_path in image_files:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: Could not load {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        images.append(img_rgb)
        filenames.append(img_path.name)
    
    print(f"Loaded {len(images)} images from {folder}")
    return images, filenames


# ========================================================
# INTERACTIVE ANNOTATION
# ========================================================

class LandmarkAnnotator:
    """Interactive point-click interface for landmark annotation."""
    
    def __init__(self, image: np.ndarray, image_name: str, num_points: int = 12):
        self.image = image.copy()
        self.image_name = image_name
        self.num_points = num_points
        self.points = []
        self.display_image = None
        self.window_name = "Landmark Annotation"
        self.finished = False
        self.quit = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < self.num_points:
            self.points.append((x, y))
            self.update_display()
    
    def update_display(self):
        """Update display with current points and labels."""
        self.display_image = self.image.copy()
        
        # Draw points and labels
        for i, (x, y) in enumerate(self.points):
            cv2.circle(self.display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.display_image, f"P{i}", (x + 10, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw lines for completed finger pairs
        for finger_id, finger_info in FINGER_LINES.items():
            idx1, idx2 = finger_info['indices']
            if len(self.points) > idx2:
                p1 = self.points[idx1]
                p2 = self.points[idx2]
                color = (255, 0, 0) if finger_info['type'] == 'along' else (0, 255, 255)
                cv2.line(self.display_image, p1, p2, color, 2)
        
        # Add instruction text
        next_point = len(self.points)
        if next_point < self.num_points:
            # Determine which finger line we're working on
            finger_name = ""
            point_type = ""
            for finger_id, finger_info in FINGER_LINES.items():
                idx1, idx2 = finger_info['indices']
                if next_point == idx1:
                    finger_name = finger_info['name']
                    point_type = "START"
                    break
                elif next_point == idx2:
                    finger_name = finger_info['name']
                    point_type = "END"
                    break
            
            instruction = f"Click {finger_name} {point_type} (point {next_point + 1}/{self.num_points})"
        else:
            instruction = "All points placed! Press ENTER to confirm"
        
        cv2.putText(self.display_image, instruction, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.display_image, "u=undo | r=restart | Enter=confirm | q=quit",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow(self.window_name, cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR))
    
    def annotate(self) -> Optional[List[Tuple[int, int]]]:
        """
        Run interactive annotation session.
        
        Returns:
            List of (x, y) points, or None if user quit
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        self.update_display()
        
        while not self.finished and not self.quit:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('u'):  # Undo
                if len(self.points) > 0:
                    self.points.pop()
                    self.update_display()
            
            elif key == ord('r'):  # Restart
                self.points = []
                self.update_display()
            
            elif key in [13, 32]:  # Enter or Space
                if len(self.points) == self.num_points:
                    self.finished = True
                else:
                    print(f"Need {self.num_points} points, only have {len(self.points)}")
            
            elif key == ord('q'):  # Quit
                self.quit = True
        
        cv2.destroyWindow(self.window_name)
        
        if self.quit:
            return None
        return self.points


def annotate_all_images(images: List[np.ndarray], filenames: List[str]) -> Optional[Dict[str, List[Tuple[int, int]]]]:
    """
    Annotate all images interactively.
    
    Returns:
        Dictionary mapping filename to list of landmark points, or None if user quit
    """
    landmarks = {}
    
    print("\n" + "="*60)
    print("STARTING INTERACTIVE ANNOTATION")
    print("="*60)
    print("\nPlease annotate each image by clicking landmarks in order:")
    print("  F1 (Thumb):  2 points")
    print("  F2 (Index):  2 points")
    print("  F3 (Middle): 2 points")
    print("  F4 (Ring):   2 points")
    print("  F5 (Pinky):  2 points")
    print("  F6 (Palm):   2 points")
    print("\nClick points INSIDE the hand region (not on boundaries).")
    print("="*60 + "\n")
    
    for i, (image, filename) in enumerate(zip(images, filenames)):
        print(f"\nAnnotating image {i+1}/{len(images)}: {filename}")
        
        annotator = LandmarkAnnotator(image, filename, NUM_POINTS_PER_IMAGE)
        points = annotator.annotate()
        
        if points is None:
            print("User quit annotation.")
            return None
        
        landmarks[filename] = points
        print(f"✓ Completed annotation for {filename}")
    
    print("\n" + "="*60)
    print("ANNOTATION COMPLETE!")
    print("="*60 + "\n")
    
    return landmarks


# ========================================================
# AXIS GENERATION
# ========================================================

def compute_perpendicular_axes(p1: Tuple[int, int], p2: Tuple[int, int],
                               num_positions: int = 3,
                               half_length: int = 80) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Compute perpendicular width measurement axes along a finger line.
    
    Args:
        p1, p2: Start and end points of finger line
        num_positions: Number of positions along the line
        half_length: Half-length of perpendicular axis in pixels
    
    Returns:
        List of axis segments, each as ((x1, y1), (x2, y2))
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Direction vector along finger
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    
    if length < 1e-6:
        return []
    
    # Normalized direction
    dx_norm = dx / length
    dy_norm = dy / length
    
    # Perpendicular direction (rotate 90 degrees)
    perp_dx = -dy_norm
    perp_dy = dx_norm
    
    axes = []
    for i in range(num_positions):
        # Position along finger line (evenly spaced)
        t = (i + 1) / (num_positions + 1)
        cx = x1 + t * dx
        cy = y1 + t * dy
        
        # Perpendicular axis endpoints
        ax1 = cx - half_length * perp_dx
        ay1 = cy - half_length * perp_dy
        ax2 = cx + half_length * perp_dx
        ay2 = cy + half_length * perp_dy
        
        axes.append(((ax1, ay1), (ax2, ay2)))
    
    return axes


def compute_all_measurement_axes(landmarks: List[Tuple[int, int]]) -> Dict[str, List]:
    """
    Compute all measurement axes for a single image.
    
    Returns:
        Dictionary mapping finger ID to list of axes
        For 'along' type (F1): single axis along the line
        For 'width' type (F2-F6): list of perpendicular axes
    """
    axes_dict = {}
    
    for finger_id, finger_info in FINGER_LINES.items():
        idx1, idx2 = finger_info['indices']
        p1 = landmarks[idx1]
        p2 = landmarks[idx2]
        
        if finger_info['type'] == 'along':
            # For thumb: axis is the line itself
            axes_dict[finger_id] = [(p1, p2)]
        else:
            # For other fingers: perpendicular width axes
            num_pos = 1 if finger_id == 'F6' else NUM_WIDTH_POSITIONS
            axes = compute_perpendicular_axes(p1, p2, num_pos, WIDTH_AXIS_HALF_LENGTH)
            axes_dict[finger_id] = axes
    
    return axes_dict


# ========================================================
# INTENSITY PROFILE SAMPLING
# ========================================================

def sample_intensity_profile(image_gray: np.ndarray,
                            p1: Tuple[float, float],
                            p2: Tuple[float, float],
                            num_samples: int = 200) -> np.ndarray:
    """
    Sample intensity values along a line segment using bilinear interpolation.
    
    Args:
        image_gray: Grayscale image
        p1, p2: Start and end points of line segment
        num_samples: Number of points to sample
    
    Returns:
        Array of intensity values (length num_samples)
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Generate sampling positions
    t = np.linspace(0, 1, num_samples)
    xs = x1 + t * (x2 - x1)
    ys = y1 + t * (y2 - y1)
    
    # Clip to image bounds
    h, w = image_gray.shape
    xs = np.clip(xs, 0, w - 1)
    ys = np.clip(ys, 0, h - 1)
    
    # Bilinear interpolation
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1_int = np.clip(x0 + 1, 0, w - 1)
    y1_int = np.clip(y0 + 1, 0, h - 1)
    
    wx = xs - x0
    wy = ys - y0
    
    # Bilinear weights
    w00 = (1 - wx) * (1 - wy)
    w01 = (1 - wx) * wy
    w10 = wx * (1 - wy)
    w11 = wx * wy
    
    # Sample and interpolate
    profile = (w00 * image_gray[y0, x0] +
               w01 * image_gray[y1_int, x0] +
               w10 * image_gray[y0, x1_int] +
               w11 * image_gray[y1_int, x1_int])
    
    return profile


# ========================================================
# THICKNESS MEASUREMENT
# ========================================================

def smooth_profile(profile: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(profile, kernel, mode='same')


def measure_thickness_from_profile(profile: np.ndarray) -> float:
    """
    Measure thickness from intensity profile using edge detection method.
    
    Strategy:
    1. Smooth profile
    2. Compute gradient magnitude
    3. Find two strongest edge peaks around center
    4. Return distance between edges
    
    Returns:
        Thickness in pixels, or NaN if edges not found
    """
    if len(profile) < 10:
        return np.nan
    
    # Smooth profile
    smoothed = smooth_profile(profile, window_size=7)
    
    # Compute gradient magnitude
    gradient = np.abs(np.gradient(smoothed))
    
    # Find center region
    center = len(profile) // 2
    quarter = len(profile) // 4
    
    # Split into left and right halves
    left_half = gradient[:center]
    right_half = gradient[center:]
    
    # Find strongest edge in each half
    if len(left_half) > 0 and len(right_half) > 0:
        left_edge_idx = np.argmax(left_half)
        right_edge_idx = center + np.argmax(right_half)
        
        # Check if edges are reasonable (not too close to ends)
        margin = len(profile) // 10
        if left_edge_idx > margin and right_edge_idx < len(profile) - margin:
            thickness = right_edge_idx - left_edge_idx
            
            # Sanity check: thickness should be reasonable
            if thickness > 5 and thickness < len(profile) * 0.9:
                return float(thickness)
    
    # Fallback: threshold-based method
    # Estimate background from ends, foreground from center
    bg_intensity = (np.mean(profile[:10]) + np.mean(profile[-10:])) / 2
    fg_intensity = np.mean(profile[center-10:center+10])
    
    if abs(fg_intensity - bg_intensity) < 5:
        return np.nan  # No clear contrast
    
    threshold = (bg_intensity + fg_intensity) / 2
    
    # Find crossings
    above_threshold = profile > threshold
    if not np.any(above_threshold):
        return np.nan
    
    # Find first and last crossing
    crossing_indices = np.where(above_threshold)[0]
    if len(crossing_indices) < 2:
        return np.nan
    
    thickness = crossing_indices[-1] - crossing_indices[0]
    
    if thickness > 5 and thickness < len(profile) * 0.9:
        return float(thickness)
    
    return np.nan


# ========================================================
# FEATURE VECTOR COMPUTATION
# ========================================================

def compute_feature_vector(image: np.ndarray,
                          landmarks: List[Tuple[int, int]],
                          axes_dict: Dict[str, List]) -> np.ndarray:
    """
    Compute feature vector for one image.
    
    Feature vector structure (16 dimensions):
    - F2 widths: 3 values (positions 0-2)
    - F3 widths: 3 values (positions 3-5)
    - F4 widths: 3 values (positions 6-8)
    - F5 widths: 3 values (positions 9-11)
    - F6 width: 1 value (position 12)
    - F1 profile stats: 3 values (mean, std, length) (positions 13-15)
    
    Returns:
        Feature vector of length 16
    """
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    features = []
    
    # Process width measurements for F2-F5 (3 measurements each)
    for finger_id in ['F2', 'F3', 'F4', 'F5']:
        axes = axes_dict[finger_id]
        widths = []
        
        for axis in axes:
            p1, p2 = axis
            profile = sample_intensity_profile(image_gray, p1, p2, SAMPLE_POINTS)
            thickness = measure_thickness_from_profile(profile)
            widths.append(thickness)
        
        features.extend(widths)
    
    # Process F6 (palm) - 1 measurement
    axes = axes_dict['F6']
    if len(axes) > 0:
        p1, p2 = axes[0]
        profile = sample_intensity_profile(image_gray, p1, p2, SAMPLE_POINTS)
        thickness = measure_thickness_from_profile(profile)
        features.append(thickness)
    else:
        features.append(np.nan)
    
    # Process F1 (thumb) - along-line profile statistics
    axes = axes_dict['F1']
    p1, p2 = axes[0]
    profile = sample_intensity_profile(image_gray, p1, p2, SAMPLE_POINTS)
    
    # Compute statistics
    profile_mean = np.mean(profile)
    profile_std = np.std(profile)
    profile_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    features.extend([profile_mean, profile_std, profile_length])
    
    # Convert to numpy array
    feature_vector = np.array(features, dtype=np.float32)
    
    # Handle NaN values by imputing with median of valid values
    valid_mask = ~np.isnan(feature_vector)
    if np.any(~valid_mask):
        if np.any(valid_mask):
            median_val = np.median(feature_vector[valid_mask])
            feature_vector[~valid_mask] = median_val
        else:
            feature_vector[~valid_mask] = 0.0
    
    return feature_vector


def compute_all_feature_vectors(images: List[np.ndarray],
                               landmarks_dict: Dict[str, List[Tuple[int, int]]],
                               filenames: List[str]) -> Dict[str, np.ndarray]:
    """
    Compute feature vectors for all images.
    
    Returns:
        Dictionary mapping filename to feature vector
    """
    feature_vectors = {}
    
    print("\n" + "="*60)
    print("COMPUTING FEATURE VECTORS")
    print("="*60 + "\n")
    
    for image, filename in zip(images, filenames):
        print(f"Processing {filename}...")
        
        landmarks = landmarks_dict[filename]
        axes_dict = compute_all_measurement_axes(landmarks)
        feature_vector = compute_feature_vector(image, landmarks, axes_dict)
        
        feature_vectors[filename] = feature_vector
        print(f"  Feature vector shape: {feature_vector.shape}")
        print(f"  Feature vector: {feature_vector}")
    
    print("\n" + "="*60)
    print("FEATURE VECTOR COMPUTATION COMPLETE")
    print("="*60 + "\n")
    
    return feature_vectors


# ========================================================
# DISTANCE MATRIX COMPUTATION
# ========================================================

def compute_distance_matrix(feature_vectors: Dict[str, np.ndarray],
                           filenames: List[str]) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix.
    
    Returns:
        5x5 distance matrix
    """
    n = len(filenames)
    distance_matrix = np.zeros((n, n), dtype=np.float32)
    
    for i in range(n):
        for j in range(i+1, n):
            fv1 = feature_vectors[filenames[i]]
            fv2 = feature_vectors[filenames[j]]
            dist = np.linalg.norm(fv1 - fv2)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist
    
    return distance_matrix


# ========================================================
# VISUALIZATION
# ========================================================

def visualize_demo(image: np.ndarray,
                  landmarks: List[Tuple[int, int]],
                  axes_dict: Dict[str, List],
                  feature_vector: np.ndarray,
                  filename: str,
                  save_path: str):
    """
    Create comprehensive visualization showing:
    - Landmark points
    - Finger lines F1-F6
    - Measurement axes
    - Thickness values
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 16))
    ax.imshow(image)
    ax.axis('off')
    ax.set_title(f"Hand Biometrics Analysis: {filename}", fontsize=14, fontweight='bold')
    
    # Draw landmark points
    for i, (x, y) in enumerate(landmarks):
        ax.plot(x, y, 'go', markersize=8, markeredgecolor='white', markeredgewidth=1.5)
        ax.text(x + 10, y - 10, f'P{i}', color='yellow', fontsize=9,
               fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    
    # Draw finger lines F1-F6
    colors = {'F1': 'red', 'F2': 'cyan', 'F3': 'magenta', 'F4': 'yellow', 'F5': 'lime', 'F6': 'orange'}
    
    for finger_id, finger_info in FINGER_LINES.items():
        idx1, idx2 = finger_info['indices']
        p1 = landmarks[idx1]
        p2 = landmarks[idx2]
        color = colors[finger_id]
        
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=3,
               label=f"{finger_id} ({finger_info['name']})")
    
    # Draw measurement axes and thickness values
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    feature_idx = 0
    
    # F2-F5 width axes
    for finger_id in ['F2', 'F3', 'F4', 'F5']:
        axes = axes_dict[finger_id]
        for axis in axes:
            p1, p2 = axis
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=1.5, alpha=0.7)
            
            # Get thickness value
            thickness = feature_vector[feature_idx]
            feature_idx += 1
            
            # Display thickness near midpoint
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            ax.text(mid_x, mid_y, f'{thickness:.1f}', color='white', fontsize=8,
                   fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.8))
    
    # F6 palm axis
    axes = axes_dict['F6']
    if len(axes) > 0:
        p1, p2 = axes[0]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=1.5, alpha=0.7)
        
        thickness = feature_vector[feature_idx]
        feature_idx += 1
        
        mid_x = (p1[0] + p2[0]) / 2
        mid_y = (p1[1] + p2[1]) / 2
        ax.text(mid_x, mid_y, f'{thickness:.1f}', color='white', fontsize=8,
               fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='blue', alpha=0.8))
    
    # F1 thumb axis (along line)
    axes = axes_dict['F1']
    p1, p2 = axes[0]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2, alpha=0.9)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {save_path}")


# ========================================================
# OUTPUT SAVING
# ========================================================

def save_feature_vectors(feature_vectors: Dict[str, np.ndarray],
                        filenames: List[str],
                        output_path: str):
    """Save feature vectors to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['filename'] + [f'f{i}' for i in range(16)]
        writer.writerow(header)
        
        # Data
        for filename in filenames:
            fv = feature_vectors[filename]
            row = [filename] + fv.tolist()
            writer.writerow(row)
    
    print(f"Saved feature vectors to {output_path}")


def save_distance_matrix(distance_matrix: np.ndarray,
                        filenames: List[str],
                        output_path: str):
    """Save distance matrix to CSV file."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = [''] + filenames
        writer.writerow(header)
        
        # Data
        for i, filename in enumerate(filenames):
            row = [filename] + distance_matrix[i].tolist()
            writer.writerow(row)
    
    print(f"Saved distance matrix to {output_path}")


def print_distance_matrix(distance_matrix: np.ndarray, filenames: List[str]):
    """Print distance matrix in a nice format."""
    print("\n" + "="*60)
    print("PAIRWISE DISTANCE MATRIX (Euclidean)")
    print("="*60 + "\n")
    
    # Print header
    print(f"{'':20s}", end='')
    for fname in filenames:
        print(f"{fname[:15]:>15s}", end='')
    print()
    
    # Print rows
    for i, fname in enumerate(filenames):
        print(f"{fname[:20]:20s}", end='')
        for j in range(len(filenames)):
            if i == j:
                print(f"{'0.00':>15s}", end='')
            elif i < j:
                print(f"{distance_matrix[i, j]:>15.2f}", end='')
            else:
                print(f"{'':>15s}", end='')
        print()
    
    print("\n" + "="*60 + "\n")


# ========================================================
# MAIN PIPELINE
# ========================================================

def main():
    """Main execution pipeline."""
    print("\n" + "="*60)
    print("HAND BIOMETRICS ASSIGNMENT - COMPLETE SOLUTION")
    print("="*60)
    print(__doc__)
    
    # Ensure output folder exists
    ensure_output_folder()
    
    # Step 1: Load images
    try:
        images, filenames = load_images(INPUT_FOLDER, NUM_IMAGES)
    except Exception as e:
        print(f"Error loading images: {e}")
        print(f"\nPlease ensure {INPUT_FOLDER} folder exists and contains at least {NUM_IMAGES} images.")
        return
    
    if len(images) == 0:
        print("No images loaded. Exiting.")
        return
    
    # Step 2: Interactive annotation
    landmarks_dict = annotate_all_images(images, filenames)
    
    if landmarks_dict is None:
        print("Annotation cancelled by user.")
        return
    
    # Step 3: Compute feature vectors
    feature_vectors = compute_all_feature_vectors(images, landmarks_dict, filenames)
    
    # Step 4: Compute distance matrix
    print("\n" + "="*60)
    print("COMPUTING DISTANCE MATRIX")
    print("="*60 + "\n")
    
    distance_matrix = compute_distance_matrix(feature_vectors, filenames)
    print_distance_matrix(distance_matrix, filenames)
    
    # Step 5: Save outputs
    print("\n" + "="*60)
    print("SAVING OUTPUTS")
    print("="*60 + "\n")
    
    save_feature_vectors(feature_vectors, filenames,
                        os.path.join(OUTPUT_FOLDER, 'feature_vectors.csv'))
    
    save_distance_matrix(distance_matrix, filenames,
                        os.path.join(OUTPUT_FOLDER, 'distance_matrix.csv'))
    
    # Step 6: Create demo visualization for first image
    print("\nCreating demo visualization...")
    
    demo_filename = filenames[0]
    demo_image = images[0]
    demo_landmarks = landmarks_dict[demo_filename]
    demo_axes = compute_all_measurement_axes(demo_landmarks)
    demo_fv = feature_vectors[demo_filename]
    
    visualize_demo(demo_image, demo_landmarks, demo_axes, demo_fv,
                  demo_filename, os.path.join(OUTPUT_FOLDER, 'annotated_demo.png'))
    
    # Optional: Create visualizations for all images
    for i, (image, filename) in enumerate(zip(images, filenames)):
        landmarks = landmarks_dict[filename]
        axes = compute_all_measurement_axes(landmarks)
        fv = feature_vectors[filename]
        
        save_path = os.path.join(OUTPUT_FOLDER, f'annotated_{i+1}_{filename}')
        visualize_demo(image, landmarks, axes, fv, filename, save_path)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {OUTPUT_FOLDER}/")
    print("  - feature_vectors.csv")
    print("  - distance_matrix.csv")
    print("  - annotated_demo.png")
    print("  - annotated_*.png (for each image)")
    print("\nYou can now screen-record the annotated images for your demo.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
