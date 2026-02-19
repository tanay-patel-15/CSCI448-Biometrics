# Hand Biometrics Assignment - Complete Solution

## Overview

This solution implements a distance-based hand biometric feature extraction system that:
1. Loads 5 hand images
2. Allows interactive landmark annotation (clicking points)
3. Automatically computes measurement axes from landmarks
4. Extracts thickness measurements from intensity profiles
5. Generates 16-dimensional feature vectors
6. Computes pairwise Euclidean distance matrix
7. Creates annotated visualizations for demo

## Installation

### Requirements
- Python 3.7 or higher
- pip package manager

### Install Dependencies

```bash
pip install numpy opencv-python matplotlib
```

Or using a virtual environment (recommended):

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install numpy opencv-python matplotlib
```

## Setup

### 1. Prepare Your Hand Images

Create a folder called `hand_images` in the same directory as the script:

```bash
mkdir hand_images
```

Place exactly 5 hand images in this folder. Supported formats:
- `.jpg`, `.jpeg`, `.png`

Images should be:
- Taken under consistent lighting conditions
- Same hand in similar pose
- Clear, high-resolution
- Named so they sort in desired order (e.g., `hand1.jpg`, `hand2.jpg`, etc.)

### 2. Directory Structure

```
Assignment 2/
├── hand_biometrics.py          # Main script
├── README_HAND_BIOMETRICS.md   # This file
├── hand_images/                # Your input images
│   ├── hand1.jpg
│   ├── hand2.jpg
│   ├── hand3.jpg
│   ├── hand4.jpg
│   └── hand5.jpg
└── outputs/                    # Generated automatically
    ├── feature_vectors.csv
    ├── distance_matrix.csv
    ├── annotated_demo.png
    └── annotated_*.png
```

## Usage

### Running the Script

```bash
python hand_biometrics.py
```

### Interactive Annotation Process

For each image, you will annotate 12 landmark points (6 lines × 2 points each):

#### Landmark Schema

**F1 - Thumb Line** (2 points)
- Click START point inside thumb (near base)
- Click END point inside thumb (near tip)
- These points should be along the thumb centerline, NOT on the boundary

**F2 - Index Finger Line** (2 points)
- Click START point inside index finger (near palm)
- Click END point inside index finger (near tip)

**F3 - Middle Finger Line** (2 points)
- Click START point inside middle finger (near palm)
- Click END point inside middle finger (near tip)

**F4 - Ring Finger Line** (2 points)
- Click START point inside ring finger (near palm)
- Click END point inside ring finger (near tip)

**F5 - Pinky Finger Line** (2 points)
- Click START point inside pinky finger (near palm)
- Click END point inside pinky finger (near tip)

**F6 - Palm Reference Line** (2 points)
- Click START point inside palm region
- Click END point inside palm region
- This defines a reference direction for palm measurements

### Keyboard Controls During Annotation

- **Left Click**: Place landmark point
- **'u' key**: Undo last point
- **'r' key**: Restart current image (clear all points)
- **Enter or Space**: Confirm and proceed to next image (only when all 12 points placed)
- **'q' key**: Quit annotation

### Tips for Good Annotation

1. **Click INSIDE the hand region**: Never click on the boundary/outline
2. **Follow the centerline**: For fingers, imagine a line through the center
3. **Be consistent**: Use the same strategy for all 5 images
4. **Take your time**: Accurate landmarks = better features
5. **Use undo**: If you misclick, press 'u' to undo

## How It Works

### 1. Automatic Axis Generation

After you click landmarks, the system automatically generates measurement axes:

- **For F2-F5 (fingers)**: 3 perpendicular width axes per finger
  - Axes are perpendicular to the finger line
  - Evenly spaced along the finger
  - Used to measure finger width at different positions

- **For F6 (palm)**: 1 perpendicular width axis
  - Perpendicular to the palm reference line
  - Used to measure palm width

- **For F1 (thumb)**: Along-line sampling
  - Samples intensity directly along the thumb line
  - Extracts profile statistics instead of width

### 2. Thickness Measurement

For each perpendicular axis:
1. Sample 200 intensity values along the axis
2. Smooth the intensity profile
3. Detect edges using gradient analysis
4. Measure distance between left and right edges
5. This distance = thickness/width in pixels

### 3. Feature Vector (16 dimensions)

| Features | Dimensions | Description |
|----------|-----------|-------------|
| F2 widths | 3 | Index finger widths at 3 positions |
| F3 widths | 3 | Middle finger widths at 3 positions |
| F4 widths | 3 | Ring finger widths at 3 positions |
| F5 widths | 3 | Pinky finger widths at 3 positions |
| F6 width | 1 | Palm width |
| F1 stats | 3 | Thumb profile (mean, std, length) |
| **Total** | **16** | Complete feature vector |

### 4. Distance Matrix

Computes pairwise Euclidean distances between all 5 feature vectors:

```
Distance(i, j) = ||FV_i - FV_j||₂ = sqrt(Σ(FV_i[k] - FV_j[k])²)
```

Results in a 5×5 symmetric matrix with zeros on diagonal.

## Output Files

### 1. `feature_vectors.csv`

Contains the 16-dimensional feature vector for each image:

```csv
filename,f0,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15
hand1.jpg,45.2,48.1,42.3,...
hand2.jpg,44.8,47.5,41.9,...
...
```

### 2. `distance_matrix.csv`

Contains the 5×5 pairwise distance matrix:

```csv
,hand1.jpg,hand2.jpg,hand3.jpg,hand4.jpg,hand5.jpg
hand1.jpg,0.00,12.45,23.67,18.92,15.34
hand2.jpg,12.45,0.00,19.23,14.78,11.56
...
```

### 3. `annotated_demo.png`

Comprehensive visualization showing:
- Original hand image
- All 12 landmark points (green circles with labels)
- Finger lines F1-F6 (colored lines)
- Perpendicular measurement axes (blue lines)
- Thickness values displayed near each axis (white text on blue background)
- Legend identifying each finger line

### 4. `annotated_*.png`

Individual annotated images for each of the 5 input images.

## Demo for Screen Recording

To demonstrate the system for your assignment:

1. **Run the script**: Shows interactive annotation process
2. **Annotate images**: Click landmarks as prompted
3. **Review outputs**: 
   - Terminal shows feature vectors and distance matrix
   - Open `outputs/annotated_demo.png` to show:
     - Manually clicked landmarks
     - Finger lines F1-F6
     - Automatically generated measurement axes
     - Computed thickness values

4. **Screen record**: Capture the annotated images showing all required elements

## Troubleshooting

### "No images found in ./hand_images"
- Ensure the `hand_images` folder exists
- Check that images have supported extensions (.jpg, .jpeg, .png)
- Verify file permissions

### "Could not load image"
- Check image file is not corrupted
- Ensure image format is supported
- Try converting image to standard RGB format

### Window doesn't appear during annotation
- Check OpenCV installation: `python -c "import cv2; print(cv2.__version__)"`
- On macOS, may need to run from terminal (not IDE)
- Try resizing window manually if it's off-screen

### Feature values are NaN
- Ensure landmarks are placed correctly inside hand region
- Check that image has good contrast between hand and background
- Try adjusting `WIDTH_AXIS_HALF_LENGTH` parameter if axes extend outside hand

### Distance matrix has unexpected values
- Verify all 5 images are of the SAME hand
- Check that annotation strategy is consistent across images
- Ensure images have similar lighting and pose

## Customization

You can adjust parameters at the top of `hand_biometrics.py`:

```python
INPUT_FOLDER = "./hand_images"          # Input folder path
OUTPUT_FOLDER = "./outputs"             # Output folder path
NUM_IMAGES = 5                          # Number of images to process
SAMPLE_POINTS = 200                     # Sampling resolution
WIDTH_AXIS_HALF_LENGTH = 80             # Length of width axes (pixels)
NUM_WIDTH_POSITIONS = 3                 # Width measurements per finger
```

## Technical Details

### Edge Detection Method

The thickness measurement uses a two-stage approach:

1. **Primary method**: Gradient-based edge detection
   - Smooth intensity profile with moving average
   - Compute gradient magnitude
   - Find strongest edges in left and right halves
   - Measure distance between edges

2. **Fallback method**: Threshold-based
   - Estimate background intensity from profile ends
   - Estimate foreground intensity from profile center
   - Compute threshold as midpoint
   - Find threshold crossings
   - Measure distance between crossings

### Robustness Features

- **NaN handling**: Invalid measurements replaced with median of valid values
- **Bilinear interpolation**: Smooth sampling along axes
- **Sanity checks**: Validates edge positions and thickness ranges
- **Consistent ordering**: Images sorted by filename for reproducibility

## Assignment Compliance

This solution strictly follows all assignment requirements:

✅ **No manual boundary points**: All landmarks are interior points only  
✅ **Automatic axis generation**: Perpendicular axes computed automatically  
✅ **Interactive annotation**: Full click-based UI with undo/restart  
✅ **Proper visualization**: Shows landmarks, lines, axes, and measurements  
✅ **Distance-based features**: Uses intensity profiles for thickness  
✅ **Euclidean distance**: Compares feature vectors correctly  
✅ **Complete outputs**: CSV files and annotated images for submission  

## Contact

For questions or issues, please refer to the assignment instructions or contact your TA.

---

**Good luck with your assignment! 🎯**
