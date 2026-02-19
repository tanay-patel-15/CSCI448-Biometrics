# Hand Landmark Annotation Guide

## Quick Reference

### Annotation Order (12 points total)

```
Point 0-1:   F1 (Thumb)        → START, END
Point 2-3:   F2 (Index)        → START, END  
Point 4-5:   F3 (Middle)       → START, END
Point 6-7:   F4 (Ring)         → START, END
Point 8-9:   F5 (Pinky)        → START, END
Point 10-11: F6 (Palm)         → START, END
```

## Visual Guide

```
        F3 (Middle)
           |
    F2     |     F4
  (Index)  |   (Ring)  F5
      \    |    /    (Pinky)
       \   |   /      /
        \  |  /      /
         \ | /      /
          \|/      /
           *------*
          /        \
         /          \
    F1 (Thumb)    F6 (Palm)
```

## Detailed Instructions

### F1 - Thumb Line (Points 0-1)
**Purpose**: Capture thumb orientation and length

1. **Point 0 (START)**: Click inside thumb near the base (where it connects to palm)
   - Stay away from the thumb boundary
   - Aim for the centerline
   
2. **Point 1 (END)**: Click inside thumb near the tip
   - Stay away from the thumb boundary
   - Should form a line roughly along thumb axis

**What happens**: System samples intensity profile ALONG this line (not perpendicular)

---

### F2 - Index Finger Line (Points 2-3)
**Purpose**: Define index finger axis for width measurements

1. **Point 2 (START)**: Click inside index finger near palm junction
   - Interior point, not on edge
   - Near where finger meets palm
   
2. **Point 3 (END)**: Click inside index finger near tip
   - Interior point, not on edge
   - Should form centerline of finger

**What happens**: System creates 3 perpendicular axes along this line to measure finger width

---

### F3 - Middle Finger Line (Points 4-5)
**Purpose**: Define middle finger axis for width measurements

1. **Point 4 (START)**: Click inside middle finger near palm junction
2. **Point 5 (END)**: Click inside middle finger near tip

**What happens**: System creates 3 perpendicular axes along this line to measure finger width

---

### F4 - Ring Finger Line (Points 6-7)
**Purpose**: Define ring finger axis for width measurements

1. **Point 6 (START)**: Click inside ring finger near palm junction
2. **Point 7 (END)**: Click inside ring finger near tip

**What happens**: System creates 3 perpendicular axes along this line to measure finger width

---

### F5 - Pinky Finger Line (Points 8-9)
**Purpose**: Define pinky finger axis for width measurements

1. **Point 8 (START)**: Click inside pinky finger near palm junction
2. **Point 9 (END)**: Click inside pinky finger near tip

**What happens**: System creates 3 perpendicular axes along this line to measure finger width

---

### F6 - Palm Reference Line (Points 10-11)
**Purpose**: Define palm region for width measurement

1. **Point 10 (START)**: Click inside palm region
   - Good location: below fingers, above wrist
   - Roughly horizontal across palm
   
2. **Point 11 (END)**: Click inside palm region on opposite side
   - Should form a line across palm width

**What happens**: System creates 1 perpendicular axis to measure palm width

---

## Important Rules

### ✅ DO:
- Click INSIDE the hand region (interior points only)
- Follow approximate centerlines for fingers
- Be consistent across all 5 images
- Take your time for accuracy
- Use 'u' to undo mistakes

### ❌ DON'T:
- Click on hand boundaries/edges
- Click outside the hand
- Rush through annotation
- Use different strategies for different images
- Click on background

## Keyboard Controls

| Key | Action |
|-----|--------|
| **Left Click** | Place next landmark point |
| **u** | Undo last point |
| **r** | Restart (clear all points for current image) |
| **Enter/Space** | Confirm and proceed (when 12 points placed) |
| **q** | Quit annotation |

## Tips for Best Results

### 1. Consistent Strategy
Use the same clicking strategy for all 5 images:
- Same finger order
- Same approximate positions
- Same interpretation of "centerline"

### 2. Finger Centerlines
For each finger, imagine a line running through the center from base to tip:
```
    Tip
     |
     |  ← Centerline (imaginary)
     |
    Base
```
Click START and END along this imaginary centerline.

### 3. Palm Line
For the palm reference line (F6), choose a consistent location:
- Option A: Horizontal line across palm below fingers
- Option B: Diagonal line across palm
- **Pick one strategy and use it for all images**

### 4. Dealing with Finger Angles
If fingers are spread or angled:
- Still click interior points
- Line should follow the finger direction
- Perpendicular axes will automatically adjust

### 5. Verification
After placing all 12 points, check:
- All points are visible (green circles)
- Lines connect correct finger pairs
- No points on boundaries
- Lines roughly follow finger directions

If anything looks wrong, press 'r' to restart the current image.

## Example Annotation Session

```
Image 1/5: hand1.jpg

[Click] Point 0 - Thumb START     ✓
[Click] Point 1 - Thumb END       ✓
[Click] Point 2 - Index START     ✓
[Click] Point 3 - Index END       ✓
[Oops, misclicked!]
[Press 'u'] Undo Point 3          ✓
[Click] Point 3 - Index END       ✓
[Click] Point 4 - Middle START    ✓
[Click] Point 5 - Middle END      ✓
[Click] Point 6 - Ring START      ✓
[Click] Point 7 - Ring END        ✓
[Click] Point 8 - Pinky START     ✓
[Click] Point 9 - Pinky END       ✓
[Click] Point 10 - Palm START     ✓
[Click] Point 11 - Palm END       ✓

All points placed! Press ENTER to confirm
[Press Enter]

✓ Completed annotation for hand1.jpg
```

## Troubleshooting

### "Can't see the annotation window"
- Check if window is minimized or behind other windows
- Try Alt+Tab (Windows) or Cmd+Tab (Mac) to find it
- Window might be off-screen; try moving it

### "Clicked wrong point"
- Press 'u' to undo last point
- Can undo multiple times
- Or press 'r' to restart entire image

### "Lines don't look right"
- Press 'r' to restart current image
- Ensure you're clicking in correct order
- Remember: 2 points per finger line

### "Program not responding to keys"
- Make sure annotation window is focused (click on it)
- Try clicking on window title bar first
- Check keyboard language settings

## After Annotation

Once all 5 images are annotated:
1. System computes feature vectors automatically
2. Distance matrix is calculated
3. Visualizations are generated
4. All outputs saved to `outputs/` folder

Check `outputs/annotated_demo.png` to verify:
- All landmarks are correctly placed
- Measurement axes look reasonable
- Thickness values are displayed

---

**Ready to annotate? Run:**
```bash
python hand_biometrics.py
```

**Good luck! 🖐️**
