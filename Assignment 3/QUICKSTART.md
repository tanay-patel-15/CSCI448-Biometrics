# Quick Start Guide - Hand Biometrics

**Get up and running in 5 minutes! ⚡**

---

## Step 1: Install Dependencies (1 minute)

```bash
pip install numpy opencv-python matplotlib
```

Or test your installation:

```bash
python test_installation.py
```

---

## Step 2: Prepare Images (2 minutes)

1. Create folder:
   ```bash
   mkdir hand_images
   ```

2. Add 5 hand images:
   ```
   hand_images/
   ├── hand1.jpg
   ├── hand2.jpg
   ├── hand3.jpg
   ├── hand4.jpg
   └── hand5.jpg
   ```

**Image Requirements:**
- Same hand in all images
- Consistent lighting
- Clear, high-resolution
- Hand visible against background

---

## Step 3: Run the Script (2 minutes)

```bash
python hand_biometrics.py
```

---

## Step 4: Annotate Images

### What to Do

For each image, click **12 points** in this order:

1. **Thumb** (F1): Start → End
2. **Index** (F2): Start → End
3. **Middle** (F3): Start → End
4. **Ring** (F4): Start → End
5. **Pinky** (F5): Start → End
6. **Palm** (F6): Start → End

### Where to Click

✅ **DO**: Click INSIDE the hand (centerline of fingers)

❌ **DON'T**: Click on edges or boundaries

### Controls

| Key | Action |
|-----|--------|
| Left Click | Place point |
| `u` | Undo last point |
| `r` | Restart image |
| Enter | Confirm (when 12 points done) |
| `q` | Quit |

---

## Step 5: View Results

Check the `outputs/` folder:

```
outputs/
├── feature_vectors.csv       ← Feature data
├── distance_matrix.csv        ← Comparison results
├── annotated_demo.png         ← Main visualization (USE THIS FOR DEMO!)
└── annotated_*.png            ← Individual images
```

### What to Show in Demo

Open `outputs/annotated_demo.png` - it shows:
- ✅ Your clicked landmarks (green circles)
- ✅ Finger lines F1-F6 (colored lines)
- ✅ Measurement axes (blue perpendicular lines)
- ✅ Thickness values (numbers on image)

**Perfect for screen recording! 🎥**

---

## Common Issues

### "No images found"
→ Check `hand_images/` folder exists and has .jpg/.png files

### "Cannot import cv2"
→ Run: `pip install opencv-python`

### "Window doesn't appear"
→ Click on taskbar/dock, window might be hidden

### "Wrong point clicked"
→ Press `u` to undo or `r` to restart

---

## Full Documentation

- **README_HAND_BIOMETRICS.md** - Complete instructions
- **ANNOTATION_GUIDE.md** - Detailed annotation help
- **SOLUTION_SUMMARY.md** - Technical details

---

## That's It! 🎉

You now have:
- ✅ 16-dimensional feature vectors for each hand image
- ✅ 5×5 distance matrix comparing all images
- ✅ Beautiful annotated visualizations for your demo
- ✅ CSV files for your submission

**Ready to screen record and submit!**

---

## Quick Reference: Annotation Order

```
Point 0:  Thumb START     (F1) 👍
Point 1:  Thumb END       (F1)
Point 2:  Index START     (F2) ☝️
Point 3:  Index END       (F2)
Point 4:  Middle START    (F3) 🖕
Point 5:  Middle END      (F3)
Point 6:  Ring START      (F4) 💍
Point 7:  Ring END        (F4)
Point 8:  Pinky START     (F5) 🤙
Point 9:  Pinky END       (F5)
Point 10: Palm START      (F6) 🖐️
Point 11: Palm END        (F6)
```

**Remember**: Click INSIDE the hand, not on edges!

---

**Need help?** Check the full documentation files or re-read the assignment instructions.

**Good luck! 🚀**
