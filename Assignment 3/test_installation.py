#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly.
Run this before running the main hand_biometrics.py script.
"""

import sys

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing package imports...")
    print("-" * 50)
    
    # Test numpy
    try:
        import numpy as np
        print(f"✓ numpy {np.__version__} - OK")
    except ImportError as e:
        print(f"✗ numpy - FAILED: {e}")
        return False
    
    # Test OpenCV
    try:
        import cv2
        print(f"✓ opencv-python {cv2.__version__} - OK")
    except ImportError as e:
        print(f"✗ opencv-python - FAILED: {e}")
        return False
    
    # Test matplotlib
    try:
        import matplotlib
        print(f"✓ matplotlib {matplotlib.__version__} - OK")
    except ImportError as e:
        print(f"✗ matplotlib - FAILED: {e}")
        return False
    
    print("-" * 50)
    return True


def test_opencv_functionality():
    """Test that OpenCV can perform basic operations."""
    print("\nTesting OpenCV functionality...")
    print("-" * 50)
    
    try:
        import cv2
        import numpy as np
        
        # Create a test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[25:75, 25:75] = [255, 0, 0]  # Red square
        
        # Test color conversion
        gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
        
        # Test resize
        resized = cv2.resize(test_img, (50, 50))
        
        print("✓ Image creation - OK")
        print("✓ Color conversion - OK")
        print("✓ Image resize - OK")
        print("-" * 50)
        return True
        
    except Exception as e:
        print(f"✗ OpenCV functionality test FAILED: {e}")
        print("-" * 50)
        return False


def test_matplotlib_functionality():
    """Test that matplotlib can create figures (without displaying)."""
    print("\nTesting matplotlib functionality...")
    print("-" * 50)
    
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Create a simple plot
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        plt.close(fig)
        
        print("✓ Figure creation - OK")
        print("✓ Plotting - OK")
        print("-" * 50)
        return True
        
    except Exception as e:
        print(f"✗ Matplotlib functionality test FAILED: {e}")
        print("-" * 50)
        return False


def test_file_system():
    """Test that required directories can be created."""
    print("\nTesting file system access...")
    print("-" * 50)
    
    try:
        from pathlib import Path
        
        # Check if we can create directories
        test_dir = Path("./test_temp_dir")
        test_dir.mkdir(exist_ok=True)
        
        # Check if we can write files
        test_file = test_dir / "test.txt"
        test_file.write_text("test")
        
        # Check if we can read files
        content = test_file.read_text()
        
        # Clean up
        test_file.unlink()
        test_dir.rmdir()
        
        print("✓ Directory creation - OK")
        print("✓ File write - OK")
        print("✓ File read - OK")
        print("-" * 50)
        return True
        
    except Exception as e:
        print(f"✗ File system test FAILED: {e}")
        print("-" * 50)
        return False


def check_python_version():
    """Check Python version."""
    print("\nChecking Python version...")
    print("-" * 50)
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 7:
        print(f"✓ Python {version_str} - OK (>= 3.7 required)")
        print("-" * 50)
        return True
    else:
        print(f"✗ Python {version_str} - FAILED (>= 3.7 required)")
        print("-" * 50)
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("HAND BIOMETRICS - INSTALLATION TEST")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= check_python_version()
    all_passed &= test_imports()
    all_passed &= test_opencv_functionality()
    all_passed &= test_matplotlib_functionality()
    all_passed &= test_file_system()
    
    # Summary
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("=" * 50)
        print("\nYour environment is ready!")
        print("You can now run: python hand_biometrics.py")
        print("\nMake sure to:")
        print("1. Place 5 hand images in ./hand_images/ folder")
        print("2. Read ANNOTATION_GUIDE.md for annotation instructions")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 50)
        print("\nPlease fix the issues above before running the main script.")
        print("\nTo install missing packages, run:")
        print("  pip install numpy opencv-python matplotlib")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
