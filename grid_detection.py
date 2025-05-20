import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from progress.bar import Bar

def detect_grid_walls(image_path):
    """Detects grid lines in the image and returns a white-on-black mask."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None

    # Preprocessing
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    edges = cv2.Canny(thresh, 30, 100, apertureSize=3)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Create a black mask
    line_mask = np.zeros_like(img)

    # Detect lines
    lines = cv2.HoughLinesP(
        dilated,
        rho=1,
        theta=np.pi/180,
        threshold=40,
        minLineLength=30,
        maxLineGap=5
    )

    # Draw white lines on the black mask
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)

    return line_mask

def batch_process_folder(input_dir, output_dir):
    """
    Applies detect_grid_walls to every image in input_dir and writes
    the white-on-black masks into output_dir, showing progress.
    """
    os.makedirs(output_dir, exist_ok=True)
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    # Gather only image files
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]
    if not files:
        print("No images found in:", input_dir)
        return

    # Create a progress bar
    bar = Bar('Processing', max=len(files), suffix='%(index)d/%(max)d img')

    for fname in files:
        in_path  = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, fname)

        mask = detect_grid_walls(in_path)
        if mask is None:
            bar.write(f"Skipped {fname} (could not read)")
        else:
            cv2.imwrite(out_path, mask)

        bar.next()

    bar.finish()
    print("All done! Masks saved to:", output_dir)

if __name__ == "__main__":
    # Hide the root tkinter window
    root = tk.Tk()
    root.withdraw()

    # Ask user to pick the input folder
    input_folder = filedialog.askdirectory(title='Select Input Folder')
    if not input_folder:
        print('No input folder selected. Exiting.')
        sys.exit(1)

    # Build output folder path next to input, named "<base>_mask_output"
    parent_dir = os.path.dirname(input_folder)
    base_name  = os.path.basename(input_folder)
    output_folder = os.path.join(parent_dir, f"{base_name}_mask_output")

    batch_process_folder(input_folder, output_folder)