import cv2
import numpy as np
import os
import time
import glob
import json
from collections import defaultdict
import csv

def calculate_psnr(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("ERROR: couldn't read image(s)")
        return None
    
    # Use OpenCV's PSNR function
    psnr = cv2.PSNR(img1, img2)
    return psnr

def save_results(subrepo, stats, filename="psnr_results.json"):
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # Load existing results if file exists
    results = {}
    if os.path.exists(os.path.join("results", filename)):
        with open(os.path.join("results", filename), 'r') as f:
            results = json.load(f)
    
    # Update results for this resolution
    resolution = subrepo.split('-')[0]  # Get '2k', '4k', or '8k' from the subrepo name
    results[resolution] = {
        'stats': stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save updated results
    with open(os.path.join("results", filename), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to results/{filename}")

def batch_calculate_psnr(subrepo):
    directory = f"images/{subrepo}"
    print(f"\nAnalyzing {subrepo} images...")
    
    # Get all images in directory
    image_files = glob.glob(os.path.join(directory, "*.jpg")) + \
                 glob.glob(os.path.join(directory, "*.jpeg")) + \
                 glob.glob(os.path.join(directory, "*.png"))
    
    image_files = sorted(image_files)[:25]  # Take only first X images
    
    if not image_files:
        print(f"No images found in {directory}")
        return
    
    # Load all images into memory first
    print("Loading all images into memory...")
    images = {}
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is not None:
            bit_depth = img.dtype.itemsize * 8
            max_pixel = float(2 ** bit_depth - 1)
            # Convert to float32 and normalize into [0, 1]
            images[img_path] = (img.astype(np.float32) / max_pixel, max_pixel)
    
    # Total comparisons
    n = len(images)
    total_comparisons = (n * (n-1)) // 2  # n choose 2
    print(f"Will perform {total_comparisons} comparisons in total")
    
    # Statistics collection
    psnr_values = []
    num_comparisons = 0
    pure_calculation_time = 0  # Track time spent only on PSNR calculation
    
    # Start timing for total execution
    total_start_time = time.time()
    
    # Compare each image with every other image
    for i, img1_path in enumerate(image_files):
        if img1_path not in images:
            continue
        img1, max_pixel = images[img1_path]
        
        for j, img2_path in enumerate(image_files[i+1:], i+1):
            if img2_path not in images:
                continue
            img2, _ = images[img2_path]
            
            # Time only the pure PSNR calculation
            pure_start = time.perf_counter()  # More precise for short durations
            
            # Calculate PSNR using pre-processed images-guaranteed identical sizes
            img1_uint8 = (img1 * max_pixel).astype(np.uint8)
            img2_uint8 = (img2 * max_pixel).astype(np.uint8)
            psnr = cv2.PSNR(img1_uint8, img2_uint8)
            pure_calculation_time += time.perf_counter() - pure_start
            
            if psnr != float('inf'):
                psnr_values.append(psnr)
            num_comparisons += 1
            
            if num_comparisons % 100 == 0:
                elapsed = time.time() - total_start_time
                rate = num_comparisons / elapsed
                print(f"Progress: {num_comparisons}/{total_comparisons} comparisons "
                      f"({(num_comparisons/total_comparisons*100):.1f}%) "
                      f"| Rate: {rate:.1f} comp/s")
    
    # Calculate total time
    total_time = time.time() - total_start_time
    comparisons_per_second = num_comparisons / total_time
    avg_time_per_comparison = total_time / num_comparisons
    avg_pure_calculation_time = pure_calculation_time / num_comparisons
    
    # Prepare statistics
    stats = {
        'num_comparisons': num_comparisons,
        'total_time': float(f"{total_time:.2f}"),
        'pure_calculation_time': float(f"{pure_calculation_time:.2f}"),
        'comparisons_per_second': float(f"{comparisons_per_second:.1f}"),
        'avg_time_per_comparison': float(f"{avg_time_per_comparison:.6f}"),
        'avg_pure_calculation_time': float(f"{avg_pure_calculation_time:.6f}")
    }
    
    if psnr_values:
        stats.update({
            'avg_psnr': float(f"{sum(psnr_values)/len(psnr_values):.2f}"),
            'min_psnr': float(f"{min(psnr_values):.2f}"),
            'max_psnr': float(f"{max(psnr_values):.2f}")
        })
    
    print(f"\nStatistics for {subrepo}:")
    print(f"Number of comparisons: {stats['num_comparisons']}")
    print(f"Total execution time: {stats['total_time']:.2f} seconds")
    print(f"Pure calculation time: {stats['pure_calculation_time']:.2f} seconds")
    print(f"Average time per comparison: {stats['avg_time_per_comparison']:.6f} seconds")
    print(f"Average pure calculation time: {stats['avg_pure_calculation_time']:.6f} seconds")
    print(f"Speed: {stats['comparisons_per_second']:.1f} comparisons/second")
    if psnr_values:
        print(f"PSNR range: {stats['min_psnr']} to {stats['max_psnr']} dB")
    print("-" * 50)
    
    save_results(subrepo, stats)
    images.clear()

def main():
    print("DLPP EVAL PSNR Calculation Benchmark")
    print("=" * 30)
    
    while True:
        print("\nOptions:")
        print("1. Run your own analysis")
        print("2. Benchmark PSNR calculation time in bulk")
        print("3. Quit")
        
        choice = input("\nEnter 1-3: ").strip()
        
        if choice == '3':
            break

        elif choice == '2':
            batch_calculate_psnr('2k-uniform')
            batch_calculate_psnr('4k-uniform')
            batch_calculate_psnr('8k-uniform')
            
        elif choice == '1':
            # List all images
            print("\nCalculate PSNR between two images--make sure they have the same dimensions!!\n Images in /other directory:")
            images = list_images_in_directory()
            if not images:
                print("No images in /other")
                continue
                
            for image in images:
                print(image)
            
            choice1 = input("\nFirst image ('q' to quit): ").strip()
            if choice1.lower() == 'q':
                continue
                
            choice2 = input("Second image ('q' to quit): ").strip()
            if choice2.lower() == 'q':
                continue
                
            img1_path = os.path.join('images/other', choice1) if not choice1.startswith('images/other/') else choice1
            img2_path = os.path.join('images/other', choice2) if not choice2.startswith('images/other/') else choice2
            
            start_time = time.time()
            psnr = calculate_psnr(img1_path, img2_path)
            end_time = time.time()
            
            if psnr is not None:
                print(f"\nPSNR: {psnr:.3f} dB")
                print(f"Calculation time: {end_time - start_time:.4f} seconds")
            
            print("\n" + "-"*50)  # Separator line

        else:
            print("Try again")

def list_images_in_directory(directory="images/other"):
    image_extensions = ['.jpg', '.jpeg', '.png']
    images = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            images.append(file)
    
    return sorted(images)  # Sort alphabetically

def calculate_average_psnr(video1_frames, video2_frames, output_file='psnr_results.csv', src_resolution=None, dst_resolution=None, dlpp_version=None):
    """
    Calculates PSNR for each frame pair and writes results to a CSV file, including the average.
    CSV columns: frame,psnr
    Always writes test details as comments at the top of the file.
    """
    if video1_frames is None or video2_frames is None:
        raise ValueError("Frame list(s) are None")
    if len(video1_frames) != len(video2_frames):
        raise ValueError("Frame lists must be of the same length.")
    if src_resolution is None or dst_resolution is None or dlpp_version is None:
        raise ValueError("src_resolution, dst_resolution, and dlpp_version must be provided.")
    
    successful_compares = 0
    rolling_psnr = 0
    psnr_values = []
    with open(output_file, 'w', newline='') as f:
        # Write test details as comments
        f.write(f"# Source resolution: {src_resolution}\n")
        f.write(f"# Destination resolution: {dst_resolution}\n")
        f.write(f"# DLPP version: {dlpp_version}\n")
        writer = csv.writer(f)
        writer.writerow(['frame', 'psnr'])
        for i in range(len(video1_frames)):
            psnr = cv2.PSNR(video1_frames[i], video2_frames[i])
            psnr_values.append(psnr)
            writer.writerow([i, psnr])
            if psnr:
                rolling_psnr += psnr
                successful_compares += 1
        avg_psnr = rolling_psnr / successful_compares if successful_compares > 0 else 0
        writer.writerow(['average', avg_psnr])

    # Plot PSNR per frame over time
    try:
        import matplotlib.pyplot as plt
        frame_indices = list(range(len(psnr_values)))
        plt.figure(figsize=(10, 5))
        plt.plot(frame_indices, psnr_values, marker='o', linestyle='-', color='b')
        plt.title('PSNR per Frame Over Video')
        plt.xlabel('Frame Index')
        plt.ylabel('PSNR (dB)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("matplotlib is not installed. Skipping PSNR plot.")

    return avg_psnr

if __name__ == '__main__':
    main()