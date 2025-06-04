import cv2
import numpy as np
import os
import time
import glob
import json
from collections import defaultdict

def calculate_psnr(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("ERROR: couldn't read image(s)")
        return None
    
    # Get bit depth of the image -- needed for numerator of PSNR formula
    bit_depth = img1.dtype.itemsize * 8     # e.g., uint8 = 8 bits, uint16 = 16 bits
    max_pixel = float(2 ** bit_depth - 1)   # e.g., 255 for 8-bit int, 65535 for 16-bit int
    
    # Convert to float for calculations
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Mean Squared Error
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
        
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def save_results(subrepo, stats, filename="psnr_results.json"):
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    stats['subrepo'] = subrepo
    
    # Save results (overwriting previous)
    with open(os.path.join("results", filename), 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"\nResults saved to results/{filename}")

def batch_calculate_psnr(subrepo):
    directory = f"images/{subrepo}"
    print(f"\nAnalyzing {subrepo} images...")
    
    # Get all images in directory
    image_files = glob.glob(os.path.join(directory, "*.jpg")) + \
                 glob.glob(os.path.join(directory, "*.jpeg")) + \
                 glob.glob(os.path.join(directory, "*.png"))
    
    image_files = sorted(image_files)[:50]  # Take only first X images
    
    if not image_files:
        print(f"No images found in {directory}")
        return
    
    # Total comparisons
    n = len(image_files)
    total_comparisons = (n * (n-1)) // 2  # n choose 2
    print(f"Will perform {total_comparisons} comparisons in total")
    
    # Statistics collection
    psnr_values = []
    num_comparisons = 0
    unusual_psnr_count = 0  # Count of PSNR values outside normal range
    
    # Start timing
    start_time = time.time()
    
    # Compare each image with every other image
    for i, img1_path in enumerate(image_files):
        for j, img2_path in enumerate(image_files[i+1:], i+1):
            psnr = calculate_psnr(img1_path, img2_path)
            
            if psnr is not None:
                if psnr != float('inf'):
                    if psnr < 20 or psnr > 50:
                        unusual_psnr_count += 1
                    psnr_values.append(psnr)
                num_comparisons += 1
                
                if num_comparisons % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = num_comparisons / elapsed
                    print(f"Progress: {num_comparisons}/{total_comparisons} comparisons "
                          f"({(num_comparisons/total_comparisons*100):.1f}%) "
                          f"| Rate: {rate:.1f} comp/s")
    
    # Calculate total time
    total_time = time.time() - start_time
    comparisons_per_second = num_comparisons / total_time
    avg_time_per_comparison = total_time / num_comparisons
    
    # Prepare statistics
    stats = {
        'num_comparisons': num_comparisons,
        'total_time': float(f"{total_time:.2f}"),
        'comparisons_per_second': float(f"{comparisons_per_second:.1f}"),
        'avg_time_per_comparison': float(f"{avg_time_per_comparison:.3f}")
    }
    
    if psnr_values:
        stats.update({
            'avg_psnr': float(f"{sum(psnr_values)/len(psnr_values):.2f}"),
            'min_psnr': float(f"{min(psnr_values):.2f}"),
            'max_psnr': float(f"{max(psnr_values):.2f}")
        })
    
    # Print statistics
    print(f"\nStatistics for {subrepo}:")
    print(f"Number of comparisons: {stats['num_comparisons']}")
    print(f"Total execution time: {stats['total_time']:.2f} seconds")
    print(f"Average time per comparison: {stats['avg_time_per_comparison']:.3f} seconds")
    print(f"Speed: {stats['comparisons_per_second']:.1f} comparisons/second")
    if psnr_values:
        print(f"PSNR range: {stats['min_psnr']} to {stats['max_psnr']} dB")
    print("-" * 50)
    
    # Save results to file
    save_results(subrepo, stats)

def main():
    print("PSNR Calculation Benchmark")
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
                
            # Add path prefix
            img1_path = os.path.join('images/other', choice1) if not choice1.startswith('images/other/') else choice1
            img2_path = os.path.join('images/other', choice2) if not choice2.startswith('images/other/') else choice2
            
            # Time the calculation
            start_time = time.time()
            psnr = calculate_psnr(img1_path, img2_path)
            end_time = time.time()
            
            if psnr is not None:
                print(f"\nPSNR: {psnr:.3f} dB")
                print(f"Calculation time: {end_time - start_time:.4f} seconds")
            
            print("\n" + "-"*50)  # Separator line

        else:
            print("Try again")

def list_images_in_directory(directory="images"):
    image_extensions = ['.jpg', '.jpeg', '.png']
    images = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            images.append(file)
    
    return sorted(images)  # Sort alphabetically

if __name__ == '__main__':
    main()