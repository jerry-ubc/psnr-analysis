import cv2
import numpy as np
from typing import List, Tuple, Union
import sys
from frame_psnr import calculate_average_psnr
import av  # PyAV for AV1 support
import os
import argparse
import shutil

import subprocess


def make_frame_pairs(video1_path: str, video2_path: str, sampling_rate: int = 100):
    # Using PyAV, CV2 doesn't support AV1 encodings
    container1 = av.open(video1_path)
    container2 = av.open(video2_path)

    stream1 = container1.streams.video[0]
    stream2 = container2.streams.video[0]

    total_frames1 = stream1.frames
    total_frames2 = stream2.frames
    if total_frames1 != total_frames2:
        print("CAUTION: frame mismatch between videos")
    print(f"Total frames: {total_frames1}")

    # Calculate step size based on percentage
    if sampling_rate <= 0 or sampling_rate > 100:
        raise ValueError("sampling_rate must be in (0, 100]")
    step = max(1, int(total_frames1 * (sampling_rate / 100.0)))
    print(f"Sampling every {step} frames (sampling rate: {sampling_rate}%)")

    # Migrate to new variables: lo_stream and hi_stream
    frame1 = next(container1.decode(stream1))
    frame2 = next(container2.decode(stream2))
    if frame1.width > frame2.width:
        lo_container, hi_container = container2, container1
        lo_stream, hi_stream = stream2, stream1
    else:
        lo_container, hi_container = container1, container2
        lo_stream, hi_stream = stream1, stream2

    # Reset containers to start
    lo_container.seek(0)
    hi_container.seek(0)

    lo_res_frames = []
    hi_res_frames = []
    lo_frames = lo_container.decode(lo_stream)
    hi_frames = hi_container.decode(hi_stream)

    idx = 0
    while True:
        try:
            lo_frame = next(lo_frames)
            hi_frame = next(hi_frames)
        except StopIteration:
            break
        if idx % step == 0:
            print(f"Capturing frame: {idx}")
            lo_res_frames.append(lo_frame.to_ndarray(format='bgr24'))       #TODO: what is bgr24 (not important rn)
            hi_res_frames.append(hi_frame.to_ndarray(format='bgr24'))
        idx += 1

    src_resolution = f"{lo_stream.width}x{lo_stream.height}"
    dst_resolution = f"{hi_stream.width}x{hi_stream.height}"

    return lo_res_frames, hi_res_frames, src_resolution, dst_resolution

def write_frames_to_png(frames, out_dir, lo):
    os.makedirs(out_dir, exist_ok=True)
    for i, frame in enumerate(frames):
        if lo:
            filename = os.path.join(out_dir, f"lo_{i:05d}.png")
        else:
            filename = os.path.join(out_dir, f"hi_{i:05d}.png")
        cv2.imwrite(filename, frame)

def calculate_psnr(video1_path: str, video2_path: str, capture_frames: bool = False, run_upscaling: bool = False, sampling_rate: int = 0, model_trim: str="LOW"):
    if run_upscaling:
        dlpp_model = choose_model()
    else:
        dlpp_model = None
    if capture_frames:
        # Clear lo_res_frames and hi_res_frames directories
        for d in ['lo_res_frames', 'hi_res_frames', 'upscaled_frames']:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)
        lo_res_frames, hi_res_frames, src_resolution, dst_resolution = make_frame_pairs(video1_path, video2_path, sampling_rate)
        write_frames_to_png(lo_res_frames, 'lo_res_frames', lo=True)
        write_frames_to_png(hi_res_frames, 'hi_res_frames', lo=False)
    else:
        # If not capturing, infer resolutions from first images in folders
        lo_files = sorted([f for f in os.listdir('lo_res_frames') if f.endswith('.png')])
        hi_files = sorted([f for f in os.listdir('hi_res_frames') if f.endswith('.png')])
        lo_img = cv2.imread(os.path.join('lo_res_frames', lo_files[0]))
        hi_img = cv2.imread(os.path.join('hi_res_frames', hi_files[0]))
        src_resolution = f"{lo_img.shape[1]}x{lo_img.shape[0]}"
        dst_resolution = f"{hi_img.shape[1]}x{hi_img.shape[0]}"

    lo_res_dir = 'lo_res_frames'
    remote_dir = '/data/local/tmp'
    remote_lo_res = f'{remote_dir}/lo_res_frames'
    remote_upscaled = f'{remote_dir}/upscaled_frames'
    if run_upscaling:
        local_model_path = os.path.join('dlpp_models', dlpp_model)

        # Try to get root and remount system as read-write
        subprocess.run(['adb', 'root'], check=True)
        subprocess.run(['adb', 'remount'], check=True)

        # Now push libcudart.so to vendor/lib64
        subprocess.run(['adb', 'push', 'libcudart.so', 'vendor/lib64/'], check=True)

        # 1. Push lo_res_frames and model binary
        subprocess.run(['adb', 'push', lo_res_dir, remote_dir], check=True)
        subprocess.run(['adb', 'push', local_model_path, remote_dir], check=True)

        subprocess.run(['adb', 'shell', f'mkdir -p {remote_upscaled}'], check=True)
        # 2. For each frame, run the binary
        frame_files = sorted([f for f in os.listdir(lo_res_dir) if f.startswith('lo_') and f.endswith('.png')])
        for i, frame_file in enumerate(frame_files):
            input_path = f'{remote_lo_res}/{frame_file}'
            output_path = f'{remote_upscaled}/upscaled_{i:05d}.png'
            command = (     #TODO: parametrize this step so no user intervention is required
                f'cd {remote_dir} && chmod +x {dlpp_model} && '
                f'./{dlpp_model} {input_path} {output_path} -dst_h 4320 -dst_w 7680 -runs 100 -model DLPP_{model_trim}'   # Use for 1080p -> 8K
                # f'./{dlpp_model} {input_path} {output_path} -runs 100 -model DLPP_{model_trim}'                           # Use for 4K -> 8K
                # f'./{dlpp_model} {input_path} {output_path} -dst_h 2160 -dst_w 3840 -runs 100 -model DLPP_{model_trim}'   # Use for 1080p -> 4K
            )
            print(f"Running: {command}")
            subprocess.run(['adb', 'shell', command], check=True)
        # Remove local upscaled_frames directory if it exists
        if os.path.exists('upscaled_frames'):
            shutil.rmtree('upscaled_frames')
        # 3. Pull upscaled_frames/ from device, then cleanup
        subprocess.run(['adb', 'pull', remote_upscaled, 'upscaled_frames'], check=True)
    cleanup_cmd = 'rm -rf /data/local/tmp/lo_res_frames /data/local/tmp/upscaled_frames'
    subprocess.run(['adb', 'shell', cleanup_cmd], check=True)

    # Calculate PSNR between upscaled_frames and hi_res_frames
    files = sorted([f for f in os.listdir('upscaled_frames') if f.endswith('.png')])    #TODO: is this line necessary
    upscaled_frames = load_frames('upscaled_frames')
    hi_res_frames = load_frames('hi_res_frames')

    avg_psnr = calculate_average_psnr(
        video1_frames=upscaled_frames, video2_frames=hi_res_frames,
        src_resolution=src_resolution, dst_resolution=dst_resolution, dlpp_version=f"DLPP_{model_trim}",
        output_file='psnr_results.csv'
    )
    print(f"PSNR: {avg_psnr}")

def choose_model():
    models_dir = 'dlpp_models'
    models = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
    print("Choose your model:")
    for idx, model in enumerate(models):
        print(f"  {idx+1}. {model}")
    choice = input(f"Enter 1-{len(models)}: ").strip()
    try:
        model_idx = int(choice) - 1
        assert 0 <= model_idx < len(models)
    except (ValueError, AssertionError):
        print("Invalid choice. Exiting.")
        sys.exit(1)
    chosen_model = models[model_idx]
    return models[model_idx]

def load_frames(dir_path):
    """
    Loads all PNG frames from the given directory into a list of numpy arrays, sorted by filename.
    Returns: list of numpy arrays
    """
    frames = []
    files = sorted([f for f in os.listdir(dir_path) if f.endswith('.png')])
    for f in files:
        img = cv2.imread(os.path.join(dir_path, f))
        if img is not None:
            frames.append(img)
    return frames

def main():
    """
    Main function to extract frames and calculate average PSNR from two video files provided as command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Video PSNR pipeline")
    parser.add_argument('video1_name', type=str, help='e.g., japan1080.mp4')
    parser.add_argument('video2_name', type=str, help='e.g., japan8k.mp4')
    parser.add_argument('frame_sampling_rate', type=int, help='Percentage of frames sampled (e.g., 5 for every 5% of frames)')
    parser.add_argument('model_trim', type=str, help="LOW, MEDIUM, or HIGH")
    parser.add_argument('--skip-capture', action='store_true', default=False, help='If set, skips frame extraction from videos; skips to upscaling (default: False)')
    parser.add_argument('--skip-upscale', action='store_true', default=False, help='If set, skips upscaling; skips to PSNR calculation (default: False)')
    args = parser.parse_args()

    capture_frames = not args.skip_capture
    run_upscaling = not args.skip_upscale

    video1_path = f"videos/{args.video1_name}"
    video2_path = f"videos/{args.video2_name}"

    if capture_frames:
        print("Capturing frames, so automatically unsetting --skip-upscale")
        args.upscale_frames = True
    if not run_upscaling:
        print ("Skipping upscaling, so automatically setting --skip-capture")
        args.capture_frames = False

    calculate_psnr(video1_path, video2_path, capture_frames, run_upscaling, sampling_rate=args.frame_sampling_rate, model_trim=args.model_trim)

if __name__ == '__main__':
    main()
