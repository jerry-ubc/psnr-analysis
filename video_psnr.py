import cv2
import numpy as np
from typing import List, Tuple, Union
import sys
from calculate_psnr import calculate_average_psnr


#TODO: make documentation
#TODO: return 2 arrays, one for lo res, one for hi res
def make_frame_pairs(video1_path: str, video2_path: str, offset: int) -> List[np.ndarray]:
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    print(f"trying path: {video1_path}")
    
    if not cap1.isOpened() or not cap2.isOpened():
        raise ValueError("ERROR: Couldn't open video file(s)")
 
    video1_width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    video2_width = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))

    lo_res_frames = cap1
    hi_res_frames = cap2

    if (video1_width > video2_width):
        lo_res_video = cap2
        hi_res_video = cap1
    else:
        lo_res_video = cap1
        hi_res_video = cap2
    
    # Get video properties
    frame_count1 = int(lo_res_frames.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count2 = int(hi_res_frames.get(cv2.CAP_PROP_FRAME_COUNT))

    # print(f"Video 1 frames: {frame_count1}, video 2 frames: {frame_count2}")
    
    if frame_count1 != frame_count2:
        print("CAUTION: Differing frame counts")

    lo_res_frames = []
    hi_res_frames = []

    # num_frames = frame_count1 // offset
    for i in range(0, frame_count1, offset):
        # print(f"getting frame: {i}/{frame_count1}")
        lo_res_video.set(cv2.CAP_PROP_POS_FRAMES, i)
        hi_res_video.set(cv2.CAP_PROP_POS_FRAMES, i)
        lo_ret, lo_frame = lo_res_video.read()
        hi_ret, hi_frame = hi_res_video.read()
        if not lo_ret or not hi_ret:
            break
        lo_res_frames.append(lo_frame)
        hi_res_frames.append(hi_frame)
    # print(f"frames captured: {len(lo_res_frames)}")
    
    # # Save the first 5 pairs of frames as images for verification
    # for idx in range(0, min(10, len(frame_pairs)), 2):
    #     cv2.imwrite(f'frame_{idx//2}_video1.jpg', frame_pairs[idx])
    #     cv2.imwrite(f'frame_{idx//2}_video2.jpg', frame_pairs[idx+1])
    
    return lo_res_frames, hi_res_frames

def main():
    """
    Main function to extract frames and calculate average PSNR from two video files provided as command-line arguments.
    """
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <video1_path> <video2_path> [offset]")
        sys.exit(1)
    video1_path = f"videos/{sys.argv[1]}"
    video2_path = f"videos/{sys.argv[2]}"
    offset = int(sys.argv[3])
    
    # TODO: currently passing lo_res_frames and hi_res_frames to make_frame pairs
    #       actually, should pass these into DLPP as jpgs, then pass DLPP result and
    #       hi_res_frames into make_frame_pairs then pass that into 
    #       calculate_average_psnr()
    # TODO: also move all these function calls out of main--main should call 1 function
    #       and that should handle the other stuff
    lo_res_frames, hi_res_frames = make_frame_pairs(video1_path, video2_path, offset)
    avg_psnr = calculate_average_psnr(lo_res_frames, hi_res_frames)
    print(f"Average PSNR: {avg_psnr}")

if __name__ == '__main__':
    main()
