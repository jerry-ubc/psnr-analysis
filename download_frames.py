import yt_dlp
import cv2
import argparse
import os

def download_video(url):
    """Download video in 8K (4320p) quality."""
    video_id = url.split('=')[1]
    output_path = f'temp/{video_id}_8k.mp4'
    
    if os.path.exists(output_path):
        print(f"Video {video_id} already downloaded, skipping...")
        return output_path
        
    print(f"Downloading {url} in 8K...")
    ydl_opts = {
        'format': 'bestvideo[height=4320][ext=mp4]+bestaudio[ext=m4a]',
        'outtmpl': output_path,
        'quiet': True,
        'cookiesfrombrowser': ('firefox', None, None, None),
        'merge_output_format': 'mp4',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            if not os.path.exists(output_path):
                print("Error: Failed to download 8K version")
                return None
            return output_path
        except Exception as e:
            print(f"Error downloading video: {str(e)}")
            return None

def extract_frames(video_path, frame_interval=30):
    """Extract frames from video at specified intervals."""
    if not video_path or not os.path.exists(video_path):
        print("Invalid video path")
        return
        
    video_id = os.path.basename(video_path).split('_')[0]
    output_dir = f"images/8k-original"
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"\nVideo info:")
    print(f"Total frames: {total_frames}")
    print(f"FPS: {fps}")
    print(f"Duration: {total_frames/fps:.2f} seconds")
    print(f"Extracting one frame every {frame_interval} frames...")
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Only save if it's exactly 8K (4320p)
            if height == 4320:
                frame_path = os.path.join(output_dir, f"{video_id}_frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved_count += 1
            else:
                print(f"Error: Video is not 8K (found {height}p)")
                cap.release()
                return 0
            
        frame_count += 1
    
    cap.release()
    print(f"\nExtracted {saved_count} frames at {width}x{height} resolution")
    return saved_count

def main():
    parser = argparse.ArgumentParser(description='Download 8K video from YouTube and extract frames')
    parser.add_argument('-u', '--urls', nargs='+', required=True, help='YouTube URLs to download')
    parser.add_argument('-i', '--interval', type=int, default=30, help='Extract one frame every N frames (default: 30)')
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('temp', exist_ok=True)
    os.makedirs('images/8k-original', exist_ok=True)
    
    total_frames = 0
    
    for url in args.urls:
        # Download video
        video_path = download_video(url)
        if video_path:
            # Extract frames
            frames_saved = extract_frames(video_path, args.interval)
            total_frames += frames_saved
            
            # Delete video file to save space
            os.remove(video_path)
            print(f"Deleted temporary video file: {video_path}")
    
    print(f"\nTotal frames extracted: {total_frames}")
    print("You can now run resize_and_verify.py to process the extracted frames")

if __name__ == '__main__':
    main() 