# File given by Daniel K

import yt_dlp
import argparse
import os

def download_video(url, resolution):
    id = url.split('=')[1]

    ydl_opts = {
        'format': f'bestvideo[height={resolution}][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': f'videos/{id}_{resolution}.mp4',
        'quiet': True,
        'cookiesfrombrowser': ('firefox', None, None, None),
        'merge_output_format': 'mp4',
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

if __name__ == '__main__':
    os.makedirs('videos', exist_ok=True)

    parser = argparse.ArgumentParser(description='Download a video from a url with a specific resolution')
    parser.add_argument('-u', '--url', type=str, required=True, help='The url of the video to download')
    parser.add_argument('-r', '--resolution', type=int, required=True, help='The resolution of the video to download')
    args = parser.parse_args()

    download_video(args.url, args.resolution)