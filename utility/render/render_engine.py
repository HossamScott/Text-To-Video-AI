import time
import os
import tempfile
import zipfile
import platform
import subprocess
from datetime import datetime
from moviepy.editor import (AudioFileClip, CompositeVideoClip, CompositeAudioClip, ImageClip,
                            TextClip, VideoFileClip)
from moviepy.audio.fx.audio_loop import audio_loop
from moviepy.audio.fx.audio_normalize import audio_normalize
import requests

from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

def download_file(url, filename):
    with open(filename, 'wb') as f:
        headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        f.write(response.content)

def search_program(program_name):
    try: 
        search_cmd = "where" if platform.system() == "Windows" else "which"
        return subprocess.check_output([search_cmd, program_name]).decode().strip()
    except subprocess.CalledProcessError:
        return None

def get_program_path(program_name):
    program_path = search_program(program_name)
    return program_path

def get_output_media(audio_file_path, timed_captions, background_video_data, video_server):
    # Create output directory if it doesn't exist
    output_dir = "/app/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE_NAME = os.path.join(output_dir, f"rendered_video_{timestamp}.mp4")
    
    magick_path = get_program_path("magick")
    print(magick_path)
    if magick_path:
        os.environ['IMAGEMAGICK_BINARY'] = magick_path
    else:
        os.environ['IMAGEMAGICK_BINARY'] = '/usr/bin/convert'
    
    visual_clips = []
    for (t1, t2), video_url in background_video_data:
        # Download the video file
        video_filename = tempfile.NamedTemporaryFile(delete=False).name
        download_file(video_url, video_filename)
        
        # Create VideoFileClip from the downloaded file
        video_clip = VideoFileClip(video_filename)
        video_clip = video_clip.set_start(t1)
        video_clip = video_clip.set_end(t2)
        visual_clips.append(video_clip)
    
    audio_clips = []
    audio_file_clip = AudioFileClip(audio_file_path)
    audio_clips.append(audio_file_clip)

    # Create text clips with customizable font settings
    for (t1, t2), text in timed_captions:
        text_clip = TextClip(
            txt=text,
            fontsize=font_settings['size'],
            color=font_settings['color'],
            stroke_color=font_settings['stroke_color'],
            stroke_width=font_settings['stroke_width'],
            font="Arial",
            method="caption",
            size=(1920, None)
        )
        text_clip = text_clip.set_start(t1)
        text_clip = text_clip.set_end(t2)
        text_clip = text_clip.set_position(["center", 800])
        visual_clips.append(text_clip)

    video = CompositeVideoClip(visual_clips)
    
    if audio_clips:
        audio = CompositeAudioClip(audio_clips)
        video.duration = audio.duration
        video.audio = audio

    # Write video file with progress logging
    print(f"Writing video to: {OUTPUT_FILE_NAME}")
    video.write_videofile(
        OUTPUT_FILE_NAME,
        codec='libx264',
        audio_codec='aac',
        fps=25,
        preset='veryfast',
        threads=4,
        logger='bar'
    )
    
    # Clean up downloaded files
    for (t1, t2), video_url in background_video_data:
        video_filename = tempfile.NamedTemporaryFile(delete=False).name
        if os.path.exists(video_filename):
            os.remove(video_filename)

    print(f"Video successfully saved to: {OUTPUT_FILE_NAME}")
    return OUTPUT_FILE_NAME