import subprocess
import math
import os
import sys
import json
import vars

############################################################################
def detect_fps():
  if vars.fps is not None and isinstance(vars.fps, int) and vars.fps > 0:
    print(f'using FPS={vars.fps}')
    return

  print('FPS was not specified. I will try to detect it')
  try:
    cmd = [
      'ffprobe',
      '-v', 'error',
      '-select_streams', 'v:0',
      '-show_entries', 'stream=r_frame_rate',
      '-of', 'default=noprint_wrappers=1:nokey=1',
      vars.source_file
    ]
    print(cmd)
    res = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    res = res.strip()
    print(f"ffprobe reply '{res}'")
    n, d = map(int, res.split('/'))
    vars.fps = math.ceil(n / d)
    print(f'detected fps: {vars.fps}')
  except subprocess.CalledProcessError as e:
    sys.exit(f'Error: {e}')
############################################################################
def check_audio_stream_presence():
  try:
    cmd = [
      'ffprobe',
      '-v', 'error',
      '-select_streams', 'a:0',  # Select the first audio stream (if it exists)
      '-show_entries', 'stream=codec_name',
      '-of', 'json',
      vars.source_file
    ]
    result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    info = json.loads(result)
    vars.has_audio = 'streams' in info and len(info['streams']) > 0
  except subprocess.CalledProcessError as e:
    sys.exit(f'Error: {e}')
############################################################################
def save_audio_stream():
  os.makedirs('/images/tmp', exist_ok=True)
  try:
    cmd = [
      'ffmpeg',
      '-i', vars.source_file,
      '-vn', '-acodec', 'libmp3lame', '-y',
      '/images/tmp/audio.mp3'
    ]
    subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
  except subprocess.CalledProcessError as e:
    sys.exit(f'Error: {e}')
############################################################################
def extract_frames():
  print(f'Extracting frames ...')
  os.makedirs('/images/tmp', exist_ok=True)
  try:
    cmd = [
      'ffmpeg',
      '-i', vars.source_file,
      '-vf', f'fps={vars.fps}',
      '/images/tmp/%04d.png'
    ]
    print(cmd)
    subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    print('Frames extracted fine')
  except subprocess.CalledProcessError as e:
    sys.exit(f'Error: {e}')

############################################################################
def create_output_video():
  print(f'Saving frames to output video ...')
  try:
    cmd = [
      'ffmpeg',
      '-framerate', f'{vars.fps}',
      '-i', '/images/tmp/%04d.png'
    ]

    if vars.has_audio:
      cmd.extend(['-i', '/images/tmp/audio.mp3'])

    cmd.extend([
      '-pix_fmt', 'yuv420p', '-shortest', '-y',
      vars.target_file
    ])
    print(cmd)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
  except subprocess.CalledProcessError as e:
    print(f'Stdout: {result.stdout}')
    print(f'Stderr: {result.stderr}')
    sys.exit(f'Error: {e}')
############################################################################
