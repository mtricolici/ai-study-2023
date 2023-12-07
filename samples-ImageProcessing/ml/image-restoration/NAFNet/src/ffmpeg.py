import subprocess
import math
import os
import sys
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
def extract_frames():
  print(f'Extracting frames ...')
  os.makedirs('/images/tmp')
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
      '-i', '/images/tmp/%04d.png',
      '-pix_fmt', 'yuv420p', '-shortest',
      vars.target_file
    ]
    print(cmd)
    subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    print('Video is ready ;)')
  except subprocess.CalledProcessError as e:
    sys.exit(f'Error: {e}')
############################################################################
