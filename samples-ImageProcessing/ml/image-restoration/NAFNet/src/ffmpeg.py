import subprocess
import math
import os
import sys
import json
import re
import vars

VIDEO_ENCODE = {
  "cpu"  : "ffmpeg               -framerate {fps} -thread_queue_size {thread_queue_size} -i /images/tmp/%04d.png {audio} -pix_fmt yuv420p -c:v libx264    -preset slower   -crf 12 -shortest -y",
  "cuda" : "ffmpeg -hwaccel cuda -framerate {fps} -thread_queue_size {thread_queue_size} -i /images/tmp/%04d.png {audio} -pix_fmt yuv420p -c:v h264_nvenc -preset slow     -b:v 5M -shortest -y",
  "amf"  : "ffmpeg -hwaccel amf  -framerate {fps} -thread_queue_size {thread_queue_size} -i /images/tmp/%04d.png {audio} -pix_fmt yuv420p -c:v h264_amf   -quality quality -b:v 5M -shortest -y"
}

############################################################################
def get_video_encode_cmd():
  cmd = VIDEO_ENCODE[vars.device]
  cmd = cmd.replace('{fps}', f'{vars.fps}')
  cmd = cmd.replace('{thread_queue_size}', f'{vars.thread_queue_size}')

  if vars.has_audio:
    cmd = cmd.replace('{audio}', '-i /images/tmp/audio.mp3')
  else:
    cmd = cmd.replace('{audio}', '')

  cmd = re.sub(r'\s+', ' ', cmd) # make just 1 space
  cmd = cmd.split()
  cmd.append(vars.target_file)
  return cmd

############################################################################
def run_executable(cmd):
  try:
    print(cmd)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
      raise subprocess.CalledProcessError(result.returncode, cmd)
  except subprocess.CalledProcessError as e:
    print(f'Stdout: {result.stdout}')
    print(f'Stderr: {result.stderr}')
    sys.exit(f'Error: {e}')
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
  print('Saving frames to output video ...')
  cmd = get_video_encode_cmd()

  if vars.skip_encode:
    print('skip-encode is true. I will run this on your host ;)')
    cmd = " ".join(cmd).replace("/images/", ".images/")
    with open('/images/encode-to-run-on-host.txt', 'w') as file:
      file.write(cmd)
  else:
    run_executable(cmd)
    print(f'Video saved in {vars.target_file} !!! ;)')
############################################################################

