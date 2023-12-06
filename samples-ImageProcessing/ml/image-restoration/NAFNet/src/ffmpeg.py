import subprocess
import math
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
    res = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    res = res.strip()
    print(f"ffprobe reply '{res}'")
    n, d = map(int, res.split('/'))
    vars.fps = math.ceil(n / d)
    print(f'detected fps: {vars.fps}')
  except subprocess.CalledProcessError as e:
    sys.exit(f'Error: {e}')
############################################################################
