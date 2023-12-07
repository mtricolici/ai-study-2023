import os
import torch
from collections import OrderedDict
import basicsr.models as bm
import basicsr.train as bt
import basicsr.utils as bu
import vars




############################################################################
def load_model():
  model_ops_path = f'/app/{vars.model_name}.yml'
  opt = bt.parse(model_ops_path, is_train=False)
  opt['num_gpu'] = torch.cuda.device_count()
  opt['dist'] = False
  model = bm.create_model(opt)
  return model
############################################################################
def process_single_frame(model, path):
  # read image
  file_client = bu.FileClient('disk')
  img = file_client.get(path, None)
  img = bu.imfrombytes(img, float32=True)
  img = bu.img2tensor(img, bgr2rgb=True, float32=True)

  # Feed model
  model.feed_data(data={'lq': img.unsqueeze(dim=0)})
  if model.opt['val'].get('grids', False):
    model.grids()
  model.test()
  if model.opt['val'].get('grids', False):
    model.grids_inverse()

  # Save image
  visuals = model.get_current_visuals()
  img = bu.tensor2img([visuals['result']])
  bu.imwrite(img, path)
############################################################################
def process_frames():
  files = [f for f in os.listdir('/images/tmp/') if f.endswith('.png')]
  files.sort()

  total = len(files)

  model = load_model()

  for i, f in enumerate(files, start=1):
    path = os.path.join('/images/tmp', f)
    process_single_frame(model, path)
    done = (i / total) * 100.0
    print(f'{f} done. {done:.2f} processed')
############################################################################
