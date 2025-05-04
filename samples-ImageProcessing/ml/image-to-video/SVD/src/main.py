import os
import random
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video

if torch.cuda.is_available():
    print("CUDA is okay ;)")

else:
    print("CUDA not available :( exit")
    os._exit(1)

if not os.path.isfile('/content/input.png'):
    print('pls add input image to content/input.png !')
    print('Make sure to make it 1024x576 or I will resize it anyway')
    os._exit(1)

device = torch.device("cuda")

pipeline = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
).to(device)

pipeline.enable_model_cpu_offload()
pipeline.unet.enable_forward_chunking()

image = load_image("/content/input.png")
#image = image.resize((1024, 576))

seed = random.randint(0, 1_000_000)
print(f"Using seed: {seed:04d}")

generator = torch.manual_seed(seed)
frames = pipeline(image, decode_chunk_size=2, generator=generator).frames[0]
print('FRAMES generated ! exporting to video ...')

export_to_video(frames, f"/content/output_{seed:04d}.mp4", fps=7)
