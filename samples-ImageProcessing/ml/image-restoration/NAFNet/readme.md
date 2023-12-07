Dockerized version of https://github.com/megvii-research/NAFNet

This was tested with the following models:
NAFNet-GoPro-width64.pth
NAFNet-REDS-width64.pth
NAFNet-SIDD-width64.pth

how to run:

1. Download models from: https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models
   into ./models/ folder

2. invoke ./prepare.sh - to build docker image

3. verify your cuda is working: ./test-gpu.sh
```
$./test-gpu.sh
Using GPU: True
```
if you cuda is not working:
- install nvidia drivers
  install Nvidia container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

4. add your image to ./.images/src.png
   now your can test image restore via ./image\_test.sh

5. add your mp4 video to ./.images/src.mp4
   new you can test video restore via ./video\_test.sh

