#!/bin/bash
set -e

IMG=my-nafnet-img-rest

#model="NAFNet-REDS-width64" # <<< deblurr but very aggresive - unreal images :)
#model="NAFNet-SIDD-width64" # << noise removal
#model="NAFNet-GoPro-width64" # << deblur but not very aggresive :)

rm -rf .images/tmp
rm -f .images/encode-to-run-on-host.txt

docker run \
  --gpus all \
  -it --rm \
  --network=none \
  -e PYTHONPATH=/nafnet \
  -w /app \
  -v $(pwd)/src:/app:ro \
  -v $(pwd)/.images:/images \
  -v $(pwd)/models:/models:ro \
  -v $(pwd)/.insightface:/home/python/.insightface \
  $IMG \
  python main.py \
    -i /images/src.mp4 \
    -o /images/result.mp4 \
    --device cuda \
    --skip-encode 1 \
    -m NAFNet-REDS-width64

if [ -f .images/encode-to-run-on-host.txt ]; then
  echo "detected .images/encode-to-run-on-host.txt file. I will run it on host"
  source .images/encode-to-run-on-host.txt
  rm -f .images/encode-to-run-on-host.txt
fi

echo -e "\nvideo process finished!"

