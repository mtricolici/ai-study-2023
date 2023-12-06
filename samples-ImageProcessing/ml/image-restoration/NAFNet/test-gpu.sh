#!/bin/bash

IMG=my-nafnet-img-rest

docker run \
  --gpus all \
  -it --rm \
  --network=none \
  $IMG \
  python -c "import torch; print(f'Using GPU: {torch.cuda.is_available()}')"
