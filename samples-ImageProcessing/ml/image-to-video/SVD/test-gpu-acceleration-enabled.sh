#!/bin/bash

IMG=my-svd

docker run \
  --gpus all,capabilities=video \
  -it --rm \
  --network=none \
  $IMG \
  python -c "import torch; print(f'Using GPU: {torch.cuda.is_available()}')"
