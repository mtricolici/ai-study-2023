#!/bin/bash
set +e

function dl_model() {
  local mt="$1"   # model type
  local murl="$2" # url to download the model

  fn=${murl##*/} # Get file name from url
  fn="models/$mt/$fn" # Append path

  mkdir -p "models/${mt}"

  if [ ! -f "$fn" ]; then
    echo "Model '${fn}' not found. let's download it ..."
    wget -q -O "${fn}" "$murl"
    if [ $? -ne 0 ]; then
      rm -f "${fn}"
      echo "Downloading model ${fn} failed :("
      exit 1
    fi
  else
    echo "Model '${fn}' already exists ;)"
  fi
}


dl_model "face-swap" "https://github.com/facefusion/facefusion-assets/releases/download/models/blendswap_256.onnx"
dl_model "face-swap" "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
dl_model "face-swap" "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx"
dl_model "face-swap" "https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_256.onnx"
dl_model "face-swap" "https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_512_unofficial.onnx"

dl_model "face-detect" "https://github.com/facefusion/facefusion-assets/releases/download/models/retinaface_10g.onnx"
dl_model "face-detect" "https://github.com/facefusion/facefusion-assets/releases/download/models/yunet_2023mar.onnx"

echo "Models downloaded fine ;)"
