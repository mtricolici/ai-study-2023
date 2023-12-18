#!/bin/bash
set +e

function dl_model() {
  local mf="$1"
  local murl="$2"

  if [ ! -f "models/${mf}" ]; then
    echo "Model 'models/${mf}' not found. let's download it ..."
    wget -q -O "models/$mf" "$murl"
    if [ $? -ne 0 ]; then
      rm -f "models/$mf"
      echo "Downloading model $mf failed :("
      exit 1
    fi
  else
    echo "Model 'models/${mf}' already exists ;)"
  fi
}

mkdir -p models

dl_model "blendswap_256.onnx" "https://github.com/facefusion/facefusion-assets/releases/download/models/blendswap_256.onnx"
dl_model "inswapper_128.onnx" "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx"
dl_model "inswapper_128_fp16.onnx" "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx"
dl_model "simswap_256.onnx" "https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_256.onnx"
dl_model "simswap_512_unofficial.onnx" "https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_512_unofficial.onnx"

echo "Models downloaded fine ;)"
