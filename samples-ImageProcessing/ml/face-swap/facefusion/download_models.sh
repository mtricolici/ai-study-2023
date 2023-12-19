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

dl_model "face-recognizer" "https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx"
dl_model "face-recognizer" "https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_simswap.onnx"

dl_model "face-enh" "https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx"
dl_model "face-enh" "https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.2.onnx"
dl_model "face-enh" "https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.3.onnx"
dl_model "face-enh" "https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx"
dl_model "face-enh" "https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_256.onnx"
dl_model "face-enh" "https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_512.onnx"
dl_model "face-enh" "https://github.com/facefusion/facefusion-assets/releases/download/models/restoreformer.onnx"

echo "Models downloaded fine ;)"
