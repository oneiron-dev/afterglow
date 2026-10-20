#!/usr/bin/env bash
# Env + weights for the drafter tune (CPU only). Idempotent. Pinned, sha-verified downloads.
set -euo pipefail
. "$(dirname "$0")/env.sh"
mkdir -p $W/{models,data,hs,ckpt,gguf,logs}
cd $W
if [ ! -x venv/bin/python ]; then uv venv --python 3.12 venv; fi
uv pip install --python venv/bin/python torch==2.13.0 transformers==5.17.0 safetensors==0.8.0 accelerate bitsandbytes==0.50.2 \
  numpy==2.3.5 tokenizers==0.23.3 sentencepiece==0.2.2 huggingface-hub==1.33.0 hf-xet==1.7.0 jinja2 protobuf einops 2>&1 | tail -3
venv/bin/python -c "import torch,transformers,bitsandbytes;print('env ok',torch.__version__,transformers.__version__,bitsandbytes.__version__)"
python3 $S/hf_fetch.py meta-models/Muse-Glimmer-30B-assistant e8192f3a8f617f74be2ce220360c89ef4789f39f $W/models/drafter-stock &
P1=$!
python3 $S/hf_fetch.py unsloth/Muse-Glimmer-30B-unsloth-bnb-4bit ecedda395d4d099b477d135a9a573eedaa5a7aad $W/models/target-nf4
wait $P1
echo SETUP-DONE
