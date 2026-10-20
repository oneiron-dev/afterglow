#!/usr/bin/env bash
# HF drafter dir -> GGUF bf16 -> Q4_K_M with llama.cpp 69f201a (the same commit as the eval llama-server).
# The quant mirrors the official dflash-Muse-Glimmer-30B-Q4_K_M.gguf layout: Q4_K, with attn_v and ffn_down at Q6_K.
# usage: convert_gguf.sh HF_DRAFTER_DIR NAME      (writes gguf/NAME-bf16.gguf and gguf/NAME-Q4_K_M.gguf)
set -euo pipefail
. "$(dirname "$0")/env.sh"
mkdir -p $W/gguf
if [ ! -f $W/gguf/$2-bf16.gguf ]; then
  $W/venv/bin/python $LC/convert_hf_to_gguf.py "$1" --target-model-dir $W/models/target-nf4 --outtype bf16 \
     --outfile $W/gguf/$2-bf16.gguf 2>&1 | grep -v Writing | tail -3
fi
rm -f $W/gguf/$2-Q4_K_M.gguf
LD_LIBRARY_PATH=$R $R/llama-quantize --tensor-type attn_v=q6_k --tensor-type ffn_down=q6_k \
   $W/gguf/$2-bf16.gguf $W/gguf/$2-Q4_K_M.gguf Q4_K_M 2>&1 | tail -2
ls -la $W/gguf/$2-*.gguf
sha256sum $W/gguf/$2-Q4_K_M.gguf
