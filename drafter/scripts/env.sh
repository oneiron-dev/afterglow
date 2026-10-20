# Shared paths for the drafter scripts. Source it; override any variable before running.
#   DRAFTER_WORK  work dir: models/, data/, hs/, ckpt/, gguf/, logs/, venv/   (default: current dir)
#   LLAMA_BIN     dir with llama-server and llama-quantize (and their shared libs)
#   LLAMA_SRC     llama.cpp checkout, for convert_hf_to_gguf.py
#   TARGET_GGUF   the served Glimmer target
S=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
W=${DRAFTER_WORK:-$PWD}
R=${LLAMA_BIN:-$W/llama.cpp/build/bin}
LC=${LLAMA_SRC:-$W/llama.cpp}
T=${TARGET_GGUF:-$W/models/official-gguf/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf}
