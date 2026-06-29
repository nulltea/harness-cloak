# ROCm runtime for AMD Strix Halo (gfx1151).
#
# Thin layer on AMD's official rocm/pytorch image — the only pre-built
# image whose PyTorch HIP layer is compiled with gfx1151 in the target
# list (no HSA_OVERRIDE_GFX_VERSION needed). This is the SAME base the
# AloePri ima-trainer uses; we reuse its cached layers rather than
# duplicating the ROCm runtime or pip-installing a torch wheel (a
# separately-installed ROCm wheel lacks gfx1151 kernels and dies with
# hipErrorInvalidDeviceFunction — see private-rag
# evals/aloepri-attacks/m2_7/Dockerfile.ima-trainer).
#
# We add ONLY the small pure-python deps needs; torch is inherited.
# Build via scripts/run_in_rocm.sh (auto-builds if missing).

FROM rocm/pytorch:rocm7.2.3_ubuntu24.04_py3.12_pytorch_release_2.10.0

# Capture stack (transformers>=4.51 for Qwen3) + the measure/attack deps.
# NOT torch — inherited from the base with gfx1151 kernels.
RUN pip install --no-cache-dir \
        "transformers>=4.51" \
        tokenizers \
        safetensors \
        scipy \
        scikit-learn \
        nnsight

# Gemma-scope SAE probes via sae-lens. Installed so it does NOT clobber the
# inherited ROCm torch/torchvision: sae-lens + transformer-lens pull a PyPI
# torchvision that exact-pins (and thus downgrades) torch, so install THOSE
# TWO with --no-deps and supply their (pure-python, non-torch-pinning) deps
# separately. Recipe validated in scripts/spikes/_resolve_sae_deps.sh.
RUN pip install --no-cache-dir --no-deps sae-lens transformer-lens \
 && pip install --no-cache-dir \
        datasets accelerate pandas rich sentencepiece fancy-einsum \
        better-abc beartype jaxtyping typeguard pydantic simple-parsing \
        einops wandb nltk plotly plotly-express python-dotenv tenacity \
        babe transformers-stream-generator \
 && python3 -c "import torch; from sae_lens import SAE; assert torch.__version__.find('rocm')>0, torch.__version__; print('sae_lens OK on', torch.__version__)"

WORKDIR /work
