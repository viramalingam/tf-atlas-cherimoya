# tf-atlas cherimoya modeling image (AnVIL/Terra WDL tasks).
# Pins match the kali `cherimoya` conda env that ran the human
# bpnet-arch-benchmark cherimoya arms (torch 2.10.0+cu126, triton 3.6.0,
# numpy 2.0.1, tangermeme 1.0.3, bpnet-lite 1.0.0).
#
# Build (from anvil/modeling/docker):
#   docker build -t vivekramalingam/tf-atlas:gcp-cherimoya_v0.2.0-g2 .
FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

LABEL purpose="tf-atlas cherimoya modeling (peak-wise teachers, distillation) for AnVIL WDLs" \
      cherimoya_src="github.com/jmschrei/cherimoya v0.2.0 (5390ce0) + cherimoya_v0.2.0_g2.patch" \
      pins="python3.12 torch2.10.0+cu126 triton3.6.0 numpy2.0.1 tangermeme1.0.3 bpnet-lite1.0.0"

# the base image ships the system python 3.12 (PEP 668 marker); install into it
ENV DEBIAN_FRONTEND=noninteractive PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HDF5_USE_FILE_LOCKING=FALSE TORCHDYNAMO_DISABLE=1

# git/jq: the WDL tasks clone the pipeline repo and parse splits json with jq
# (as in tf-atlas-pipeline). gcc: triton JIT-compiles its launcher at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git jq wget ca-certificates build-essential \
        zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt

# cherimoya v0.2.0 + the g2 patch (validation_loci for peak-wise folds,
# optional fixed count weight / val-loss selection; defaults unchanged)
COPY cherimoya_v0.2.0_g2.patch /opt/cherimoya_v0.2.0_g2.patch
RUN git clone https://github.com/jmschrei/cherimoya /opt/cherimoya_src \
    && cd /opt/cherimoya_src && git checkout 5390ce0 \
    && git apply /opt/cherimoya_v0.2.0_g2.patch \
    && pip install --no-cache-dir --no-deps . \
    && python -c "import cherimoya_cli, tangermeme, bpnetlite, pyBigWig, pyfaidx, plotnine, sklearn, torch, triton; print('imports ok', torch.__version__, triton.__version__)"

WORKDIR /cromwell_root
