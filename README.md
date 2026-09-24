<h1 align="center">InterQ-RL</h1>

<p align="center"><b>Interaction-Aware Correction of Execution Mismatch in Quantized Agentic RL</b></p>

Official code for InterQ-RL, built on [verl](https://github.com/verl-project/verl) (commit `11c94ad`). It includes InterQ-RL, the quantized-RL baselines in the paper, and the ALFWorld, WebShop, and ScienceWorld environments.

# Repository Structure

```text
InterQ-RL/
├── interq_rl/
│   ├── core_interq.py        # InterQ-RL correction weights
│   ├── ais.py                # AIS weights
│   └── qurl.py               # QuRL adaptive clipping range (ACR)
├── agent_system/environments/
│   ├── alfworld/             # ALFWorld agent loop and data preparation
│   ├── webshop/              # WebShop agent loop and env server setup
│   └── scienceworld/         # ScienceWorld agent loop and data preparation
├── examples/                 # One launcher per setting and method
├── scripts/apply_uaq.py      # QuRL update-aware quantization (UAQ)
└── verl/                     # W8A8-KV8 rollout, W/A/KV QAT learner, and policy losses
```

# Installation

## Install verl

We use verl's official image `verlai/verl:vllm023.dev1`, built from `docker/Dockerfile.stable.vllm` (CUDA 13.0, PyTorch 2.11.0, vLLM 0.23.0, transformers 5.3.0, Megatron-Core 0.16.1). Inside the image, install this repository and the additional dependencies:

```bash
pip install --no-deps -e .
pip install -r requirements-interq.txt
pip install --no-deps megatron-bridge==0.3.1
```

## Prepare Environments

### 1. ALFWorld

Download the official ALFWorld data (stored in `~/.cache/alfworld/`), then build the `train`, `valid_seen`, and `valid_unseen` splits and copy the game files into `data/alfworld`:

```bash
alfworld-download
python agent_system/environments/alfworld/data_preprocess/process_alfworld.py \
    --input_dir ~/.cache/alfworld/json_2.1.1 --output_dir data/alfworld
```

### 2. WebShop

WebShop runs as an HTTP server on the training node (`127.0.0.1:4111`), using the WebShop implementation of [Agent-R1](https://github.com/0russwest0/Agent-R1). Download its full-catalog data (products, Lucene index, and `train/test.parquet`) from the [Agent-R1-data release](https://www.modelscope.cn/datasets/Melmaphother/Agent-R1-data):

```bash
pip install modelscope
modelscope download --dataset Melmaphother/Agent-R1-data --include "webshop_full/*" --local_dir data
```

Install the server in its own virtual environment, then start it and keep it running during training:

```bash
bash agent_system/environments/webshop/env_setup.sh
WEBSHOP_DATA_DIR=data/webshop_full bash agent_system/environments/webshop/start_env_server.sh
```

### 3. ScienceWorld

ScienceWorld runs on Java. Install Java 21, then build the official train, dev, and test splits and the 259-variation dev monitor:

```bash
apt-get install -y openjdk-21-jre-headless
python agent_system/environments/scienceworld/data_preprocess/process_scienceworld.py --output-dir data/scienceworld
```

# Run Examples

## Training

Each setting provides `run_bf16.sh`, `run_flash_rl.sh`, `run_ais.sh`, `run_qurl.sh`, `run_qat.sh`, and `run_interq_rl.sh`, for example:

```bash
bash examples/alfworld/run_interq_rl.sh
```

| Setting | Model | Hardware |
|---|---|---|
| `alfworld` | Qwen3-4B-Instruct-2507 | 1 × 8 GPUs, FSDP |
| `webshop` | Qwen3-4B-Instruct-2507 | 1 × 8 GPUs, FSDP |
| `scienceworld_8b` | Qwen3-8B | 1 × 8 GPUs, Megatron |
| `scienceworld_30b_a3b` | Qwen3-30B-A3B-Instruct-2507 | 2 × 8 GPUs, Megatron |

Model and data paths can be set with `MODEL_PATH`, `ALFWORLD_DATA_ROOT`, `WEBSHOP_DATA_DIR`, and `SCIENCEWORLD_DATA_DIR`.

QuRL starts from a UAQ-rescaled copy of the base model:

```bash
python scripts/apply_uaq.py --src <base model dir> --dst <uaq model dir>
MODEL_PATH=<uaq model dir> bash examples/alfworld/run_qurl.sh
```

`scienceworld_30b_a3b` runs on a two-node Ray cluster; export the variables set at the top of its `common.sh` on both nodes before starting Ray.

## Evaluation

The launchers evaluate every 10 training steps on `valid_seen` and `valid_unseen` (ALFWorld), `test` (WebShop), and `dev_monitor` (ScienceWorld). To evaluate a saved checkpoint on the ScienceWorld `test` split, append to its launcher:

```bash
trainer.val_only=true trainer.resume_mode=resume_path trainer.resume_from_path=<checkpoint> \
    data.val_files="['data/scienceworld/test.parquet']"
```

# Acknowledgement

InterQ-RL is built on [verl](https://github.com/verl-project/verl). The ALFWorld and WebShop environments are adapted from [Agent-R1](https://github.com/0russwest0/Agent-R1). The environments are [ALFWorld](https://github.com/alfworld/alfworld), [WebShop](https://github.com/princeton-nlp/WebShop), and [ScienceWorld](https://github.com/allenai/ScienceWorld).

# License

Apache License 2.0.
