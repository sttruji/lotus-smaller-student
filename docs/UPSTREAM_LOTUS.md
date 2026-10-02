# Retained upstream LOTUS workflow

This guide describes the original LOTUS method configs retained in this fork.
For the frozen 3B → recurrent 1B experiment, start with the
[current README](../README.md) and [research guide](../RESEARCH_SCAFFOLD.md).

Upstream: [yingfan-bot/lotus](https://github.com/yingfan-bot/lotus), base commit
eb77e2f7909c5006f58ff0ad7cd6629b942caa9e.
Paper: [Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers](https://arxiv.org/abs/2606.31779).
The [project page](https://yingfan-bot.github.io/lotus/) and
[HF collection](https://huggingface.co/collections/yingfanbot/looped-padded-6a552f7ef667cb41db2431a3)
describe upstream results. They are not accuracy measurements from the new
cross-size distillation experiment.
The retained [paper reference page](index.html) links to the fork's current
documentation and labels its figures as upstream results.

## Original method

LOTUS initializes from explicit CoT fine-tuning and trains a looped latent
student with direct supervision of gold reasoning tokens. The research fork
uses the same recurrent backbone machinery but replaces that intermediate
supervision with cross-size boundary distillation and auxiliary explicit
student CoT training.

The original args/gsm8k_*.yaml files remain available. They have different
training durations and batch settings from args/research/*.yaml. Use the actual
resolved config; not every retained file matches the paper's 30-epoch setup.

| Retained config | GPU microbatch | GPUs for batch 128 | Epochs |
| --- | ---: | ---: | ---: |
| [GPT-2 CoT](../args/gsm8k_cot_gpt2.yaml) | 64 | 2 | 100 |
| [GPT-2 LOTUS](../args/gsm8k_lotus_gpt2.yaml) | 64 | 2 | 30 |
| [Llama 1B CoT](../args/gsm8k_cot_llama1b.yaml) | 32 | 4 | 100 |
| [Llama 1B LOTUS](../args/gsm8k_lotus_llama1b.yaml) | 32 | 4 | 100 |
| [Llama 3B CoT](../args/gsm8k_cot_llama3b.yaml) | 32 | 4 | 10 |
| [Llama 3B LOTUS](../args/gsm8k_lotus_llama3b.yaml) | 16 | 8 | 30 |

These files do not set gradient accumulation; the current training entry point
defaults it to one. Counts in the table therefore use physical batch × GPUs.
The research batch-128 options are documented separately in
[training batches](TRAINING_BATCHES.md).

## Environment and data

Use the prepared CUDA environment and pinned preprocessing described in
[the research guide](../RESEARCH_SCAFFOLD.md#environment-and-data).
The NGC alternative mounts the repository into a prepared container:

~~~bash
docker run --gpus all --rm -it \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/lotus -w /workspace/lotus \
  nvcr.io/nvidia/pytorch:25.03-py3

pip install -r requirements.txt
bash preprocessing/gsm_icot.bash
~~~

The preprocessing scripts are adapted from
[Coconut](https://github.com/facebookresearch/coconut) and use augmented GSM8K
data from [Internalize_CoT_Step_by_Step](https://github.com/da03/Internalize_CoT_Step_by_Step).
Outputs and data are git-ignored.

## Original launcher

[launch_train.sh](../launch_train.sh) launches one node using these variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| CONFIG | args/gsm8k_cot_llama1b.yaml | YAML passed to scripts/run.py |
| RUN_NAME | empty | Optional run-name override and output directory name |
| NPROC_PER_NODE | 4 | GPU process count; set explicitly for the selected config |
| MASTER_PORT | 29500 | Distributed master port |

The launcher does not infer GPU count from batch size and does not forward
--set overrides. To override YAML keys from the CLI, use the research launcher
with the desired CONFIG and explicit overrides.

The original latent configs initialize from published CoT checkpoints, so an
initialization pilot can start directly at the latent-training recipe:

~~~bash
CONFIG=args/gsm8k_lotus_gpt2.yaml RUN_NAME=gsm-lotus-gpt2 NPROC_PER_NODE=2 \
bash launch_train.sh

CONFIG=args/gsm8k_lotus_llama1b.yaml RUN_NAME=gsm-lotus-llama1b NPROC_PER_NODE=4 \
bash launch_train.sh

CONFIG=args/gsm8k_lotus_llama3b.yaml RUN_NAME=gsm-lotus-llama3b NPROC_PER_NODE=8 \
bash launch_train.sh
~~~

To retrain the CoT initializations, choose the corresponding CoT config with the
GPU count shown above. Point the latent config's load_model_path at the selected
CoT checkpoint. Set teacher_model_path when a chosen CODI objective requires it.

## Published upstream checkpoints

| HF repository | Purpose |
| --- | --- |
| [yingfanbot/gsm-lotus-llama3b](https://huggingface.co/yingfanbot/gsm-lotus-llama3b) | Upstream LOTUS 3B |
| [yingfanbot/gsm-lotus-llama3b-codi](https://huggingface.co/yingfanbot/gsm-lotus-llama3b-codi) | Upstream LOTUS + CODI 3B |
| [yingfanbot/gsm-cot-gpt2](https://huggingface.co/yingfanbot/gsm-cot-gpt2) | Explicit CoT GPT-2 initialization |
| [yingfanbot/gsm-cot-llama1b](https://huggingface.co/yingfanbot/gsm-cot-llama1b) | Explicit CoT 1B initialization |
| [yingfanbot/gsm-cot-llama3b](https://huggingface.co/yingfanbot/gsm-cot-llama3b) | Explicit CoT 3B initialization |

Evaluation now uses the fork's strict loading and accounting behavior. Consult
[the evaluation guide](EVALUATION.md) for raw checkpoints, local exports, remote
HF settings and output reports. For legacy GPT-2 evaluation, pass --fp32 and the
trained c_thought value, usually 13; the Llama latent configs use 25.

## Citation and license

~~~bibtex
@article{fan2026bridging,
  title={Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers},
  author={Fan, Ying and Svete, Anej and Lee, Kangwook},
  journal={arXiv preprint arXiv:2606.31779},
  year={2026}
}
~~~

The training, evaluation and preprocessing code is adapted from Coconut.
The repository is released under the [MIT License](../LICENSE).
