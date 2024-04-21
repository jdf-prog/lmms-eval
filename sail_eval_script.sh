sailctl job create lmmseval -g 8 -p low -f ~/sailctl_high_shm_config.yaml --image asia-docker.pkg.dev/sail-tpu-02/images/language/miqaenv:latest --debug --args


export WANDB__SERVICE_WAIT=300
export HUGGINGFACE_TOKEN=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
export HF_HOME="/home/aiops/jiangdf/.cache/huggingface"
export WANDB_API_KEY="fe326ce9780c4e20aead0e848c27ed537a470dcf"
source /home/aiops/jiangdf/miniconda3/bin/activate lmmseval
cd /home/aiops/jiangdf/Workspace/lmms-eval
accelerate launch --num_processes=8 -m lmms_eval --config mllava_eval_bakllava_1.5_7b_v2.yaml