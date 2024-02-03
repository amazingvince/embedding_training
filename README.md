# embedding_training

Lots of code lifted from https://github.com/staoxiao/RetroMAE for pretraining. Updating and combining into modern huggingface stuff. Added fused attention.


# Set up
```bash
pip install -r requirments.txt
wandb login
huggingface-cli login
chmod +x run_pretraining.sh
accelerate config
```

# runpod
```bash
lscpu | head
export HF_HOME=/workspace/local-HF-cache
sleep 3
apt update && apt install zip unzip tmux nano p7zip-full git -y && apt upgrade -y
```
