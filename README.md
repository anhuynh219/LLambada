# Llambada SongGen
Welcome to the official implementation of Llambada SongGen repository! This project provides the tools and resources to use the Llambada model, an advanced system for music generation.

Please note: At this time, the repository includes only the inference code and pre-trained model checkpoint. Training scripts will be added in a future update.

```
© 2024 SongGen Team. All rights reserved.
```

### Release checklist

- [x] Model code
- [x] Inference script
- [ ] Checkpoint
- [ ] Training script
- [ ] Gradio inference
- [ ] Model serving
# 🛠️ Installation
Follow the steps below to set up your Python 3.10 environment using Conda and install the required dependencies.

Step 1: Create the environment
```bash
conda env create -f environment.yml
conda activate llambada
```
Step 2: Install dependencies
Install ffmpeg (for ubuntu, the script is here) and the dependencies.
``` bash
apt update && apt install ffmpeg
pip install -r requirements.txt
```
# 🚅 Training (Coming Soon)
Instructions and scripts for training will be provided in a future release.

# 🖥️ Inference
Utilize the pre-trained Llambada model to generate music easily.

## Checkpoint download

All of the checkpoints for semantic stage and the coarse stage can be downloaded in the [HuggingFace of SongGen](https://huggingface.co/songgen/Llambada)

## Running Inference
``` bash
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python infer.py
```
Create stunning music compositions with Llambada effortlessly!

