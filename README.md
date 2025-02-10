# Llambada-v0 🐑 🎵

❗Note: This repository is in-progress for the final debugging, please wait until we release the full version 

Welcome to the official implementation of Llambada version 0 repository! This project provides the tools and resources to use the Llambada model, an advanced system for music generation.

- Paper: [Arxiv](https://arxiv.org/pdf/2411.01661)

- Demo page: [Llambada demo](https://songgen-ai.github.io/llambada-demo/)

This model is trained on totally 4.4k music hours dataset with 2xA100 GPUS. The training cost for this model is about 720 USD in 5 days for 2 stages: the semantic stage and the coarse stage.

⭐ Hopefully, we want open the a.i for everyone, so all of the source code of the model, the training script, and the hyperparameters will be released :)

Please note: At this time, the repository includes only the inference code and pre-trained model checkpoint. Training scripts will be added in a future update.

### ☑️ Release checklist

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

### Pretrained setup

After downloading the checkpoints, you need to create the ```ckpts/``` folder, then you move all files to the ```ckpts/``` folder. 

Regarding the tokenizer ```bpe_simple_vocab_16e6.txt.gz```, you need to copy that file to the ```/workspace/llambada_test/LLambada/models/base/tokenizers/laion_clap/clap_module``` for the setup.

## Running Inference

To run the inference, please run via the python file below:

``` bash
python infer.py
```

Create stunning music compositions with Llambada effortlessly!

Moreover, you can change the gpu for the inference via add this config to the front ```CUDA_VISIBLE_DEVICES=<your device id> ```

Total inference time for 10 seconds singing accompaniment is about 1 minute and 30 seconds on 1xH100.

### Contact

If you have any further questions or having new ideas for the model features, you can raise in the issue or you can contact us in songgen.ai and we can have support in our ability!

### License 

```
Copyright 2025 Songgen.ai

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
