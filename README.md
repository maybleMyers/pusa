<p align="center">
    <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/f867c49d9570b88e7bbce6e25583a0ad2417cdf7/icon.png" width="70"/>
</p>

# Pusa: Thousands Timesteps Video Diffusion Model
[![ModelHub](https://img.shields.io/badge/⚡-Model%20Hub-FFD700?logo=huggingface)](https://huggingface.co/RaphaelLiu/Pusa-V0.5) [![Training](https://img.shields.io/badge/Codebase-Training%20Tools-32CD32?logo=git)](https://github.com/Yaofang-Liu/Mochi-Full-Finetuner)  [![DataRepo](https://img.shields.io/badge/📁-Dataset%20Repo-6495ED?logo=huggingface)](https://huggingface.co/datasets/RaphaelLiu/PusaV0.5_Training) 
[![Paper](https://img.shields.io/badge/📜-FVDM%20Paper-B31B1B?logo=arxiv)](https://arxiv.org/abs/2410.03160) [![Twitter](https://img.shields.io/badge/🐦-Follow%20on%20Twitter-1DA1F2?logo=twitter)](https://x.com/stephenajason)

<p align="center">
    <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/8e507887f27f31f011ca2ecf318d99fd3633116f/assets/demo0.gif" width="800" autoplay loop muted/>
    <br>
    <em>Pusa: Image(s)-to-Video examples</em>
</p>

## Table of Contents
- [Overview](#overview)
- [Method Overview](#method-overview)
- [Key Features](#-key-features)
- [Unique Architecture](#-unique-architecture)
- [Changelog](#changelog)
- [Installation and Usage](#installation-and-usage)
- [Training](#training)
- [Limitations](#limitations)
- [Current Status and Roadmap](#current-status-and-roadmap)
- [Related Work](#related-work)
- [Citation](#citation)

## Overview

Pusa introduces a paradigm shift in video diffusion modeling through frame-level noise control (thus it has thousands of timesteps, rather than one thousand of timesteps), departing from conventional approaches. This shift was first presented in our [FVDM](https://arxiv.org/abs/2410.03160) paper. Leveraging this architecture, Pusa seamlessly supports diverse video generation tasks (Text/Image/Video-to-Video) while maintaining exceptional motion fidelity and prompt adherence with our refined base model adaptations. Pusa-V0.5 represents an early preview based on [Mochi1-Preview](https://huggingface.co/genmo/mochi-1-preview). We are open-sourcing this work to foster community collaboration, enhance methodologies, and expand capabilities.

## Method Overview

<p align="center">
    <img src="https://github.com/Yaofang-Liu/Pusa-VidGen/blob/8e507887f27f31f011ca2ecf318d99fd3633116f/assets/methods_overview.gif" width="800" autoplay loop muted/>
    <br>
    <em>Pusa-VidGen's novel frame-level noise architecture with vectorized timesteps</em>
</p>


## ✨ Key Features

- **Comprehensive Multi-task Support**:
  - Text-to-Video generation
  - Image-to-Video transformation
  - Frame interpolation
  - Video transitions
  - Seamless looping
  - Extended video generation
  - And more...

- **Unprecedented Efficiency**:
  - Trained with only 0.1k H800 GPU hours
  - Total training cost: $0.1k
  - Hardware: 16 H800 GPUs
  - Configuration: Batch size 32, 500 training iterations, 1e-5 learning rate
  - *Note: Efficiency can be further improved with single-node training and advanced parallelism techniques. Collaborations welcome!*

- **Complete Open-Source Release**:
  - Full codebase
  - Detailed architecture specifications
  - Comprehensive training methodology

## 🔍 Unique Architecture

- **Novel Diffusion Paradigm**: Implements frame-level noise control with vectorized timesteps, originally introduced in the [FVDM paper](https://arxiv.org/abs/2410.03160), enabling unprecedented flexibility and scalability.

- **Non-destructive Modification**: Our adaptations to the base model preserve its original Text-to-Video generation capabilities. After this adaptation, we only need a slight fine-tuning.

- **Universal Applicability**: The methodology can be readily applied to other leading video diffusion models including Hunyuan Video, Wan2.1, and others. *Collaborations enthusiastically welcomed!*


## Changelog

**v0.5 (March 28, 2025)**
- Initial public release
- Released model weights and basic inference code
- Support for Text-to-Video and Image-to-Video generation

**v0.5 (April 10, 2025)**
- Released our training codes and details [here](https://github.com/Yaofang-Liu/Mochi-Full-Finetuner)
- Support multi-nodes/single-node full finetuning code for both Pusa and Mochi
- Released our training dataset [dataset](https://huggingface.co/datasets/RaphaelLiu/PusaV0.5_Training)

  
## Installation and Usage

### Installation 

You may install using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/genmoai/models
cd models 
pip install uv
uv venv .venv
source .venv/bin/activate
uv pip install setuptools
uv pip install -e . --no-build-isolation
```

If you want to install flash attention, you can use:
```
uv pip install -e .[flash] --no-build-isolation
```

### Download Weights

**Option 1**: Use the Hugging Face CLI:
```bash
pip install huggingface_hub
huggingface-cli download RaphaelLiu/Pusa-V0.5 --local-dir <path_to_downloaded_directory>
```

**Option 2**: Download directly from [Hugging Face](https://huggingface.co/RaphaelLiu/Pusa-V0.5) to your local machine.


### Basic Usage

#### Text-to-Video Generation

```bash
python ./demos/cli_test_ti2v_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --prompt "A man is playing basketball" \
  --num_steps 30
```

#### Image-to-Video Generation

```bash
python ./demos/cli_test_ti2v_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --prompt "The camera remains still, the boy waves the baseball bat and knocks the baseball away." \
  --image_dir "/path/to/input/image.jpg" \
  --cond_position 1 \
  --num_steps 30
```

#### Processing A Group of Images
```bash
python ./demos/cli_test_ti2v_release.py \
  --model_dir "/path/to/Pusa-V0.5" \
  --dit_path "/path/to/Pusa-V0.5/pusa_v0_dit.safetensors" \
  --image_dir "/path/to/image/directory" \
  --prompt_dir "/path/to/prompt/directory" \
  --cond_position 1 \
  --num_steps 30
```

For group processing, each image should have a corresponding text file with the same name in the prompt directory.

#### Using the Provided Shell Script
We also provide a shell script for convenience:

```bash
# Edit cli_test_ti2v_release.sh to set your paths
# Then run:
bash ./demos/cli_test_ti2v_release.sh
```

## Training

Please find our Pusa training code and details here [here](https://github.com/Yaofang-Liu/Mochi-Full-Finetuner), which also support training for original model.

## Limitations

Pusa currently has several known limitations:
- The base Mochi model generates videos at low resolution (480p)
- We anticipate significant quality improvements when applying our methodology to more advanced models like Wan2.1
- We welcome community contributions to enhance model performance and extend its capabilities

### Currently Available
- ✅ Model weights
- ✅ Inference code for Text-to-Video generation
- ✅ Inference code for Image-to-Video generation
- ✅ Basic documentation
- ✅ Training code and details
- ✅ Model full fine-tuning guide for both Pusa and Mochi
- ✅ Training dataset
  
### TODO List
- 🔄 Inference scripts for start & end frames, video transition, video extension, and more ...
- 🔄 Release more advanced versions with SOTA models like Wan 2.1 and Hunyuan Video
- 🔄 Release Paper
- 🔄 ....

## Related Work

- [FVDM](https://arxiv.org/abs/2410.03160): Introduces the groundbreaking frame-level noise control with vectorized timestep approach that inspired Pusa.
- [Mochi](https://huggingface.co/genmo/mochi-1-preview): Our foundation model, recognized as a leading open-source video generation system on the Artificial Analysis Leaderboard.

## BibTeX
If you use this work in your project, please cite the following references.
```
@misc{Liu2025pusa,
  title={Pusa: Thousands Timesteps Video Diffusion Model},
  author={Yaofang Liu and Rui Liu},
  year={2025},
  url={https://github.com/Yaofang-Liu/Pusa-VidGen},
}
```

```
@article{liu2024redefining,
  title={Redefining Temporal Modeling in Video Diffusion: The Vectorized Timestep Approach},
  author={Liu, Yaofang and Ren, Yumeng and Cun, Xiaodong and Artola, Aitor and Liu, Yang and Zeng, Tieyong and Chan, Raymond H and Morel, Jean-michel},
  journal={arXiv preprint arXiv:2410.03160},
  year={2024}
}
```


