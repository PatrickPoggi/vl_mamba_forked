# Forked from [vl_mamba](https://github.com/gpantaz/vl_mamba.git)

This is a repository meant as a reproducibility analysis for the paper [Shaking Up VLMs: Comparing Transformers and Structured State Space Models for Vision & Language Modeling (Pantazopoulos et al., EMNLP 2024)](https://aclanthology.org/2024.emnlp-main.793/).


## Setting up the workspace
Even though the original repo (whose README.md is appened below) is quite clear and the repo itself was quite well maintained, a few adjustments have been necessary. In short:

- Reverting to an older commit to solve broken inter-dependencies (files needed but missing)
- Struggling a bit with the package dependencies
- Code was not compatible with my setup: NVIDIA 2080-Ti (different architecture)

Therefore, aiming at making the setup process as easy as possible, I provide my version of the repo, which I obtained by forking from an older commit and then manually fixing the code.

The only thing that's left is:

```bash
# from main directory (vl_mamba)
chmod +x prepare_dirs.sh
./prepare_dirs.sh

conda env create -f vl_mamba_conda_environment.yml
```

As `prepare_dirs.sh` automatically replicates the directory structure as in my own setup and `vl_mamba_conda_environment.yml` is the YML file describing my conda environment 

NOTE: If you don't like the name of the environment because it is too long, then:

```bash
# from main directory (vl_mamba)
mv vl_mamba_conda_environment.yml name_you_like.yml
conda env create -f name_you_like.yml

```

## Reproducing the experiments I have reproduced

Due to resource constraints and technical problems (i.e., mambavl_790m was unable to be loaded as, apparently, the file was broken) I wasn't able to reproduce all the results. However, I reproduced a good deal of them, selecting one dataset per task and comparing both models for the 1.4B parameters variant.

In order to replicate that:

```bash
# from main directory (vl_mamba)
chmod +x scripts/reproducibility_evaluation.sh
./reproducibility_evaluation.sh
```


## Citation of the original work
**Pantazopoulos, Georgios, Nikandrou, Malvina, Suglia, Alessandro, Lemon, Oliver, and Eshghi, Arash.**  
*Shaking Up VLMs: Comparing Transformers and Structured State Space Models for Vision & Language Modeling*.  
Proceedings of EMNLP 2024. [ACL Anthology](https://aclanthology.org/2024.emnlp-main.793/)  
DOI: [10.18653/v1/2024.emnlp-main.793](https://doi.org/10.18653/v1/2024.emnlp-main.793)

And link to the original repo: [link](https://github.com/gpantaz/vl_mamba)


---
### Original README below

---
---

# Shaking Up VLMs: Comparing Transformers and Structured State Space Models for Vision \& Language Modeling (EMNLP 2024)
[[Paper](https://arxiv.org/pdf/2409.05395)][[Model Checkpoints](#model-checkpoints)][[Data](#data)][[Training](#training)]

## Requirements

## Model Checkpoints 🤗


### Pretrained Checkpoints


| Model                                                                     |
| :------------------------------------------------------------------------ |
| [Pythia-VL-1B](https://huggingface.co/gpantaz/pretrained_pythiavl_1b)     |
| [Mamba-VL-790M](https://huggingface.co/gpantaz/pretrained_mambavl_790m)   |
| [Pythia-VL-1.4B](https://huggingface.co/gpantaz/pretrained_pythiavl_1.4b) |
| [Mamba-VL-1.4B](https://huggingface.co/gpantaz/pretrained_mambavl_1.4b)   |
| [Pythia-VL-2.8B](https://huggingface.co/gpantaz/pretrained_pythiavl_2.8b) |
| [Mamba-VL-2.8B](https://huggingface.co/gpantaz/pretrained_mambavl_2.8b)   |

### Instruction-tuned Checkpoints

| Model                                                                    |  COCO  | NoCaps | VQAv2 |  GQA  | V7W (test-T) |  VSR  | POPE  | RefCOCO (testA) | RefCOCO (testB) | RefCOCO+ (testA) | RefCOCO+ (testB) | RefCOCOg | V7W (test-P) | TextCaps | TextVQA | AI2D  |
| :----------------------------------------------------------------------- | :----: | :----: | :---: | :---: | :----------: | :---: | :---: | :-------------: | :-------------: | :--------------: | :--------------: | :------: | :----------: | :------: | :-----: | :---: |
|                                                                          |        |        |       |       |              |       |       |                 |                 |                  |                  |          |              |          |         |       |
| [Pythia-VL-1B](https://huggingface.co/gpantaz/finetuned_pythiavl_1b)     | 132.89 | 97.61  | 72.26 | 53.79 |    81.96     | 72.43 | 86.77 |      76.00      |      62.48      |      45.36       |      47.44       |  67.58   |    83.78     |  92.73   |  35.22  | 77.62 |
| [Mamba-VL-790M](https://huggingface.co/gpantaz/finetuned_mambavl_790m)   | 133.81 | 99.00  | 71.67 | 54.95 |    81.82     | 75.39 | 86.77 |      67.84      |      56.35      |      57.97       |      41.43       |  59.16   |    74.01     |  94.30   |  40.72  | 79.27 |
| [Pythia-VL-1.4B](https://huggingface.co/gpantaz/finetuned_pythiavl_1.4b) | 134.06 | 100.72 | 73.57 | 57.05 |    83.06     | 77.72 | 86.40 |      82.43      |      68.39      |      72.35       |      55.16       |  72.56   |    86.13     |  94.60   |  37.54  | 79.27 |
| [Mamba-VL-1.4B](https://huggingface.co/gpantaz/finetuned_mambavl_1.4b)   | 134.76 | 100.56 | 74.46 | 58.44 |    83.78     | 80.18 | 85.32 |      76.60      |      63.48      |      68.40       |      52.11       |  68.82   |    80.18     |  98.68   |  41.30  | 80.86 |
| [Pythia-VL-2.8B](https://huggingface.co/gpantaz/finetuned_pythiavl_2.8b) | 134.97 | 101.27 | 75.08 | 59.76 |    84.34     | 80.86 | 86.87 |      85.39      |      70.82      |      75.39       |      58.62       |  76.24   |    86.61     |  99.74   |  39.14  | 81.57 |
| [Mamba-VL-2.8B](https://huggingface.co/gpantaz/finetuned_mambavl_2.8b)   | 135.53 | 102.00 | 76.08 | 60.41 |    85.31     | 81.45 | 87.33 |      79.29      |      64.97      |      71.64       |      53.94       |  71.27   |    82.50     |  100.47  |  42.14  | 83.71 |


## Data

## Training

### Pretraining

### Instruction-tuning

### Training logs

All the logs regarding pretraining / finetuning can be found on [wandb](https://wandb.ai/gpantaz/vl_mamba?nw=nwusergpantaz)
Note that some of the runs were resumed from a previous checkpoint.
