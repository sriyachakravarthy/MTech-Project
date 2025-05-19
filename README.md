# Target Speaker Extraction Guided by EEG Cues via Structured State Space Models

In this work, we revisit the [NeuroHeed](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10683957) framework by replacing its original DPRNN-based separator with a Structured State Space Model (S4M).  This substitution aims to explore whether structured state space models can better capture the alignment between neural and acoustic features, especially over short windows ranging from 1 to 10 seconds — a key constraint in practical neuro-steered systems. Our architecture retains NeuroHeed’s encoder-decoder structure but introduces [S4 block](https://arxiv.org/pdf/2305.16932) as a drop-in separator module, potentially enhancing attention decoding robustness and temporal generalization.

## Contribution to the Original Neuroheed [Code](https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/target_speaker_extraction) 

1. Replacing DPRNN with S4 in the [mddel](https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/target_speaker_extraction/models/neuroheed). Included S4 related files in the folder models/neuroheed
2. Adding [InfoNCE loss](https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/target_speaker_extraction/solver.py) in the training objective
3. Adding [Performance metrics](https://github.com/modelscope/ClearerVoice-Studio/blob/main/train/target_speaker_extraction/solver.py) like Inference time and peak memory

# Usage

## Step 1- Clone the repository 
```
git clone https://github.com/sriyachakravarthy/MTech-Project/tree/master
```
## Step 2- Create Conda Environment
```
cd ClearerVoice-Studio/train/target_speaker_extraction/
conda create -n clear_voice_tse python=3.9
conda activate clear_voice_tse
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1  pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Step 3- Download Dataset

Download dataset from [HuggingFace Repository](https://huggingface.co/datasets/alibabasglab/KUL-mix)
This is a modified version of the Auditory Attention Detection Dataset KULeuven. Original data can be downloaded [here](https://zenodo.org/records/4004271).

## Step 4- Train

```
bash train.sh
```

Make sure that the working folder is ClearerVoice-Studio/train/target_speaker_extraction/ 
Modify the Dataset paths in config file- ClearerVoice-Studio/train/target_speaker_extraction/config/config_KUL_eeg_neuroheed_2spk.yaml

## Step 5- Inference and Evaluation

Change the trained model path (checkpoint_dir) in ClearerVoice-Studio/train/target_speaker_extraction/evaluate_only.sh and run

```
bash evaluate_only.sh
```

# Preliminary Results

We trained our model on a system equipped with three NVIDIA GeForce RTX 3090 GPUs, each with 25.75 GB memory, using the NVIDIA DGX platform. The effective batch size used during training is 12.

![image7](https://github.com/user-attachments/assets/77885b8b-5aec-4393-813f-6c229dcf9e54)

![image](https://github.com/user-attachments/assets/fcb5774c-1b43-4cd8-a715-a16638038bd4)
