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
