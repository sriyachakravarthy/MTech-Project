<div align="center">
<img src="https://github.com/user-attachments/assets/a4ccbc60-5248-4dca-8cec-09a6385c6d0f" width="768" height="192">
</div>

<strong>ClearerVoice-Studio</strong> is an open-source, AI-powered speech processing toolkit designed for researchers, developers, and end-users. It provides capabilities of speech enhancement, speech separation, speech super-resolution, target speaker extraction, and more. The toolkit provides state-of-the-art pre-trained models, along with training and inference scripts, all accessible from this repository.
 
#### 👉🏻[HuggingFace Demo](https://huggingface.co/spaces/alibabasglab/ClearVoice)👈🏻  | 👉🏻[ModelScope Demo](https://modelscope.cn/studios/iic/ClearerVoice-Studio) ｜ 👉🏻[SpeechScore Demo](https://huggingface.co/spaces/alibabasglab/SpeechScore)👈🏻

---
![GitHub Repo stars](https://img.shields.io/github/stars/modelscope/ClearerVoice-Studio) Please leave your ⭐ on our GitHub to support this community project！

记得点击右上角的星星⭐来支持我们一下，您的支持是我们更新模型的最大动力！

## News :fire:
- Upcoming: More tasks will be added to ClearVoice.
- [2025.1] ClearVoice demo is ready for try on both [HuggingFace](https://huggingface.co/spaces/alibabasglab/ClearVoice) and [ModelScope](https://modelscope.cn/studios/iic/ClearerVoice-Studio). However, HuggingFace has limited GPU usage, and ModelScope has more GPU usage quota.
- [2025.1] ClearVoice now offers **speech super-resolution**, also known as bandwidth extension. This feature improves the perceptual quality of speech by converting low-resolution audio (with an effective sampling rate of at least 16,000 Hz) into high-resolution audio with a sampling rate of 48,000 Hz. A full upscaled **LJSpeech-1.1-48kHz dataset** can be downloaded from [HuggingFace](https://huggingface.co/datasets/alibabasglab/LJSpeech-1.1-48kHz) and [ModelScope](https://modelscope.cn/datasets/iic/LJSpeech-1.1-48kHz).
- [2025.1] ClearVoice now supports more audio formats including **"wav", "aac", "ac3", "aiff", "flac", "m4a", "mp3", "ogg", "opus", "wma", "webm"**, etc. It also supports both mono and stereo channels with 16-bit or 32-bit precisions. A latest version of [ffmpeg](https://github.com/FFmpeg/FFmpeg) is required for audio codecs.  
- [2024.12] Upload pre-trained models on ModelScope. User now can download the models from either [ModelScope](https://www.modelscope.cn/models/iic/ClearerVoice-Studio/summary) or [Huggingface](https://huggingface.co/alibabasglab)  
- [2024.11] Our FRCRN speech denoiser has been used over **3.0 million** times on [ModelScope](https://modelscope.cn/models/iic/speech_frcrn_ans_cirm_16k)
- [2024.11] Our MossFormer speech separator has been used over **2.5 million** times on [ModelScope](https://modelscope.cn/models/iic/speech_mossformer_separation_temporal_8k)
- [2024.11] Release of this repository

### 🌟 Why Choose ClearerVoice-Studio?

- **Pre-Trained Models:** Includes cutting-edge pre-trained models, fine-tuned on extensive, high-quality datasets. No need to start from scratch!
- **Ease of Use:** Designed for seamless integration with your projects, offering a simple yet flexible interface for inference and training.
- **Comprehensive Features:** Combines advanced algorithms for multiple speech processing tasks in one platform.
- **Community-Driven:** Built for researchers, developers, and enthusiasts to collaborate and innovate together.

## Contents of this repository
This repository is organized into three main components: **[ClearVoice](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice)**, **[Train](https://github.com/modelscope/ClearerVoice-Studio/tree/main/train)**, and **[SpeechScore](https://github.com/modelscope/ClearerVoice-Studio/tree/main/speechscore)**.

### 1. **ClearVoice [[Readme](https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/README.md)][[文档](https://github.com/modelscope/ClearerVoice-Studio/blob/main/clearvoice/README.md)]**  
ClearVoice offers a user-friendly  solution for speech processing tasks such as speech denoising, separation, super-resolution, audio-visual target speaker extraction, and more. It is designed as a unified inference platform leveraged pre-trained models (e.g., [FRCRN](https://arxiv.org/abs/2206.07293), [MossFormer](https://arxiv.org/abs/2302.11824)), all trained on extensive datasets. If you're looking for a tool to improve speech quality, ClearVoice is the perfect choice. Simply click on [`ClearVoice`](https://github.com/modelscope/ClearerVoice-Studio/tree/main/clearvoice) and follow our detailed instructions to get started.

### 2. **Train**  
For advanced researchers and developers, we provide model finetune and training scripts for all the tasks offerred in ClearVoice and more:

- **Task 1: [Speech enhancement](train/speech_enhancement)** (16kHz & 48kHz)
- **Task 2: [Speech separation](train/speech_separation)** (8kHz & 16kHz)
- **Task 2: [Speech super-resolution](https://github.com/modelscope/ClearerVoice-Studio/tree/main/train/speech_super_resolution)** (48kHz) (comming soon)
- **Task 4: [Target speaker extraction](train/target_speaker_extraction)** 
  - **Sub-Task 1: Audio-only Speaker Extraction Conditioned on a Reference Speech** (8kHz)
  - **Sub-Task 2: Audio-visual Speaker Extraction Conditioned on Face (Lip) Recording** (16kHz)
  - **Sub-Task 3: Audio-visual Speaker Extraction Conditioned on Body Gestures** (16kHz)
  - **Sub-Task 4: Neuro-steered Speaker Extraction Conditioned on EEG Signals** (16kHz)

Contributors are welcomed to include more model architectures and tasks!

### 3. **SpeechScore [[Readme](https://github.com/modelscope/ClearerVoice-Studio/blob/main/speechscore/README.md)][[文档](https://github.com/modelscope/ClearerVoice-Studio/blob/main/speechscore/README.md)]**  
<a href="https://github.com/modelscope/ClearerVoice-Studio/tree/main/speechscore">`SpeechScore`<a/> is a speech quality assessment toolkit. We include it here to evaluate different model performance. SpeechScore includes many popular speech metrics:

- Signal-to-Noise Ratio (SNR)
- Perceptual Evaluation of Speech Quality (PESQ)
- Short-Time Objective Intelligibility (STOI)
- Deep Noise Suppression Mean Opinion Score (DNSMOS)
- Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
- and many more quality benchmarks  
  
## Contact
If you have any comments or questions about ClearerVoice-Studio, feel free to raise an issue in this repository or contact us directly at:
- email: {shengkui.zhao, zexu.pan}@alibaba-inc.com

Alternatively, welcome to join our DingTalk and WeChat groups to share and discuss algorithms, technology, and user experience feedback. You may scan the following QR codes to join our official chat groups accordingly. 

<p align="center">
  <table>
    <tr>
      <td style="text-align:center;">
        <a href="./asset/QR.jpg"><img alt="ClearVoice in DingTalk" src="https://img.shields.io/badge/ClearVoice-DingTalk-d9d9d9"></a>
      </td>
      <td style="text-align:center;">
        <a href="./asset/QR.jpg"><img alt="ClearVoice in WeChat" src="https://img.shields.io/badge/ClearVoice-WeChat-d9d9d9"></a>
      </td>
    </tr>
    <tr>
       <td style="text-align:center;">
      <img alt="Light" src="./asset/dingtalk.png" width="68%" />
      <td style="text-align:center;">
      <img alt="Light" src="./asset/qr.png" width="23%" />
      </td>
    </tr>
  </table>
</p>
 
## Friend Links
Checkout some awesome Github repositories from Speech Lab of Institute for Intelligent Computing, Alibaba Group.

<p align="center">
<a href="https://github.com/FunAudioLLM/InspireMusic" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Repo | Space-InspireMusic?labelColor=&label=InspireMusic&color=green"></a>
<a href="https://github.com/modelscope/FunASR" target="_blank">
        <img alt="Github" src="https://img.shields.io/badge/Repo | Space-FunASR?labelColor=&label=FunASR&color=green"></a>
<a href="https://github.com/FunAudioLLM" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Repo | Space-FunAudioLLM?labelColor=&label=FunAudioLLM&color=green"></a>
<a href="https://github.com/modelscope/3D-Speaker" target="_blank">
        <img alt="Demo" src="https://img.shields.io/badge/Repo | Space-3DSpeaker?labelColor=&label=3D-Speaker&color=green"></a>
</p>


## Acknowledge
ClearerVoice-Studio contains third-party components and code modified from some open-source repos, including: <br>
[Speechbrain](https://github.com/speechbrain/speechbrain), [ESPnet](https://github.com/espnet), [TalkNet-ASD
](https://github.com/TaoRuijie/TalkNet-ASD)
