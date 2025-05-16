#!/bin/sh

#####
# Modify these lines
gpu_id=2													# Visible GPUs
n_gpu=1														# Number of GPU used, currently only support 1
checkpoint_dir='/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-04-27(01:28:46)'


#'/home/sriyar/neuroheed/log_KUL_eeg_neuroheed_2spk/checkpoints/log_KUL_eeg_neuroheed_2spk'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-09(20:53:58)'



#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-04-27(01:28:46)'


#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-09(20:53:58)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-07(12:00:41)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-01(16:12:06)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-03(10:10:10)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-04-27(01:28:46)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-07(12:00:41)'


#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-04-27(01:28:46)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-07(12:00:41)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-07(11:36:19)'


#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-03(10:10:10)'


#'/home/sriyar/neuroheed/log_KUL_eeg_neuroheed_2spk/checkpoints/log_KUL_eeg_neuroheed_2spk'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-03(10:10:10)'


#'/home/sriyar/neuroheed/log_KUL_eeg_neuroheed_2spk/checkpoints/log_KUL_eeg_neuroheed_2spk'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-03(10:10:10)'
#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-04-27(01:28:46)'

train_from_last_checkpoint=1
config_pth=${checkpoint_dir}/config.yaml

# call evaluation
export PYTHONWARNINGS="ignore"
CUDA_VISIBLE_DEVICES="$gpu_id" \
python -W ignore \
-m torch.distributed.launch \
--nproc_per_node=$n_gpu \
--master_port=$(date '+88%S') \
train.py \
--evaluate_only 1 \
--config $config_pth \
--checkpoint_dir $checkpoint_dir \
--train_from_last_checkpoint $train_from_last_checkpoint \
--evaluate_only 1 \
>>${checkpoint_dir}/evaluation.txt 2>&1


