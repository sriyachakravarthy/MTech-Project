#!/bin/sh

#####
# Modify these lines
gpu_id=0,1,2													# Visible GPUs
n_gpu=3												# Number of GPU used for training
checkpoint_dir=''

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-04-27(01:28:46)'


#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-12(10:40:34)'

#'/home/sriyar/neuroheed/log_KUL_eeg_neuroheed_2spk/checkpoints/log_KUL_eeg_neuroheed_2spk' #'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-12(10:40:34)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-04-27(01:28:46)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-09(15:47:11)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-04-27(01:28:46)'

#'/home/sriyar/neuroheed/log_KUL_eeg_neuroheed_2spk/checkpoints/log_KUL_eeg_neuroheed_2spk'


#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-04-27(01:28:46)'



#'/home/sriyar/neuroheed/log_KUL_eeg_neuroheed_2spk/checkpoints/log_KUL_eeg_neuroheed_2spk'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-03(10:10:10)'

#'/home/sriyar/neuroheed/log_KUL_eeg_neuroheed_2spk/checkpoints/log_KUL_eeg_neuroheed_2spk'


#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-03(10:10:10)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-01(16:12:06)' #'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-03(10:10:10)'

#'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-05-01(16:12:06)' #'/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/checkpoints/log_2025-03-04(11:40:06)' #'/home/sriyar/neuroheed/log_KUL_eeg_neuroheed_2spk/checkpoints/log_KUL_eeg_neuroheed_2spk'											# Leave empty if it's a new training, otherwise provide the name as 'checkpoints/log_...'
config_pth=/home/sriyar/neuroheed/ClearerVoice-Studio/train/target_speaker_extraction/config/config_KUL_eeg_neuroheed_2spk.yaml		# The config file, only used if it's a new training
#####



# create checkpoint log folder
if [ -z ${checkpoint_dir} ]; then
	checkpoint_dir='checkpoints/log_'$(date '+%Y-%m-%d(%H:%M:%S)')
	train_from_last_checkpoint=0
	mkdir -p ${checkpoint_dir}
	cp $config_pth ${checkpoint_dir}/config.yaml
else
	train_from_last_checkpoint=1
	config_pth=${checkpoint_dir}/config.yaml
fi
yaml_name=log_$(date '+%Y-%m-%d(%H:%M:%S)')
cat $config_pth > ${checkpoint_dir}/${yaml_name}.txt

# call training
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONWARNINGS="ignore"
CUDA_VISIBLE_DEVICES="$gpu_id" \
python -m torch.distributed.launch \
--nproc_per_node=$n_gpu \
--master_port=$(date '+88%S') \
train.py \
--config $config_pth \
--checkpoint_dir $checkpoint_dir \
--train_from_last_checkpoint $train_from_last_checkpoint \
>>${checkpoint_dir}/$yaml_name.txt 2>&1
# CUDA_VISIBLE_DEVICES="$gpu_id" \
# torchrun \
# --nproc_per_node=$n_gpu \
# --master_port=$(date '+88%S') \
# train.py \
# --config $config_pth \
# --checkpoint_dir $checkpoint_dir \
# --train_from_last_checkpoint $train_from_last_checkpoint \
# >>${checkpoint_dir}/$yaml_name.txt 2>&1
