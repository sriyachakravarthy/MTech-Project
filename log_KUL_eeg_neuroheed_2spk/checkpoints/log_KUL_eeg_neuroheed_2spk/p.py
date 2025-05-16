import torch

checkpoint_path = '/home/sriyar/neuroheed/log_KUL_eeg_neuroheed_2spk/checkpoints/log_KUL_eeg_neuroheed_2spk/last_best_checkpoint.pt'

try:
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    print("Weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
