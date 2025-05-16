# Copyright 2018 Kaituo XU
#  MIT License (https://opensource.org/licenses/MIT)

import torch
import torch.nn as nn

import torch.nn.functional as F

EPS = 1e-8

def cal_SISNR(source, estimate_source):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        source: torch tensor, [batch size, sequence length]
        estimate_source: torch tensor, [batch size, sequence length]
    Returns:
        SISNR, [batch size]
    """

    T_source = source.size(-1)
    T_est = estimate_source.size(-1)

    if T_est < T_source:
        estimate_source = F.pad(estimate_source, (0, T_source - T_est))
    elif T_est > T_source:
        source = F.pad(source, (0, T_est - T_source))

    assert source.size() == estimate_source.size()
    # Step 1. Zero-mean norm
    source = source - torch.mean(source, axis = -1, keepdim=True)
    estimate_source = estimate_source - torch.mean(estimate_source, axis = -1, keepdim=True)
    # Step 2. SI-SNR
    # s_target = <s', s>s / ||s||^2
    ref_energy = torch.sum(source ** 2, axis = -1, keepdim=True) + EPS
    proj = torch.sum(source * estimate_source, axis = -1, keepdim=True) * source / ref_energy
    # e_noise = s' - s_target
    noise = estimate_source - proj
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
    ratio = torch.sum(proj ** 2, axis = -1) / (torch.sum(noise ** 2, axis = -1) + EPS)
    sisnr = 10 * torch.log10(ratio + EPS)
    return sisnr


def cal_SDR(target, est_target):
    assert target.size() == est_target.size()
    # Step 1. Zero-mean norm
    mean_source = torch.mean(target, dim=1, keepdim=True)
    mean_estimate = torch.mean(est_target, dim=1, keepdim=True)
    target = target - mean_source
    est_target = est_target - mean_estimate
    # Step 2. SDR
    scaled_target = target
    e_noise = est_target - target
    sdr = torch.sum(scaled_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + EPS)
    sdr = 10 * torch.log10(sdr + EPS)
    return sdr  

    
def clip_loss(eeg_embed, speech_embed, temperature=0.07):
    """
    Computes symmetric contrastive CLIP loss between EEG and speech embeddings.
    Args:
        eeg_embed: [B, D]
        speech_embed: [B, D]
        temperature: scalar float
    Returns:
        scalar contrastive loss
    """
    eeg_embed = F.normalize(eeg_embed, dim=-1)
    speech_embed = F.normalize(speech_embed, dim=-1)
    logits = torch.matmul(eeg_embed, speech_embed.T) / temperature
    labels = torch.arange(logits.size(0)).to(logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
