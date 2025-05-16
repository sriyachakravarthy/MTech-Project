import torch.nn as nn
import torch 
import torch.nn.functional as F
import os
import sys
sys.path.append(os.path.dirname(__file__))
from .conv_stft import ConvSTFT, ConviSTFT
import numpy as np
from models.frcrn.unet import UNet

class FRCRN_Wrapper_StandAlone(nn.Module):
    def __init__(self, args):
        super(FRCRN_Wrapper_StandAlone, self).__init__()
        self.model = DCCRN(
            complex=True,
            model_complexity=45,
            model_depth=14,
            log_amp=False,
            padding_mode="zeros",
            win_len=640,
            win_inc=320,
            fft_len=640,
            win_type='hanning'
        )

    def forward(self, x):
        output = self.model(x)       
        return output[1][0]

class FRCRN_SE_16K(nn.Module):
    def __init__(self, args):
        super(FRCRN_SE_16K, self).__init__()
        self.model = DCCRN(
            complex=True,
            model_complexity=45,
            model_depth=14,
            log_amp=False,
            padding_mode="zeros",
            win_len=args.win_len,
            win_inc=args.win_inc,
            fft_len=args.fft_len,
            win_type=args.win_type
        )

    def forward(self, x):
        output = self.model(x)
        return output[1][0]

class DCCRN(nn.Module):
    def __init__(self, complex, model_complexity, model_depth, log_amp, padding_mode, win_len=400, win_inc=100, fft_len=512, win_type='hanning'):
        """
        :param complex: whether to use complex networks.
        :param model_complexity: only used for model_depth 20
        :param model_depth: Only two options are available : 14 or 20
        :param log_amp: Whether to use log amplitude to estimate signals
        :param padding_mode: Encoder's convolution filter. 'zeros', 'reflect'
        """
        super().__init__()
        self.feat_dim = fft_len // 2 +1

        self.win_len = win_len
        self.win_inc = win_inc
        self.fft_len = fft_len
        self.win_type = win_type

        fix = True
        self.stft = ConvSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex', fix=fix)
        self.istft = ConviSTFT(self.win_len, self.win_inc, self.fft_len, self.win_type, feature_type='complex', fix=fix)
        self.unet = UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode)
        self.unet2 = UNet(1, complex=complex, model_complexity=model_complexity, model_depth=model_depth, padding_mode=padding_mode)

    def forward(self, inputs):
        out_list = []
        # [B, D*2, T]
        cmp_spec = self.stft(inputs)
        # [B, 1, D*2, T]
        cmp_spec = torch.unsqueeze(cmp_spec, 1)
        # to [B, 2, D, T] real_part/imag_part
        cmp_spec = torch.cat([
                                cmp_spec[:,:,:self.feat_dim,:],
                                cmp_spec[:,:,self.feat_dim:,:],
                                ],
                                1)
        # [B, 2, D, T]
        #cmp_spec_orig = cmp_spec.clone()
        cmp_spec = torch.unsqueeze(cmp_spec, 4)
        # [B, 1, D, T, 2]
        cmp_spec = torch.transpose(cmp_spec, 1, 4)
        # (*, channel, num_freqs, time, complex=2)
        ## cmp_mask: [*, 1, 513, 64, 2]
        unet1_out = self.unet(cmp_spec)
        cmp_mask1 = torch.tanh(unet1_out)
        unet2_out = self.unet2(unet1_out)
        cmp_mask2 = torch.tanh(unet2_out)
        cmp_mask2 = cmp_mask2 + cmp_mask1
        est_spec, est_wav, est_mask = self.apply_mask(cmp_spec, cmp_mask2)
        out_list.append(est_spec)
        out_list.append(est_wav)
        out_list.append(est_mask)
        return out_list

    def inference(self, inputs):
        #cmp_spec: [B, D*2, T]
        cmp_spec = self.stft(inputs)
        # [B, 1, D*2, T]
        cmp_spec = torch.unsqueeze(cmp_spec, 1)
        # to [B, 2, D, T] real_part/imag_part
        cmp_spec = torch.cat([
                                cmp_spec[:,:,:self.feat_dim,:],
                                cmp_spec[:,:,self.feat_dim:,:],
                                ],
                                1)
        # [B, 2, D, T, 1]
        cmp_spec = torch.unsqueeze(cmp_spec, 4)
        # [B, 1, D, T, 2]
        cmp_spec = torch.transpose(cmp_spec, 1, 4)
        # (*, channel, num_freqs, time, complex=2)
        ## cmp_mask: [*, 1, 513, 64, 2]
        unet1_out = self.unet(cmp_spec)
        cmp_mask1 = torch.tanh(unet1_out)
        unet2_out = self.unet2(unet1_out)
        cmp_mask2 = torch.tanh(unet2_out)
        cmp_mask2 = cmp_mask2 + cmp_mask1
        _, est_wav, _ = self.apply_mask(cmp_spec, cmp_mask2)
        return est_wav[0]

    def apply_mask(self, cmp_spec, cmp_mask):
        est_spec = torch.cat([cmp_spec[:,:,:,:,0]*cmp_mask[:,:,:,:,0]-cmp_spec[:,:,:,:,1]*cmp_mask[:,:,:,:,1], cmp_spec[:,:,:,:,0]*cmp_mask[:,:,:,:,1]+cmp_spec[:,:,:,:,1]*cmp_mask[:,:,:,:,0]],1)
        est_spec = torch.cat([est_spec[:,0,:,:], est_spec[:,1,:,:]], 1)
        cmp_mask = torch.squeeze(cmp_mask, 1)
        cmp_mask = torch.cat([cmp_mask[:,:,:,0], cmp_mask[:,:,:,1]],1)
        est_wav = self.istft(est_spec)
        est_wav = torch.squeeze(est_wav, 1)
        return est_spec, est_wav, cmp_mask

    def get_params(self, weight_decay=0.0):
            # add L2 penalty
        weights, biases = [], []
        for name, param in self.named_parameters():
            if 'bias' in name:
                biases += [param]
            else:
                weights += [param]
        params = [{
                     'params': weights,
                     'weight_decay': weight_decay,
                 }, {
                     'params': biases,
                     'weight_decay': 0.0,
                 }]
        return params 
''' 
    #def loss(self, noisy, est, est_wav, labels, cmp_mask, mode='Mix'):
    def loss(self, noisy_wav, clean_wav, out_list, device, mode='Mix'):
        if mode == 'SiSNR':
            #count = 0
            #while count < len(out_list):
            est_spec = out_list[0]
            #   count = count + 1
            est_wav = out_list[1]
            #   count = count + 1
            est_mask = out_list[2]
            #   count = count + 1
            #   if count == 3:
            #       SiSNR_loss1 = self.loss_1layer(noisy, est_spec, est_wav, labels, est_mask, mode)
            #       loss1 = SiSNR_loss1
            #   else:
            loss = self.loss_1layer(noisy_wav, est_spec, est_wav, clean_wav, est_mask, device, mode)
            #       loss2 = SiSNR_loss2
            #num_layers = int(len(out_list)/3)
            #loss = loss2
            return loss

        elif mode == 'Mix':
            #count = 0
            #while count < len(out_list):
            est_spec = out_list[0]
            #   count = count + 1
            est_wav = out_list[1]
            #   count = count + 1
            est_mask = out_list[2]
            #   count = count + 1
            #   if count == 3:
            #       amp_loss, phase_loss, SiSNR_loss = self.loss_1layer(noisy, est_spec, est_wav, labels, est_mask, mode)
            #       loss1 = amp_loss + phase_loss + SiSNR_loss
            #   else:
            mask_real_loss, mask_imag_loss, SiSNR_loss = self.loss_1layer(noisy_wav, est_spec, est_wav, clean_wav, est_mask, device, mode)
            loss = mask_real_loss + mask_imag_loss + SiSNR_loss
            #num_layers = int(len(out_list)/3)
            #loss = loss2 #/num_layers
            return loss, mask_real_loss, mask_imag_loss
 
    def loss_1layer(self, noisy_wav, est_spec, est_wav, clean_wav, cmp_mask, device, mode='Mix'):
        if mode == 'SiSNR':
            if clean_wav.dim() == 3:
                clean_wav = torch.squeeze(clean_wav,1)
            if est_wav.dim() == 3:
                est_wav = torch.squeeze(est_wav,1)
            return -si_snr(est_wav, clean_wav)
        elif mode == 'Mix':

            if clean_wav.dim() == 3:
                clean_wav = torch.squeeze(clean_wav,1)
            if est_wav.dim() == 3:
                est_wav = torch.squeeze(est_wav,1)
            SiSNR_loss = -si_snr(est_wav, clean_wav)

            b, d, t = est_spec.size()
            #print('est: {}'.format(est.size()))
            #S_torch = stft(clean_wav, self.win_type, self.win_len, self.win_inc, self.fft_len, device)
            S = self.stft(clean_wav)
            #print(f'S-torch: {S_torch.shape}, S: {S.shape}')
            Sr = S[:, :self.feat_dim, :]
            Si = S[:, self.feat_dim:, :]
            #Sr = S_torch[...,0]
            #Si = S_torch[...,1]
            Y = self.stft(noisy_wav)
            Yr = Y[:, :self.feat_dim, :]
            Yi = Y[:, self.feat_dim:, :]
            #Y_torch = stft(noisy_wav, self.win_type, self.win_len, self.win_inc, self.fft_len, device)
            #Yr = Y_torch[...,0]
            #Yi = Y_torch[...,1]
            Y_pow = Yr**2 + Yi**2
            Y_mag = torch.sqrt(Y_pow)
            gth_mask = torch.cat([(Sr*Yr+Si*Yi)/(Y_pow + 1e-8),(Si*Yr-Sr*Yi)/(Y_pow + 1e-8)], 1)
            gth_mask[gth_mask > 2] = 1
            gth_mask[gth_mask < -2] = -1
            mask_real_loss = F.mse_loss(gth_mask[:,:self.feat_dim, :], cmp_mask[:,:self.feat_dim, :]) * d
            mask_imag_loss = F.mse_loss(gth_mask[:,self.feat_dim:, :], cmp_mask[:,self.feat_dim:, :]) * d
            #all_loss = mask_real_loss + mask_imag_loss + SiSNR_loss
            return mask_real_loss, mask_imag_loss, SiSNR_loss

def stft(x, win_type, win_len, win_inc, fft_len, device):
    #win_type = args.win_type
    #win_len = args.win_len
    #win_inc = args.win_inc
    #fft_len = args.fft_len
    if win_type == 'hamming':
        window = torch.hamming_window(win_len, periodic=False)
    elif win_type == 'hanning':
        window = torch.hann_window(win_len, periodic=False)
    else:
        print(f"{win_type} is not supported!")
        return
    window = window.to(device)
    return torch.stft(x, fft_len, win_inc, win_len, center=False, window=window)

def remove_dc(data):
    mean = torch.mean(data, -1, keepdim=True) 
    data = data - mean
    return data

def l2_norm(s1, s2):
    norm = torch.sum(s1*s2, -1, keepdim=True)
    return norm 

def si_snr(s1, s2, eps=1e-8):
    #s1 = remove_dc(s1)
    #s2 = remove_dc(s2)
    s1_s2_norm = l2_norm(s1, s2)
    s2_s2_norm = l2_norm(s2, s2)
    s_target =  s1_s2_norm/(s2_s2_norm+eps)*s2
    e_nosie = s1 - s_target
    target_norm = l2_norm(s_target, s_target)
    noise_norm = l2_norm(e_nosie, e_nosie)
    snr = 10*torch.log10((target_norm)/(noise_norm+eps)+eps)
    return torch.mean(snr)

#if __name__ == '__main__':
#    test_DCCRN()
'''
