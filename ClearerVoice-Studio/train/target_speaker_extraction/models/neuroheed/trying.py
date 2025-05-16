
# Copyright 2020 Kai Li
# Apache-2.0 license http://www.apache.org/licenses/LICENSE-2.0
# Modified from https://github.com/JusperLee/Dual-Path-RNN-Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder



EPS = 1e-8

from .s4 import LinearActivation, S4
from .sashimi import FFBlock, ResidualBlock
from torch.nn import TransformerEncoder, TransformerEncoderLayer
class S4_Block(nn.Module):
    """
    Replaces the Dual_RNN_Block with an S4-based block
    """
    def __init__(self, out_channels, hidden_channels, dropout=0, bidirectional=False, num_spks=2):
        super(S4_Block, self).__init__()
        
        # Intra-processing with S4
        self.intra_s4 = S4(
            d_model=out_channels,
            d_state=hidden_channels,
            dropout=dropout,
            transposed=True,
            lr=0.001
        )
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.intra_linear = nn.Linear(out_channels, out_channels)
        
        # Inter-processing with S4
        self.inter_s4 = S4(
            d_model=out_channels,
            d_state=hidden_channels,
            dropout=dropout,
            transposed=True,
            lr=0.001
        )
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_linear = nn.Linear(out_channels, out_channels)
        
    def forward(self, x):
        """
        x: [B, N, K, S]
        out: [B, N, K, S]
        """
        B, N, K, S = x.shape
        
        # Intra processing (along K dimension)
        # [B, N, K, S] -> [B*S, N, K]
        intra = x.permute(0, 3, 1, 2).contiguous().view(B*S, N, K)
        intra,_ = self.intra_s4(intra)  # [B*S, N, K]
        intra = self.intra_linear(intra.view(B*S*K, -1)).view(B*S, N, K)
        intra = intra.view(B, S, N, K).permute(0, 2, 3, 1).contiguous()
        intra = self.intra_norm(intra)
        intra = intra + x
        
        # Inter processing (along S dimension)
        # [B, N, K, S] -> [B*K, N, S]
        inter = intra.permute(0, 2, 1, 3).contiguous().view(B*K, N, S)
        inter,_ = self.inter_s4(inter)  # [B*K, N, S]
        inter = self.inter_linear(inter.view(B*K*S, -1)).view(B*K, N, S)
        inter = inter.view(B, K, N, S).permute(0, 2, 1, 3).contiguous()
        inter = self.inter_norm(inter)
        
        out = inter + intra
        
        return out


class neuroheed(nn.Module):
    def __init__(self, args, N=256, L=20, B=64, H=128, K=100, R=6):
        super(neuroheed, self).__init__()
        self.args = args
        self.N, self.L, self.B, self.H, self.K, self.R = args.network_audio.N, args.network_audio.L, args.network_audio.B, args.network_audio.H, args.network_audio.K, args.network_audio.R
        
        self.encoder = Encoder(self.L, self.N)
        self.separator = rnn(self.args, self.N, self.B, self.H, self.K, self.R)
        self.decoder = Decoder(self.N, self.L)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture, eeg, reference=None):
        eeg = eeg.to(self.args.device)
        #print('shape of eeg',eeg.shape)

        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w, eeg, reference, mixture)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source

class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super(Decoder, self).__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        est_source = mixture_w * est_mask  # [M,  N, K]
        est_source = torch.transpose(est_source, 2, 1) # [M,  K, N]
        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source



class rnn(nn.Module):
    def __init__(self, args, N, B, H, K, R):
        super(rnn, self).__init__()
        self.args = args
        self.K , self.R = K, R
        
        self.layer_norm = nn.GroupNorm(1, N, eps=1e-8)
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)

        # Replace Dual-RNN blocks with S4 blocks
        self.blocks = nn.ModuleList([])
        for i in range(R):
            self.blocks.append(S4_Block(
                out_channels=B,
                hidden_channels=H,
                dropout=0.1
            ))

        self.prelu = nn.PReLU()
        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)

        self.po_encoding = PositionalEncoding(d_model=64)
        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64*4)
        self.eeg_net = TransformerEncoder(encoder_layers, num_layers=5)
        self.fusion = nn.Conv1d(B+64, B, 1, bias=False)

    def forward(self, x, eeg, reference, speech):
        mixture_w = x
        M, N, D = x.size()
        
        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)

        # EEG processing
        eeg = self.po_encoding(eeg.transpose(0,1))
        eeg = self.eeg_net(eeg)
        eeg = eeg.transpose(0,1).transpose(1,2)
        eeg = F.interpolate(eeg, (D), mode='linear')
        x = torch.cat((x, eeg), 1)
        x = self.fusion(x)

        # Segmentation
        x, gap = self._Segmentation(x, self.K)

        # Process through S4 blocks
        for block in self.blocks:
            x = block(x)

        # Overlap-add and output
        x = self._over_add(x, gap)
        x = self.prelu(x)
        x = self.mask_conv1x1(x)
        x = x.view(M, N, D)
        x = F.relu(x)
        
        return x

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K).transpose(2, 3)

        return input.contiguous(), gap


    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length
    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.
    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length
    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long().cuda()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result

# if __name__ == "__main__":
#     import yamlargparse
#     parser = yamlargparse.ArgumentParser("Settings")
#     parser.add_argument('--config', help='config file path', action=yamlargparse.ActionConfigFile) 
#     args = parser.parse_args()
#     import sys
#     import os
#     sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

#     from s4 import S4
#     N=64
#     B=64 # op channels
#     S=104 #lmax 
#     K=100
#     M=4 #batch size
#     H=64
#     R=5
#     model = Dual_RNN_Block(out_channels=B,hidden_channels=2*B)
    
#     input=torch.randn([B,N,K,S])
#     op=model(input)
#     print(op.shape)
#     print(model)
#     # #mixture_w: [M, N, K],  N,B,H,K,R
#     ip_for_rnn=torch.randn([M,N,K]) #audio
#     model_rnn=rnn(args,N=N,B=B,H=H,K=K,R=R)
#     dummy_eeg=torch.randn([M, 803, B])
#     op_of_rnn=model_rnn(ip_for_rnn,reference=None,eeg=dummy_eeg,speech=None)
#     print('op of rnn shape',op_of_rnn.shape)
#     print(model)