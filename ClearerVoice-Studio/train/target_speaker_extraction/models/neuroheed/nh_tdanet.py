# Copyright 2020 Kai Li
# Apache-2.0 license http://www.apache.org/licenses/LICENSE-2.0
# Modified from https://github.com/JusperLee/Dual-Path-RNN-Pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


EPS = 1e-8

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
       # print('EEG shape line 30',eeg.shape)
        print('mixture shape line 31',mixture.shape)
        mixture_w = self.encoder(mixture)
        #print('mixture_w shape line 33',mixture_w.shape)
        est_mask = self.separator(mixture_w, eeg, reference, mixture)
       # print('est_mask shape line 35',est_mask.shape)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        print('est_mask shape line 35',est_mask.shape)
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

class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            in_channels: The number of expected features in the input x
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            norm: gln = "Global Norm", cln = "Cumulative Norm", ln = "Layer Norm"
            dropout: If non-zero, introduces a Dropout layer on the outputs 
                     of each LSTM layer except the last layer, 
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(self, out_channels,
                 hidden_channels, rnn_type='LSTM',
                 dropout=0, bidirectional=False, num_spks=2):
        super(Dual_RNN_Block, self).__init__()
        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        # Norm
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        self.inter_linear = nn.Linear(
            hidden_channels*2 if bidirectional else hidden_channels, out_channels)
        

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [Spks, B, N, K, S]
        '''
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)
        
        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out

class rnn(nn.Module):
    def __init__(self, args, N, B, H, K, R):
        super(rnn, self).__init__()
        self.args = args
        self.K , self.R = K, R
        # [M, N, K] -> [M, N, K]
        self.layer_norm = nn.GroupNorm(1, N, eps=1e-8)
        # [M, N, K] -> [M, B, K]
        self.bottleneck_conv1x1 = nn.Conv1d(N, B, 1, bias=False)

        self.dual_rnn = nn.ModuleList([])
        for i in range(R):
            self.dual_rnn.append(TDANetBlock(out_channels=128, in_channels=512, upsampling_depth=4))

        self.prelu = nn.PReLU()
        self.mask_conv1x1 = nn.Conv1d(B, N, 1, bias=False)


        self.po_encoding = PositionalEncoding(d_model=64)
        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64*4)
        self.eeg_net = TransformerEncoder(encoder_layers, num_layers=5)
        self.fusion = nn.Conv1d(B+64, B, 1, bias=False)


    def forward(self, x, eeg, reference, speech):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        mixture_w = x
        M, N, D = x.size()

        x = self.layer_norm(x) # [M, N, K]
        x = self.bottleneck_conv1x1(x) # [M, B, K]


        eeg = self.po_encoding(eeg.transpose(0,1))
        eeg = self.eeg_net(eeg)
        eeg = eeg.transpose(0,1).transpose(1,2)
        eeg = F.interpolate(eeg, (D), mode='linear')
        print('eeg after encoding',eeg.shape)
        x = torch.cat((x, eeg),1)
        x  = self.fusion(x)


        x, gap = self._Segmentation(x, self.K) # [M, B, k, S]

        for i in range(self.R):
            x = self.dual_rnn[i](x)

        x = self._over_add(x, gap)

        x = self.prelu(x)
        x = self.mask_conv1x1(x)

        x = x.view(M, N, D) # [M, C*N, K] -> [M, C, N, K]
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


from audioop import bias
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .s4 import LinearActivation, S4
from .sashimi import FFBlock, ResidualBlock

def s4_block(dim):
    layer = S4(
        d_model=dim,
        d_state=16,
        bidirectional=True,
        dropout=0.0,
        transposed=True,
    )
    return ResidualBlock(
        d_model=dim,
        layer=layer,
        dropout=0.0,
    )


def ff_block(dim):
    layer = FFBlock(
        d_model=dim,
        expand=2,
        dropout=0.0,
    )
    return ResidualBlock(
        d_model=dim,
        layer=layer,
        dropout=0.0,
    )


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob

    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size), requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) + self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    """
    This class defines the convolution layer with normalization and a PReLU
    activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=True, groups=groups
        )
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    """
    This class defines the convolution layer with normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1, bias=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(
            nIn, nOut, kSize, stride=stride, padding=padding, bias=bias, groups=groups
        )
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    """
    This class defines a normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: number of output channels
        """
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    """
    This class defines the dilated convolution.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    """
    This class defines the dilated convolution with normalized output.
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        self.conv = nn.Conv1d(
            nIn,
            nOut,
            kSize,
            stride=stride,
            dilation=d,
            padding=((kSize - 1) // 2) * d,
            groups=groups,
        )
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)

class LALayer(nn.Module):
    def __init__(self, inp: int, oup: int, kernel: int = 1) -> None:
        super().__init__()
        groups = 1
        if inp == oup:
            groups = inp
        self.local_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_embedding = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.global_act = ConvNorm(inp, oup, kernel, groups=groups, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x_l, x_g,):
        """
        x_g: global features
        x_l: local features
        """
        B, N, T = x_l.shape
        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(x_g)
        sig_act = F.interpolate(self.act(global_act), size=T, mode="nearest")

        global_feat = self.global_embedding(x_g)
        global_feat = F.interpolate(global_feat, size=T, mode="nearest")

        out = local_feat * sig_act + global_feat

        return out


class TDANetBlock(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1, stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList()
        self.spp_dw.append(
            DilatedConvNorm(
                in_channels, in_channels, kSize=5, stride=1, groups=in_channels, d=1
            )
        )

        for i in range(1, upsampling_depth):
            if i == 0:
                stride = 1
            else:
                stride = 2
            self.spp_dw.append(
                DilatedConvNorm(
                    in_channels,
                    in_channels,
                    kSize=2 * stride + 1,
                    stride=stride,
                    groups=in_channels,
                    d=1,
                )
            )

        self.loc_glo_fus = nn.ModuleList([])
        for i in range(upsampling_depth):
            self.loc_glo_fus.append(LALayer(in_channels, in_channels))

        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.s4 = s4_block(in_channels)
        self.ff = ff_block(in_channels)

        self.last_layer = nn.ModuleList([])
        for i in range(self.depth - 1):
            self.last_layer.append(LALayer(in_channels, in_channels, 5))

    def forward(self, x):
        residual = x.clone()
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]

        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        # global features
        global_f = torch.zeros(
            output[-1].shape, requires_grad=True, device=output1.device
        )
        for fea in output:
            global_f = global_f + F.adaptive_avg_pool1d(
                fea, output_size=output[-1].shape[-1]
            )
        
        global_f, _ = self.s4(global_f)
        global_f, _ = self.ff(global_f)


        x_fused = []
        # Gather them now in reverse order
        for idx in range(self.depth):
            local = output[idx]
            x_fused.append(self.loc_glo_fus[idx](local, global_f))

        expanded = None
        for i in range(self.depth - 2, -1, -1):
            if i == self.depth - 2:
                expanded = self.last_layer[i](x_fused[i], x_fused[i - 1])
            else:
                expanded = self.last_layer[i](x_fused[i], expanded)
        # import pdb; pdb.set_trace()
        return self.res_conv(expanded) + residual


class Recurrent(nn.Module):
    def __init__(self, out_channels=128, in_channels=512, upsampling_depth=4, _iter=4):
        super().__init__()
        self.unet = TDANetBlock(out_channels, in_channels, upsampling_depth)
        self.iter = _iter
        self.concat_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels), nn.PReLU()
        )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.unet(x)
            else:
                x = self.unet(self.concat_block(mixture + x))
        return x
    
    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.unet(x)
            else:
                x = self.unet(self.concat_block(mixture + x))

        return x
