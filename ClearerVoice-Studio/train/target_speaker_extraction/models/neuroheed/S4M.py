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

        self.mask_conv = nn.Conv1d(out_channels,  self.enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), self.mask_conv)
    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.unet(x)
            else:
                x = self.unet(self.concat_block(mixture + x))

        return x

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


class S4M(nn.Module):
    def __init__(
        self,
        #args,
        out_channels=128,
        in_channels=512,
        num_blocks=16,
        upsampling_depth=4,
        enc_kernel_size=21,
        num_sources=2,
        sample_rate=8000,
    ):
        super(S4M, self).__init__()

        # Number of sources to produce
        #self.args=args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size * sample_rate // 1000
        self.enc_num_basis = self.enc_kernel_size // 2 + 1
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(
            self.enc_kernel_size // 4 * 4 ** self.upsampling_depth
        ) // math.gcd(self.enc_kernel_size // 4, 4 ** self.upsampling_depth)
        # Front end
        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=self.enc_num_basis,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(self.enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=self.enc_num_basis, out_channels=out_channels, kernel_size=1
        )

        # Separation module
        self.sm = Recurrent(out_channels, in_channels, upsampling_depth, num_blocks, enc_num_basis=self.enc_num_basis)

        mask_conv = nn.Conv1d(out_channels,  self.enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
        self.decoder = nn.ConvTranspose1d(
            in_channels=self.enc_num_basis ,
            out_channels=1,
            kernel_size=self.enc_kernel_size,
            stride=self.enc_kernel_size // 4,
            padding=self.enc_kernel_size // 2,
            groups=1,
            bias=False,
        )
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()
        self.po_encoding = PositionalEncoding(d_model=64)
        encoder_layers = TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=64*4)
        self.eeg_net =  TransformerEncoder(encoder_layers, num_layers=5)

    def pad_input(self, input, window, stride):
        """
        Zero-padding input according to window/stride size.
        """
        batch_size, nsample = input.shape

        # pad the signals at the end for matching the window/stride size
        rest = window - (stride + nsample % window) % window
        if rest > 0:
            pad = torch.zeros(batch_size, rest).type(input.type())
            input = torch.cat([input, pad], 1)
        pad_aux = torch.zeros(batch_size, window - stride).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 1)

        return input, rest

    # Forward pass
    def forward(self, input_wav,eeg):
        # input shape: (B, T)
        print(input_wav.shape)
        print('dim of ip wave',input_wav.ndim)
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0)
        if input_wav.ndim == 2:
            input_wav = input_wav
        if input_wav.ndim == 3:
            print('Alert.............')
            input_wav = input_wav.reshape(input_wav.size(0), -1)  # or audio.reshape(audio.size(0), -1)
            print('dim of ip wave',input_wav.ndim)
            print(input_wav.shape)
            #input_wav = input_wav.squeeze(1)
        
        print('shape of eeg before pe',eeg.shape)
        eeg = self.po_encoding(eeg.transpose(0,1))
        print('shape of eeg after pe',eeg.shape)

        eeg = self.eeg_net(eeg)
        print('shape of eeg after eeg net',eeg.shape)

        eeg = eeg.transpose(0,1).transpose(1,2)
        print('shape of eeg after eeg net and transposed back',eeg.shape)

        

        x, rest = self.pad_input(
            input_wav, self.enc_kernel_size, self.enc_kernel_size // 4
        )
        print('padded.........')
        print(x.shape)
        # Front end
        

        x = self.encoder(x.unsqueeze(1))
        M, N, D = x.size()
        print('..............')
        print('audio size after encoding',x.size())
        print(eeg.shape)
        eeg = F.interpolate(eeg, (D), mode='linear')
        print('eeg shape after interpolation',eeg.shape)
        if eeg.ndim==3:
            eeg = eeg.reshape(eeg.size(0), -1)

        print(eeg.shape)
        print('.............')
        # Split paths
        s=x.clone()
        # Separation module
        x = self.ln(x)
        # [batch, 9, 8010] -> [batch, 128, 8010]
        x = self.bottleneck(x)

        # [batch, 9, 8010] -> [batch, 128, 8010]
        x = self.sm(x)

        print('shape after sm',x.shape)

        # [batch, 128, 8010] -> [batch, 18, 8010]
        x = self.mask_net(x)

        # [batch, 128, 8010] -> [batch, 2, 9, 8010]
        x = x.view(x.shape[0],  self.enc_num_basis, -1)
        # [batch, 2, 9, 8010] -> [batch, 2, 9, 8010]
        x = self.mask_nl_class(x)
        print('mask shape',x.shape)
        x = x * s.unsqueeze(1)
        print('..............',x.shape)
        # Back end  ##THINKING 
        estimated_waveforms = self.decoder(x.view(x.shape[0], x.shape[-1]))
        # estimated_waveforms = estimated_waveforms[
        #     :,
        #     :,
        #     self.enc_kernel_size
        #     - self.enc_kernel_size
        #     // 4 : -(rest + self.enc_kernel_size - self.enc_kernel_size // 4),
        # ].contiguous()
        estimated_waveforms=estimated_waveforms.contiguous()
        if was_one_d:
            return estimated_waveforms.squeeze(0)

        return estimated_waveforms


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = S4M()

    M=4
    N=64
    K=100

    audio = torch.randn([M,N,K])
    dummy_eeg=torch.randn([4, 803, 64])

    print("=== Model Architecture ===")
    print(model)

    print("\n=== Forward Output Shape ===")
    output = model(audio,eeg=dummy_eeg)
    print(output.shape)

    print("\n=== Trainable Parameters ===")
    print(f"{count_parameters(model):,} parameters")