import torch
import torch.nn as nn
from core.layers.STGRUCell import STGRUCell
import math


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        # print(configs.srcnn_tf)
        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.tau = configs.tau
        self.time = configs.time
        self.time_stride = configs.time_stride
        cell_list = []

        width = configs.img_width // configs.patch_size // configs.sr_size
        height = configs.img_height // configs.patch_size // configs.sr_size
        # print(width)

        for i in range(num_layers):
            in_channel = num_hidden[i - 1]
            cell_list.append(
                STGRUCell(in_channel, num_hidden[i], height, width, configs.filter_size,
                          configs.stride, self.frame_channel)
            )
        self.cell_list = nn.ModuleList(cell_list)

        # Encoder
        n = int(math.log2(configs.sr_size))
        # spatial encoders
        s_encoder3d = nn.Sequential()
        s_encoder3d.add_module(name='encoder3d_s_conv{0}'.format(-1),
                               module=nn.Conv3d(in_channels=self.frame_channel,
                                                out_channels=self.num_hidden[0],
                                                stride=1,
                                                padding=0,
                                                kernel_size=(self.time, 1, 1)))
        s_encoder3d.add_module(name='relu3d_s_{0}'.format(-1),
                               module=nn.LeakyReLU(0.2))
        self.s_encoder3d = s_encoder3d

        s_encoders = []
        for i in range(n):
            s_encoder = nn.Sequential()
            s_encoder.add_module(name='encoder_s{0}'.format(i),
                                 module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                  out_channels=self.num_hidden[0],
                                                  stride=(2, 2),
                                                  padding=(1, 1),
                                                  kernel_size=(3, 3)
                                                  ))
            s_encoder.add_module(name='encoder_s_relu{0}'.format(i),
                                 module=nn.LeakyReLU(0.2))
            s_encoders.append(s_encoder)
        self.s_encoders = nn.ModuleList(s_encoders)

        # temporal encoders
        t_encoder3d = nn.Sequential()
        t_encoder3d.add_module(name='encoder3d_t_conv{0}'.format(-1),
                               module=nn.Conv3d(in_channels=self.frame_channel,
                                                out_channels=self.num_hidden[0],
                                                stride=1,
                                                padding=0,
                                                kernel_size=(self.time, 1, 1)))
        t_encoder3d.add_module(name='relu3d_t_{0}'.format(-1),
                               module=nn.LeakyReLU(0.2))
        self.t_encoder3d = t_encoder3d

        t_encoders = []
        for i in range(n):
            t_encoder = nn.Sequential()
            t_encoder.add_module(name='encoder_t{0}'.format(i),
                                 module=nn.Conv2d(in_channels=self.num_hidden[0],
                                                  out_channels=self.num_hidden[0],
                                                  stride=(2, 2),
                                                  padding=(1, 1),
                                                  kernel_size=(3, 3)
                                                  ))
            t_encoder.add_module(name='encoder_t_relu{0}'.format(i),
                                 module=nn.LeakyReLU(0.2))
            t_encoders.append(t_encoder)
        self.t_encoders = nn.ModuleList(t_encoders)

        # Decoder
        # spatial decoder
        s_decoders = []
        for i in range(n - 1):
            s_decoder = nn.Sequential()
            s_decoder.add_module(name='s_decoder{0}'.format(i),
                                 module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                           out_channels=self.num_hidden[-1],
                                                           stride=(2, 2),
                                                           padding=(1, 1),
                                                           kernel_size=(3, 3),
                                                           output_padding=(1, 1)
                                                           ))
            s_decoder.add_module(name='s_decoder_relu{0}'.format(i),
                                 module=nn.LeakyReLU(0.2))
            s_decoders.append(s_decoder)

        if n > 0:
            s_decoder = nn.Sequential()
            s_decoder.add_module(name='s_decoder{0}'.format(n - 1),
                                 module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                           out_channels=self.num_hidden[-1],
                                                           stride=(2, 2),
                                                           padding=(1, 1),
                                                           kernel_size=(3, 3),
                                                           output_padding=(1, 1)
                                                           ))
            s_decoders.append(s_decoder)
        self.s_decoders = nn.ModuleList(s_decoders)

        # temporal decoder
        t_decoders = []
        for i in range(n - 1):
            t_decoder = nn.Sequential()
            t_decoder.add_module(name='t_decoder{0}'.format(i),
                                 module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                           out_channels=self.num_hidden[-1],
                                                           stride=(2, 2),
                                                           padding=(1, 1),
                                                           kernel_size=(3, 3),
                                                           output_padding=(1, 1)
                                                           ))
            t_decoder.add_module(name='t_decoder_relu{0}'.format(i),
                                 module=nn.LeakyReLU(0.2))
            t_decoders.append(t_decoder)

        if n > 0:
            t_decoder = nn.Sequential()
            t_decoder.add_module(name='t_decoder{0}'.format(n - 1),
                                 module=nn.ConvTranspose2d(in_channels=self.num_hidden[-1],
                                                           out_channels=self.num_hidden[-1],
                                                           stride=(2, 2),
                                                           padding=(1, 1),
                                                           kernel_size=(3, 3),
                                                           output_padding=(1, 1)
                                                           ))
            t_decoders.append(t_decoder)
        self.t_decoders = nn.ModuleList(t_decoders)

        self.s_srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.t_srcnn = nn.Sequential(
            nn.Conv2d(self.num_hidden[-1], self.frame_channel, kernel_size=1, stride=1, padding=0)
        )
        self.merge_t = nn.Conv2d(self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)

        self.merge = nn.Conv2d(self.num_hidden[-1] * 2, self.num_hidden[-1], kernel_size=1, stride=1, padding=0)
        self.conv_last_sr = nn.Conv2d(self.frame_channel * 2, self.frame_channel, kernel_size=1, stride=1, padding=0)

    def forward(self, frames, mask_true):
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()
        batch_size = frames.shape[0]
        height = frames.shape[3] // self.configs.sr_size
        width = frames.shape[4] // self.configs.sr_size
        frame_channels = frames.shape[2]
        next_frames = []
        features_t = []
        features_s = []
        R_gates = []
        U_gates = []
        T_t = []
        x_gen = []
        time = self.time
        time_stride = self.time_stride
        input_list = []
        for time_step in range(time - 1):
            input_list.append(
                torch.zeros(
                    [batch_size, frame_channels, height * self.configs.sr_size, width * self.configs.sr_size]).to(
                    self.configs.device))
        for t in range(self.configs.total_length - 1):
            if t < self.configs.input_length:
                net = frames[:, t]
            else:
                time_diff = t - self.configs.input_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
            input_list.append(net)
            input_frm = []
            if t % (time - time_stride) == 0:
                input_frm = torch.stack(input_list[t:])
                input_frm = input_frm.permute(1, 2, 0, 3, 4).contiguous()

            frames_s_feature = self.s_encoder3d(input_frm).squeeze(2)
            frames_t_feature = self.t_encoder3d(input_frm).squeeze(2)

            frames_s_feature_encoded = []
            frames_t_feature_encoded = []
            for i in range(len(self.s_encoders)):
                frames_s_feature = self.s_encoders[i](frames_s_feature)
                frames_s_feature_encoded.append(frames_s_feature)
                frames_t_feature = self.t_encoders[i](frames_t_feature)
                frames_t_feature_encoded.append(frames_t_feature)
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros([batch_size, self.num_hidden[i], height, width]).to(self.configs.device)
                    T_t.append(zeros)
            S_t = frames_s_feature

            features_t.append(frames_t_feature)
            features_s.append(frames_s_feature)
            for i in range(self.num_layers):
                if i == 0:
                    T_t[i] = self.merge_t(torch.cat([T_t[i], frames_t_feature], dim=1))
                T_t[i], S_t, r, u = self.cell_list[i](T_t[i], S_t)
                if(i==1):
                    R_gates.append(r)
                    U_gates.append(u)
            out_s = T_t[-1]
            out_t = S_t
            for i in range(len(self.s_decoders)):
                out_s = self.s_decoders[i](out_s)
                if i < len(self.s_decoders) - 1:
                    out_s = out_s + self.configs.alpha*frames_s_feature_encoded[-2 - i]
                out_t = self.t_decoders[i](out_t)
                if i < len(self.s_decoders) - 1:
                    out_t = out_t + self.configs.alpha*frames_t_feature_encoded[-2 - i]

            out_s = self.s_srcnn(out_s)
            out_t = self.t_srcnn(out_t)
            x_gen = self.conv_last_sr(torch.cat([out_s, out_t], dim=1))
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2, 3, 4).contiguous()
        features = []
        features_t = torch.stack(features_t, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        features_s = torch.stack(features_s, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        R_gates = torch.stack(R_gates, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        U_gates = torch.stack(U_gates, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        features.append(features_t)
        features.append(features_s)
        features.append(R_gates)
        features.append(U_gates)
        features = torch.stack(features, dim=0)
        return next_frames, features
