import torch
import torch.nn as nn


class STGRUCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride, frame_channel):
        super(STGRUCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = (filter_size[0] // 2, filter_size[1] // 2)
        self._forget_bias = 1.0
        self.frame_channel = frame_channel
        self.conv_t = nn.Sequential(
            nn.Conv2d(in_channel, 4 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([4 * num_hidden, height, width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(num_hidden, 4 * num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding,
                      ),
            nn.LayerNorm([4 * num_hidden, height, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0,
                                   )

    def forward(self, T_t, S_t):
        # print('yes')
        T_concat = self.conv_t(T_t)
        S_concat = self.conv_s(S_t)
        t_z, t_r, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1)
        s_z, s_r, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1)
        Z_t = torch.sigmoid(t_z + s_z + self._forget_bias)
        R_t = torch.sigmoid(t_r + s_r)
        T_tmp = torch.tanh(t_t + R_t * s_t)
        S_tmp = torch.tanh(s_s + R_t * t_s)
        T_new = (1 - Z_t) * T_tmp + Z_t * T_t
        S_new = (1 - Z_t) * S_tmp + Z_t * S_t
        # H_new = self.conv_last(torch.cat([T_new, S_new], dim=1))

        return T_new, S_new, R_t, Z_t
