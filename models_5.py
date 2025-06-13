import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell that keeps spatial structure in its hidden state.
    Input x:  [B, C_in, H, W]
    Hidden h, c: [B, C_hidden, H, W]
    Returns: (h_next, c_next), both [B, C_hidden, H, W]
    """
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        # A single Conv2d to produce 4 * hidden_dim channels: i, f, g, o gates
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, hidden):
        """
        Args:
            x:      tensor of shape [B, C_in, H, W]
            hidden: tuple (h_prev, c_prev), each [B, C_hidden, H, W]
        Returns:
            (h_next, c_next), both [B, C_hidden, H, W]
        """
        h_prev, c_prev = hidden
        # concatenate along channel dimension
        combined = torch.cat([x, h_prev], dim=1)   # [B, C_in + C_hidden, H, W]
        gates = self.conv(combined)                # [B, 4*C_hidden, H, W]
        # split into 4 equal chunks of size hidden_dim
        i, f, g, o = torch.split(gates, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)    # input gate
        f = torch.sigmoid(f)    # forget gate
        g = torch.tanh(g)       # cell candidate
        o = torch.sigmoid(o)    # output gate

        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class SpatioTemporalClimateNet(nn.Module):
    """
    A 3D‐Conv U‐Net + ConvLSTM bottleneck for predicting climate fields.
    Input:  x of shape [B, T, C_in, H, W]  (T may be ≥1)
    Output: [B, 2, H, W]  (channel 0 = precipitation, channel 1 = temperature)
    """
    def __init__(
        self,
        in_channels: int = 11,
        seq_length: int = 12,
        base_channels: int = 32,
        hidden_channels: int = 64,
        lstm_channels: int = 64,
        dropout: float = 0.25
    ):
        """
        Args:
            in_channels    : number of input variables (e.g. 11 climate features) per time step.
            seq_length     : nominal sequence length (for bookkeeping; actual T can differ).
            base_channels  : number of channels after the first 3D conv block.
            hidden_channels: number of channels used in decoder upsampling blocks.
            lstm_channels  : number of hidden channels in ConvLSTMCell.
            dropout        : probability for Dropout3d in each conv block (0 ⇒ no dropout).
        """
        super(SpatioTemporalClimateNet, self).__init__()
        self.seq_length = seq_length

        # ------------------------
        # Encoder: two 3D conv blocks that spatially downsample by factor of 2 twice.
        # We will permute x from [B, T, C_in, H, W] → [B, C_in, T, H, W] before feeding into Conv3d.
        # ------------------------
        def conv3d_block(in_ch, out_ch, ker=(3,3,3), stride=(1,2,2), pad=(1,1,1)):
            layers = [
                nn.Conv3d(in_ch, out_ch, kernel_size=ker, stride=stride, padding=pad),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if dropout > 0:
                layers.append(nn.Dropout3d(p=dropout))
            return nn.Sequential(*layers)

        # enc1: [B, in_ch, T, H, W] → [B, base_ch, T, H/2, W/2]
        self.enc1 = conv3d_block(in_ch=in_channels, out_ch=base_channels)

        # enc2: [B, base_ch, T, H/2, W/2] → [B, base_ch*2, T, H/4, W/4]
        self.enc2 = conv3d_block(in_ch=base_channels, out_ch=base_channels * 2)

        # ------------------------
        # Bottleneck ConvLSTM: processes sequence of T frames at 1/4 resolution.
        # ------------------------
        self.conv_lstm = ConvLSTMCell(
            input_dim=base_channels * 2,
            hidden_dim=lstm_channels,
            kernel_size=3
        )

        # ------------------------
        # Decoder: two 3D transpose conv blocks to upsample back to full resolution.
        #   We treat the final ConvLSTM hidden state as a “1‐frame” volume: [B, lstm_ch, 1, H/4, W/4].
        # ------------------------
        def deconv3d_block(in_ch, out_ch, ker=(3,3,3), stride=(1,2,2),
                           pad=(1,1,1), out_pad=(0,1,1)):
            layers = [
                nn.ConvTranspose3d(
                    in_ch, out_ch,
                    kernel_size=ker, stride=stride,
                    padding=pad, output_padding=out_pad
                ),
                nn.BatchNorm3d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            if dropout > 0:
                layers.append(nn.Dropout3d(p=dropout))
            return nn.Sequential(*layers)

        # dec1: [B, lstm_ch, 1, H/4, W/4] → [B, hidden_ch, 1, H/2, W/2]
        self.dec1 = deconv3d_block(in_ch=lstm_channels, out_ch=hidden_channels)

        # dec2: will take concatenation of [dec1, skip2] as input:
        #   in_ch = hidden_ch + (base_ch*2), out_ch = hidden_ch // 2
        self.dec2 = deconv3d_block(
            in_ch=hidden_channels + (base_channels * 2),
            out_ch=hidden_channels // 2
        )

        # final_conv: will take concatenation of [dec2, skip1]:
        #   in_ch = (hidden_ch // 2) + base_ch, out_ch = 2
        self.final_conv = nn.Conv3d(
            in_channels=(hidden_channels // 2) + base_channels,
            out_channels=2,
            kernel_size=(1,1,1),
            stride=1,
            padding=0
        )

        # ------------------------
        # Spatio-Temporal Attention (optional)
        #   Produces a mask [B,1,1,H,W] via 3D conv → Sigmoid, to modulate final features.
        # ------------------------
        self.attn_conv = nn.Sequential(
            nn.Conv3d(
                in_channels=(hidden_channels // 2) + base_channels,
                out_channels=1,
                kernel_size=(1,1,1),
                stride=1,
                padding=0
            ),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: tensor of shape [B, T, C_in, H, W]. If x.ndim == 4, we assume T=1 and do x = x.unsqueeze(1).

        Returns:
            out: [B, 2, H, W], where channel 0 = precipitation and channel 1 = temperature.
        """
        # 1) Guarantee that x has 5 dims: [B, T, C_in, H, W]
        if x.ndim == 4:
            # If loader gave [B, C_in, H, W], add time‐axis of length 1
            x = x.unsqueeze(1)  # → [B, 1, C_in, H, W]
        elif x.ndim == 5:
            # Possibly loader gave [B, C_in, H, W, T]; detect and permute:
            B, d1, d2, d3, d4 = x.shape
            if d1 == self.enc1[0].in_channels and d4 == self.seq_length:
                # x is [B, C_in, H, W, T] → permute to [B, T, C_in, H, W]
                x = x.permute(0, 4, 1, 2, 3)
            # Else assume x is already [B, T, C_in, H, W]
        else:
            raise ValueError(f"Expected 4D or 5D input, but got x.ndim={x.ndim}")

        B, T, C_in, H, W = x.shape
        # Permute for 3D conv: [B, T, C_in, H, W] → [B, C_in, T, H, W]
        x = x.permute(0, 2, 1, 3, 4)

        # ----------------
        # Encoder
        # ----------------
        e1 = self.enc1(x)
        # e1: [B, base_ch, T, H/2, W/2]

        e2 = self.enc2(e1)
        # e2: [B, base_ch*2, T, H/4, W/4]

        # ----------------
        # Bottleneck ConvLSTM
        # ----------------
        B_e2, C_e2, T_e2, H2, W2 = e2.shape

        device = x.device
        h = torch.zeros(B_e2, self.conv_lstm.hidden_dim, H2, W2, device=device)
        c = torch.zeros(B_e2, self.conv_lstm.hidden_dim, H2, W2, device=device)

        for t in range(T_e2):
            x_t = e2[:, :, t, :, :]  # [B, C_e2, H/4, W/4]
            h, c = self.conv_lstm(x_t, (h, c))
        # Now h: [B, lstm_ch, H/4, W/4]

        # ----------------
        # Decoder
        # ----------------
        bottleneck = h.unsqueeze(2)
        # → [B, lstm_ch, 1, H/4, W/4]

        # Dec1: upsample to [B, hidden_ch, 1, H/2, W/2]
        d1 = self.dec1(bottleneck)

        # Skip‐connection from e2: last time slice, spatially upsample
        skip2 = e2[:, :, -1:, :, :]  # [B, base_ch*2, 1, H/4, W/4]

        # Temporarily disable deterministic algorithms for this interpolation only
        prev_flag = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)  # turn off deterministic‐only for interpolation
        skip2_upsampled = F.interpolate(
            skip2,
            size=(1, H // 2, W // 2),
            mode='trilinear',
            align_corners=False
        )
        torch.use_deterministic_algorithms(prev_flag)  # restore original flag
        # skip2_upsampled: [B, base_ch*2, 1, H/2, W/2]

        d1_cat = torch.cat([d1, skip2_upsampled], dim=1)
        # → [B, hidden_ch + (base_ch*2), 1, H/2, W/2]

        # Dec2: upsample to [B, hidden_ch//2, 1, H, W]
        d2 = self.dec2(d1_cat)

        # Skip‐connection from e1: last time slice, upsample to full res
        skip1 = e1[:, :, -1:, :, :]  # [B, base_ch, 1, H/2, W/2]

        prev = torch.are_deterministic_algorithms_enabled()

        # 2) Disable deterministic‐only for the upcoming interpolate
        torch.use_deterministic_algorithms(False)
        
        out = F.interpolate(
            skip1,
            size=(1, H//2, W//2),
            mode='trilinear',
            align_corners=False
        )
        
        # 3) Restore the previous deterministic setting
        torch.use_deterministic_algorithms(prev)
        
        prev_flag = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(False)
        skip1_upsampled = F.interpolate(
            skip1,
            size=(1, H, W),
            mode='trilinear',
            align_corners=False
        )
        torch.use_deterministic_algorithms(prev_flag)
        # skip1_upsampled: [B, base_ch, 1, H, W]

        d2_cat = torch.cat([d2, skip1_upsampled], dim=1)
        # → [B, (hidden_ch//2) + base_ch, 1, H, W]

        # ----------------
        # Spatio-Temporal Attention
        # ----------------
        attn_map = self.attn_conv(d2_cat)  # [B, 1, 1, H, W], values in [0,1]
        d2_attn = d2_cat * attn_map         # broadcast multiply

        # ----------------
        # Final 3D conv → 2 channels (precip + temp)
        # ----------------
        out3d = self.final_conv(d2_attn)  # [B, 2, 1, H, W]
        out = out3d.squeeze(2)            # → [B, 2, H, W]

        return out