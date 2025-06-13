import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributed(nn.Module):
    """Applies a module over multiple time steps"""
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        # Reshape to (batch_size * time_steps, ...)
        batch_size, time_steps = x.size(0), x.size(1)
        x_reshaped = x.contiguous().view(batch_size * time_steps, *x.size()[2:])
        
        # Apply module
        y = self.module(x_reshaped)
        
        # Reshape back
        return y.contiguous().view(batch_size, time_steps, *y.size()[1:])

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for focusing on important regions"""
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        
    def forward(self, x):
        # Generate attention map
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class TemporalAttention(nn.Module):
    """Temporal attention mechanism for focusing on important time steps"""
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: [batch, time_steps, features]
        # Generate attention weights
        attention_weights = F.softmax(self.fc(x), dim=1)
        # Apply attention
        context = torch.sum(x * attention_weights, dim=1)
        return context

class ResidualBlock(nn.Module):
    """Residual block with larger kernels for climate patterns"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 
                              padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(identity)
        return self.act(out)

class TimeSeriesClimateCNN(nn.Module):
    """
    Time Series CNN for climate emulation
    Combines temporal processing with spatial CNN features
    """
    def __init__(
        self, 
        n_input_channels, 
        n_output_channels, 
        seq_length=12,
        kernel_size=5, 
        init_dim=64, 
        depth=4, 
        dropout_rate=0.3,
        use_lstm=True
    ):
        super(TimeSeriesClimateCNN, self).__init__()
        
        self.seq_length = seq_length
        self.use_lstm = use_lstm
        
        # Initial convolution applied to each time step
        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, 
                     padding=kernel_size//2),
            nn.BatchNorm2d(init_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Time-distributed CNN blocks
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim
        
        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(
                TimeDistributed(ResidualBlock(current_dim, out_dim))
            )
            if i < depth - 1:
                current_dim *= 2
        
        # Spatial attention after CNN processing
        self.spatial_attention = TimeDistributed(SpatialAttention(current_dim))
        
        # Global average pooling to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # LSTM for temporal processing
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=current_dim,
                hidden_size=current_dim*2,
                num_layers=2,
                batch_first=True,
                dropout=dropout_rate if depth > 1 else 0
            )
            lstm_output_dim = current_dim*2
        else:
            lstm_output_dim = current_dim
            
        # Temporal attention
        self.temporal_attention = TemporalAttention(lstm_output_dim)
        
        # Final prediction layers
        self.decoder = nn.Sequential(
            nn.Conv2d(lstm_output_dim, current_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(current_dim, current_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(current_dim//2, n_output_channels, kernel_size=1)
        )

    def forward(self, x):
        # batch_size, seq_len, channels, height, width = x.size()
        batch_size, channels, height, width = x.size()
        
        # Apply initial convolution to each time step
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.initial_conv(x)
        x = x.view(batch_size, seq_len, -1, height, width)
        
        # Apply residual blocks with time distribution
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        # Global average pooling to reduce spatial dimensions
        x = x.view(batch_size * seq_len, -1, height, width)
        x = self.global_avg_pool(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Apply LSTM for temporal processing
        if self.use_lstm:
            x, _ = self.lstm(x)
        
        # Apply temporal attention to focus on important time steps
        x = self.temporal_attention(x)
        
        # Reshape for decoder
        x = x.view(batch_size, -1, 1, 1)
        
        # Expand spatially to match original dimensions
        x = x.expand(-1, -1, height, width)
        
        # Final prediction
        x = self.decoder(x)
        
        return x

class ClimateTransformerCNN(nn.Module):
    """
    Transformer-based model for climate emulation
    Combines self-attention for temporal processing with CNN for spatial features
    """
    def __init__(
        self, 
        n_input_channels, 
        n_output_channels, 
        seq_length=12,
        kernel_size=5, 
        init_dim=64, 
        depth=4, 
        n_heads=8,
        dropout_rate=0.3
    ):
        super(ClimateTransformerCNN, self).__init__()
        
        self.seq_length = seq_length
        
        # Initial convolution applied to each time step
        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=kernel_size, 
                     padding=kernel_size//2),
            nn.BatchNorm2d(init_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Time-distributed CNN blocks
        self.res_blocks = nn.ModuleList()
        current_dim = init_dim
        
        for i in range(depth):
            out_dim = current_dim * 2 if i < depth - 1 else current_dim
            self.res_blocks.append(
                TimeDistributed(ResidualBlock(current_dim, out_dim))
            )
            if i < depth - 1:
                current_dim *= 2
        
        # Spatial attention
        self.spatial_attention = TimeDistributed(SpatialAttention(current_dim))
        
        # Global average pooling to reduce spatial dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Positional encoding for transformer
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, seq_length, current_dim)
        )
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=current_dim,
            nhead=n_heads,
            dim_feedforward=current_dim*4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        
        # Transformer encoder
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=2
        )
        
        # Final prediction layers
        self.decoder = nn.Sequential(
            nn.Conv2d(current_dim, current_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(current_dim, current_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(current_dim//2, n_output_channels, kernel_size=1)
        )

    def forward(self, x):
        # batch_size, seq_len, channels, height, width = x.size()
        batch_size, channels, height, width = x.size()
        
        # Apply initial convolution to each time step
        x = x.view(batch_size * seq_len, channels, height, width)
        x = self.initial_conv(x)
        x = x.view(batch_size, seq_len, -1, height, width)
        
        # Apply residual blocks with time distribution
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply spatial attention
        x = self.spatial_attention(x)
        
        # Global average pooling to reduce spatial dimensions
        x = x.view(batch_size * seq_len, -1, height, width)
        x = self.global_avg_pool(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Take the last time step's output
        x = x[:, -1]
        
        # Reshape for decoder
        x = x.view(batch_size, -1, 1, 1)
        
        # Expand spatially to match original dimensions
        x = x.expand(-1, -1, height, width)
        
        # Final prediction
        x = self.decoder(x)
        
        return x 

class SequentialClimateCNN(nn.Module):
    """
    Sequential Climate CNN that processes data month by month
    while maintaining temporal state between time steps
    """
    def __init__(
        self, 
        n_input_channels, 
        n_output_channels, 
        seq_length=12,
        kernel_size=5, 
        hidden_dim=256,  # Increased from 64
        spatial_depth=8,  # Increased from 3
        temporal_hidden_dim=512,  # Increased from 128
        dropout_rate=0.3
    ):
        super(SequentialClimateCNN, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        
        # Multi-scale spatial feature extraction
        self.spatial_encoder = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_input_channels, hidden_dim//4, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_dim//4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(n_input_channels, hidden_dim//4, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim//4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(n_input_channels, hidden_dim//4, kernel_size=7, padding=3),
                nn.BatchNorm2d(hidden_dim//4),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(n_input_channels, hidden_dim//4, kernel_size=9, padding=4),
                nn.BatchNorm2d(hidden_dim//4),
                nn.LeakyReLU(0.2, inplace=True),
            )
        ])
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Residual blocks for spatial processing
        self.res_blocks = nn.ModuleList()
        current_dim = hidden_dim
        
        # First set of residual blocks
        for i in range(spatial_depth // 2):
            self.res_blocks.append(
                ResidualBlock(current_dim, current_dim, kernel_size=kernel_size)
            )
        
        # Increase channels midway
        self.mid_conv = nn.Sequential(
            nn.Conv2d(current_dim, current_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(current_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        current_dim *= 2
        
        # Second set of residual blocks with increased channels
        for i in range(spatial_depth // 2, spatial_depth):
            self.res_blocks.append(
                ResidualBlock(current_dim, current_dim, kernel_size=kernel_size)
            )
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(current_dim)
        
        # Global context module
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(current_dim, current_dim//8, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(current_dim//8, current_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Temporal state processor (ConvLSTM)
        self.temporal_cell = ConvLSTMCell(
            input_dim=current_dim,
            hidden_dim=temporal_hidden_dim,
            kernel_size=3,
            bias=True
        )
        
        # Secondary temporal processor for long-range dependencies
        self.temporal_cell2 = ConvLSTMCell(
            input_dim=temporal_hidden_dim,
            hidden_dim=temporal_hidden_dim,
            kernel_size=3,
            bias=True
        )
        
        # Climate-specific feature enhancement
        self.climate_enhancer = nn.Sequential(
            nn.Conv2d(temporal_hidden_dim, temporal_hidden_dim, kernel_size=3, padding=1, groups=4),  # Grouped convolution
            nn.BatchNorm2d(temporal_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(temporal_hidden_dim, temporal_hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(temporal_hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Separate decoders for precipitation and temperature (assuming 2-channel output)
        self.precip_decoder = nn.Sequential(
            nn.Conv2d(temporal_hidden_dim, temporal_hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(temporal_hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(temporal_hidden_dim//2, temporal_hidden_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(temporal_hidden_dim//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(temporal_hidden_dim//4, 1, kernel_size=1)
        )
        
        self.temp_decoder = nn.Sequential(
            nn.Conv2d(temporal_hidden_dim, temporal_hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(temporal_hidden_dim//2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(temporal_hidden_dim//2, temporal_hidden_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(temporal_hidden_dim//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(temporal_hidden_dim//4, 1, kernel_size=1)
        )
        
        # SSP-specific adaptation layers
        self.ssp_adaptation = nn.ModuleList([
            nn.Conv2d(temporal_hidden_dim, temporal_hidden_dim, kernel_size=1)
            for _ in range(5)  # For 5 different SSP scenarios
        ])
        
        # Final output layer to combine outputs if needed
        self.final_output = nn.Conv2d(2, n_output_channels, kernel_size=1)

    def forward(self, x, ssp_idx=0):
        batch_size, channels, height, width = x.size()
        
        # Initialize hidden states and cell states for both temporal cells
        h_t = torch.zeros(batch_size, self.temporal_hidden_dim, height, width, device=x.device)
        c_t = torch.zeros(batch_size, self.temporal_hidden_dim, height, width, device=x.device)
        h_t2 = torch.zeros(batch_size, self.temporal_hidden_dim, height, width, device=x.device)
        c_t2 = torch.zeros(batch_size, self.temporal_hidden_dim, height, width, device=x.device)
        
        # Process single time step
        x_t = x
        
        # Multi-scale feature extraction
        multi_scale_features = []
        for encoder in self.spatial_encoder:
            multi_scale_features.append(encoder(x_t))
        
        # Concatenate multi-scale features
        x_t = torch.cat(multi_scale_features, dim=1)
        
        # Apply feature fusion
        x_t = self.fusion(x_t)
        
        # Apply first set of residual blocks
        for i in range(len(self.res_blocks) // 2):
            x_t = self.res_blocks[i](x_t)
        
        # Apply mid-network channel expansion
        x_t = self.mid_conv(x_t)
        
        # Apply second set of residual blocks
        for i in range(len(self.res_blocks) // 2, len(self.res_blocks)):
            x_t = self.res_blocks[i](x_t)
        
        # Apply spatial attention
        x_t = self.spatial_attention(x_t)
        
        # Apply global context
        global_context = self.global_context(x_t)
        x_t = x_t * global_context
        
        # Update primary temporal state
        h_t, c_t = self.temporal_cell(x_t, (h_t, c_t))
        
        # Update secondary temporal state for long-range dependencies
        h_t2, c_t2 = self.temporal_cell2(h_t, (h_t2, c_t2))
        
        # Combine temporal states
        h_combined = h_t + h_t2
        
        # Apply climate-specific feature enhancement
        h_enhanced = self.climate_enhancer(h_combined)
        
        # Apply SSP-specific adaptation
        h_adapted = self.ssp_adaptation[min(ssp_idx, len(self.ssp_adaptation)-1)](h_enhanced)
        
        # Generate separate predictions for precipitation and temperature
        precip_output = self.precip_decoder(h_adapted)
        temp_output = self.temp_decoder(h_adapted)
        
        # Combine outputs
        combined_output = torch.cat([precip_output, temp_output], dim=1)
        
        # Final output processing
        output = self.final_output(combined_output)
        
        return output

class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM Cell that maintains spatial structure in hidden state
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        padding = kernel_size // 2
        
        # Gates: input, forget, cell, output
        self.gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )
        
    def forward(self, x, hidden_states):
        h_prev, c_prev = hidden_states
        
        # Concatenate input and previous hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Calculate gates
        gates = self.gates(combined)
        
        # Split gates
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_dim, dim=1)
        
        # Apply activations
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        # Update cell state
        c_next = forgetgate * c_prev + ingate * cellgate
        
        # Update hidden state
        h_next = outgate * torch.tanh(c_next)
        
        return h_next, c_next 