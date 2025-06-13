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
        hidden_dim=64, 
        spatial_depth=3,
        temporal_hidden_dim=128,
        dropout_rate=0.3
    ):
        super(SequentialClimateCNN, self).__init__()
        
        self.seq_length = seq_length
        self.hidden_dim = hidden_dim
        self.temporal_hidden_dim = temporal_hidden_dim
        
        # Spatial feature extractor (applied to each time step)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Residual blocks for spatial processing
        self.res_blocks = nn.ModuleList()
        current_dim = hidden_dim
        
        for i in range(spatial_depth):
            self.res_blocks.append(
                ResidualBlock(current_dim, current_dim, kernel_size=kernel_size)
            )
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(current_dim)
        
        # Temporal state processor (ConvLSTM or ConvGRU)
        self.temporal_cell = ConvLSTMCell(
            input_dim=current_dim,
            hidden_dim=temporal_hidden_dim,
            kernel_size=3,
            bias=True
        )
        
        # Decoder for final prediction
        self.decoder = nn.Sequential(
            nn.Conv2d(temporal_hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(hidden_dim, n_output_channels, kernel_size=1)
        )

    def forward(self, x):
        # Check if input is 4D or 5D and handle accordingly
        # if len(x.size()) == 4:
            # If 4D input [batch, channels, height, width], add sequence dimension
        batch_size, channels, height, width = x.size()
        #wtf is this shit reading sdlfldskfjdlskjfkljl
        # nInitialize hidden state and cell state
        h_t = torch.zeros(batch_size, self.temporal_hidden_dim, height, width, device=x.device)
        c_t = torch.zeros(batch_size, self.temporal_hidden_dim, height, width, device=x.device)
        
        # Process single time step
        x_t = x
        
        # Extract spatial features
        x_t = self.spatial_encoder(x_t)
        
        # Apply residual blocks
        for res_block in self.res_blocks:
            x_t = res_block(x_t)
        
        # Apply spatial attention
        x_t = self.spatial_attention(x_t)
        
        # Update temporal state
        h_t, c_t = self.temporal_cell(x_t, (h_t, c_t))
            
        # else:
        #     # Original code for 5D input
        #     batch_size, seq_len, channels, height, width = x.size()
            
        #     # Initialize hidden state and cell state
        #     h_t = torch.zeros(batch_size, self.temporal_hidden_dim, height, width, device=x.device)
        #     c_t = torch.zeros(batch_size, self.temporal_hidden_dim, height, width, device=x.device)
            
        #     # Process each time step sequentially
        #     for t in range(seq_len):
        #         # Extract current time step
        #         x_t = x[:, t]
                
        #         # Extract spatial features
        #         x_t = self.spatial_encoder(x_t)
                
        #         # Apply residual blocks
        #         for res_block in self.res_blocks:
        #             x_t = res_block(x_t)
                
        #         # Apply spatial attention
        #         x_t = self.spatial_attention(x_t)
                
        #         # Update temporal state
        #         h_t, c_t = self.temporal_cell(x_t, (h_t, c_t))
        
        # Final prediction using the last hidden state
        output = self.decoder(h_t)
        
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