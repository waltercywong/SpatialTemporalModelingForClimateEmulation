import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvLSTMCell(nn.Module):
    """
    ConvLSTM cell for spatio-temporal processing.
    This processes both spatial and temporal information simultaneously.
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        
        padding = kernel_size // 2
        
        # Combined convolution for all gates
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,  # 4 gates: i, f, g, o
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, x, hidden_state):
        h_prev, c_prev = hidden_state
        
        # Concatenate input and hidden state
        combined = torch.cat([x, h_prev], dim=1)
        
        # Apply convolution for all gates
        combined_conv = self.conv(combined)
        
        # Split into 4 gates
        cc_i, cc_f, cc_g, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply gate functions
        i = torch.sigmoid(cc_i)      # Input gate
        f = torch.sigmoid(cc_f)      # Forget gate
        g = torch.tanh(cc_g)         # New content
        o = torch.sigmoid(cc_o)      # Output gate
        
        # Update cell state
        c_next = f * c_prev + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class MultiScaleSpatialBlock(nn.Module):
    """
    Multi-scale spatial processing block that captures patterns at different scales.
    Important for climate data where phenomena occur at various spatial scales.
    """
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super(MultiScaleSpatialBlock, self).__init__()
        
        self.branches = nn.ModuleList()
        branch_out_channels = out_channels // len(kernel_sizes)
        
        for kernel_size in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_out_channels, 
                         kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm2d(branch_out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.branches.append(branch)
        
        # Adjust final channels to match exactly
        self.final_channels = branch_out_channels * len(kernel_sizes)
        if self.final_channels != out_channels:
            self.channel_adjust = nn.Conv2d(self.final_channels, out_channels, kernel_size=1)
        else:
            self.channel_adjust = None
            
    def forward(self, x):
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x))
        
        # Concatenate all branches
        out = torch.cat(branch_outputs, dim=1)
        
        # Adjust channels if necessary
        if self.channel_adjust is not None:
            out = self.channel_adjust(out)
            
        return out

class SpatialChannelAttention(nn.Module):
    """
    Combined spatial and channel attention mechanism.
    Helps the model focus on important regions and features.
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(SpatialChannelAttention, self).__init__()
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Channel attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights
        
        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)
        spatial_weights = self.spatial_attention(spatial_input)
        x = x * spatial_weights
        
        return x

class TemporalAttentionModule(nn.Module):
    """
    Temporal attention mechanism that learns to focus on important time steps.
    """
    def __init__(self, hidden_dim, num_heads=8):
        super(TemporalAttentionModule, self).__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.size()
        
        # Compute Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        attended = attended.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        output = self.out_proj(attended)
        
        return output, attention_weights

class ClimateEnhancedCRNN(nn.Module):
    """
    Enhanced CRNN for climate prediction with multi-scale spatial processing,
    ConvLSTM for spatio-temporal modeling, and attention mechanisms.
    
    Architecture Intuition:
    1. Multi-scale spatial processing captures climate patterns at different scales
    2. ConvLSTM preserves spatial structure while modeling temporal dynamics
    3. Attention mechanisms help focus on important regions and time steps
    4. Residual connections help with gradient flow in deep networks
    """
    
    def __init__(
        self,
        n_input_channels=5,
        n_output_channels=2,
        seq_length=12,
        init_dim=64,
        convlstm_layers=[128, 96, 64],  # Multiple ConvLSTM layers with decreasing size
        spatial_depths=[2, 2, 2],       # Depth for each scale
        kernel_sizes=[3, 5, 7],         # Multi-scale kernels
        attention_heads=8,
        dropout_rate=0.3,
        use_residual=True
    ):
        super(ClimateEnhancedCRNN, self).__init__()
        
        self.seq_length = seq_length
        self.init_dim = init_dim
        self.use_residual = use_residual
        
        # Initial spatial feature extraction with multi-scale processing
        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, init_dim, kernel_size=7, padding=3),
            nn.BatchNorm2d(init_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Multi-scale spatial processing blocks
        self.spatial_blocks = nn.ModuleList()
        current_dim = init_dim
        
        for i, depth in enumerate(spatial_depths):
            for j in range(depth):
                out_dim = convlstm_layers[0] if i == len(spatial_depths) - 1 and j == depth - 1 else current_dim
                
                block = nn.Sequential(
                    MultiScaleSpatialBlock(current_dim, out_dim, kernel_sizes),
                    SpatialChannelAttention(out_dim),
                    nn.Dropout2d(dropout_rate)
                )
                self.spatial_blocks.append(block)
                current_dim = out_dim
        
        # ConvLSTM layers for spatio-temporal processing
        self.convlstm_layers = nn.ModuleList()
        for i, hidden_dim in enumerate(convlstm_layers):
            input_dim = current_dim if i == 0 else convlstm_layers[i-1]
            self.convlstm_layers.append(
                ConvLSTMCell(input_dim, hidden_dim, kernel_size=3)
            )
        
        # Global temporal processing
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Temporal attention
        self.temporal_attention = TemporalAttentionModule(
            convlstm_layers[-1], num_heads=attention_heads
        )
        
        # Decoder for final prediction
        decoder_input_dim = convlstm_layers[-1]
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_input_dim, decoder_input_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_input_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(dropout_rate),
            
            nn.Conv2d(decoder_input_dim // 2, decoder_input_dim // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_input_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(decoder_input_dim // 4, n_output_channels, kernel_size=1)
        )
        
        # Residual projection if needed
        if use_residual and n_input_channels == n_output_channels:
            self.residual_proj = nn.Conv2d(n_input_channels, n_output_channels, kernel_size=1)
        else:
            self.residual_proj = None
    
    def forward(self, x):
        """
        Forward pass of the enhanced CRNN.
        
        Args:
            x: Input tensor of shape [batch, channels, height, width]
               This represents a single time step that gets expanded to a sequence
        
        Returns:
            Prediction tensor of shape [batch, n_output_channels, height, width]
        """
        batch_size, channels, height, width = x.size()
        
        # Store original input for potential residual connection
        x_residual = x if self.residual_proj is not None else None
        
        # Initial spatial processing
        x = self.initial_conv(x)
        
        # Apply spatial blocks
        for block in self.spatial_blocks:
            identity = x
            x = block(x)
            # Residual connection if dimensions match
            if self.use_residual and x.shape == identity.shape:
                x = x + identity
        
        # Create temporal sequence by replicating across time
        # This simulates having a sequence of similar conditions
        x_seq = x.unsqueeze(1).repeat(1, self.seq_length, 1, 1, 1)
        
        # Process through ConvLSTM layers
        convlstm_outputs = []
        
        for t in range(self.seq_length):
            x_t = x_seq[:, t]  # Current time step
            
            # Process through ConvLSTM layers
            for i, convlstm_layer in enumerate(self.convlstm_layers):
                if t == 0:  # Initialize hidden states
                    h_t = torch.zeros(batch_size, convlstm_layer.hidden_dim, height, width, device=x.device)
                    c_t = torch.zeros(batch_size, convlstm_layer.hidden_dim, height, width, device=x.device)
                    # Store initial states
                    if i == 0:
                        hidden_states = [(h_t, c_t)]
                    else:
                        hidden_states.append((h_t, c_t))
                
                # Update hidden state
                h_t, c_t = convlstm_layer(x_t, hidden_states[i])
                hidden_states[i] = (h_t, c_t)
                x_t = h_t  # Output becomes input to next layer
            
            convlstm_outputs.append(x_t)
        
        # Stack temporal outputs
        temporal_features = torch.stack(convlstm_outputs, dim=1)  # [batch, seq_len, channels, H, W]
        
        # Global pooling for temporal attention
        temporal_pooled = self.global_pool(temporal_features.view(-1, *temporal_features.shape[2:]))
        temporal_pooled = temporal_pooled.view(batch_size, self.seq_length, -1)
        
        # Apply temporal attention
        attended_temporal, attention_weights = self.temporal_attention(temporal_pooled)
        
        # Use the last time step's attended features
        final_temporal_features = attended_temporal[:, -1]  # [batch, channels]
        
        # Reshape back to spatial dimensions
        final_temporal_features = final_temporal_features.view(
            batch_size, -1, 1, 1
        ).expand(-1, -1, height, width)
        
        # Decode to final prediction
        output = self.decoder(final_temporal_features)
        
        # Add residual connection if applicable
        if self.residual_proj is not None:
            output = output + self.residual_proj(x_residual)
        
        return output
    
    def get_attention_weights(self, x):
        """
        Extract attention weights for analysis.
        Useful for understanding what the model focuses on.
        """
        with torch.no_grad():
            # Run forward pass and extract attention weights
            _ = self.forward(x)
            # Note: You'd need to modify forward to return attention weights
            # This is a placeholder for attention visualization
        return None

# Example of usage and model size analysis
if __name__ == "__main__":
    # Test the model
    batch_size = 4
    n_input_channels = 5  # CO2, SO2, CH4, BC, rsdt
    n_output_channels = 2  # tas, pr
    height, width = 48, 72  # Based on the zarr data structure
    
    # Create model
    model = ClimateEnhancedCRNN(
        n_input_channels=n_input_channels,
        n_output_channels=n_output_channels,
        seq_length=12,
        init_dim=64,
        convlstm_layers=[128, 96, 64],
        spatial_depths=[2, 2, 2],
        kernel_sizes=[3, 5, 7],
        attention_heads=8,
        dropout_rate=0.3
    )
    
    # Test input
    test_input = torch.randn(batch_size, n_input_channels, height, width)
    
    # Forward pass
    output = model(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}") 