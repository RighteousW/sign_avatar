import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class GestureEncoder(nn.Module):
    """Encoder for gesture sequences"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        use_transformer: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_transformer = use_transformer

        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        if use_transformer:
            self.pos_encoding = PositionalEncoding(hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        else:
            self.lstm = nn.LSTM(
                hidden_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=True,
            )
            self.output_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, input_dim]
        Returns:
            encoded: [batch_size, seq_len, hidden_dim]
            context: [batch_size, hidden_dim] - context vector
        """
        # Project input
        x = self.input_projection(x)
        x = self.dropout(x)

        if self.use_transformer:
            x = self.pos_encoding(x)
            encoded = self.transformer(x)
            # Use mean pooling for context
            context = encoded.mean(dim=1)
        else:
            encoded, (h_n, c_n) = self.lstm(x)
            encoded = self.output_projection(encoded)
            # Use final hidden state as context
            context = h_n[-1]  # Last layer, forward direction

        return encoded, context


class AttentionDecoder(nn.Module):
    """Attention-based decoder for gesture generation"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        use_transformer: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_transformer = use_transformer

        # Input projection (for target conditioning)
        self.input_projection = nn.Linear(input_dim, hidden_dim)

        if use_transformer:
            self.pos_encoding = PositionalEncoding(hidden_dim)
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        else:
            self.lstm = nn.LSTM(
                hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout
            )

            # Attention mechanism
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads=8, dropout=dropout, batch_first=True
            )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, target_condition, encoder_output, context, target_length):
        """
        Args:
            target_condition: [batch_size, target_seq_len, input_dim] - target gesture frames
            encoder_output: [batch_size, source_seq_len, hidden_dim] - from encoder
            context: [batch_size, hidden_dim] - context vector
            target_length: int - length of sequence to generate
        Returns:
            output: [batch_size, target_length, output_dim]
        """
        batch_size = encoder_output.size(0)

        if self.use_transformer:
            # Create target sequence embeddings
            # Start with context, then use target conditioning
            target_input = self.input_projection(target_condition)
            target_input = self.pos_encoding(target_input)

            # Generate causal mask for decoder
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                target_length, device=encoder_output.device
            )

            # Transformer decoder
            decoded = self.transformer(target_input, encoder_output, tgt_mask=tgt_mask)
        else:
            # LSTM with attention
            target_input = self.input_projection(target_condition)
            decoded, _ = self.lstm(target_input)

            # Apply attention
            attended, _ = self.attention(decoded, encoder_output, encoder_output)
            decoded = decoded + attended

        # Project to output dimension
        output = self.output_projection(decoded)

        return output


class GestureSeq2Seq(nn.Module):
    """Complete Seq2Seq model for gesture interpolation"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_transformer: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Encoder for source gesture
        self.encoder = GestureEncoder(
            input_dim, hidden_dim, num_layers, use_transformer, dropout
        )

        # Decoder for generating interpolation
        self.decoder = AttentionDecoder(
            input_dim, hidden_dim, input_dim, num_layers, use_transformer, dropout
        )

        # Context fusion layer
        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        source_gesture,
        target_gesture,
        target_length: int,
        teacher_forcing_ratio: float = 0.5,
    ):
        """
        Args:
            source_gesture: [batch_size, source_len, input_dim]
            target_gesture: [batch_size, target_len, input_dim]
            target_length: length of interpolation to generate
            teacher_forcing_ratio: probability of using ground truth in training
        """
        batch_size = source_gesture.size(0)

        # Encode source gesture
        encoder_output, source_context = self.encoder(source_gesture)

        # Create target conditioning by combining context with target
        target_mean = target_gesture.mean(dim=1, keepdim=True)  # [batch, 1, input_dim]
        target_expanded = target_mean.expand(-1, target_length, -1)

        # Fuse source context with target information
        source_context_expanded = source_context.unsqueeze(1).expand(
            -1, target_length, -1
        )
        fused_input = torch.cat([source_context_expanded, target_expanded], dim=-1)
        target_condition = self.context_fusion(fused_input)

        # Decode interpolation
        output = self.decoder(
            target_condition, encoder_output, source_context, target_length
        )

        return output

    def generate(self, source_gesture, target_gesture, target_length: int):
        """Generate interpolation without teacher forcing"""
        self.eval()
        with torch.no_grad():
            return self.forward(source_gesture, target_gesture, target_length, 0.0)


class GestureLoss(nn.Module):
    """Custom loss function for gesture interpolation"""

    def __init__(
        self,
        mse_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        endpoint_weight: float = 2.0,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.smoothness_weight = smoothness_weight
        self.endpoint_weight = endpoint_weight

    def forward(self, predicted, target, source_end=None, target_start=None):
        """
        Args:
            predicted: [batch, seq_len, feature_dim]
            target: [batch, seq_len, feature_dim]
            source_end: [batch, 1, feature_dim] - last frame of source
            target_start: [batch, 1, feature_dim] - first frame of target
        """
        # Basic MSE loss
        mse_loss = F.mse_loss(predicted, target)

        # Smoothness loss (penalize large changes between consecutive frames)
        diff = predicted[:, 1:] - predicted[:, :-1]
        smoothness_loss = torch.mean(torch.sum(diff**2, dim=-1))

        # Endpoint consistency loss
        endpoint_loss = 0.0
        if source_end is not None:
            # First predicted frame should be close to end of source
            endpoint_loss += F.mse_loss(predicted[:, 0:1], source_end)
        if target_start is not None:
            # Last predicted frame should be close to start of target
            endpoint_loss += F.mse_loss(predicted[:, -1:], target_start)

        total_loss = (
            self.mse_weight * mse_loss
            + self.smoothness_weight * smoothness_loss
            + self.endpoint_weight * endpoint_loss
        )

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "smoothness_loss": smoothness_loss,
            "endpoint_loss": endpoint_loss,
        }


def create_model(feature_dim: int, config: Dict = None) -> GestureSeq2Seq:
    """Create and initialize the model"""
    if config is None:
        config = {
            "hidden_dim": 256,
            "num_layers": 3,
            "use_transformer": True,
            "dropout": 0.1,
        }

    model = GestureSeq2Seq(
        input_dim=feature_dim,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        use_transformer=config["use_transformer"],
        dropout=config["dropout"],
    )

    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if "weight" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    torch.nn.init.constant_(param, 0)

    model.apply(init_weights)
    return model


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    feature_dim = 335  # From your landmark extractor
    model = create_model(feature_dim)

    # Test forward pass
    batch_size = 4
    source_len = 5
    target_len = 3
    interpolation_len = 15

    source_gesture = torch.randn(batch_size, source_len, feature_dim)
    target_gesture = torch.randn(batch_size, target_len, feature_dim)
    ground_truth = torch.randn(batch_size, interpolation_len, feature_dim)

    # Forward pass
    output = model(source_gesture, target_gesture, interpolation_len)
    print(f"Output shape: {output.shape}")

    # Test loss
    loss_fn = GestureLoss()
    loss_dict = loss_fn(output, ground_truth)
    print(f"Loss: {loss_dict['total_loss'].item():.4f}")

    print("Model test successful!")
