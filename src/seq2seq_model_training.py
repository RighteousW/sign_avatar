import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import random
import json
import math
from typing import Dict, List, Tuple, Optional

from constants import (
    LANDMARKS_DIR,
    LANDMARKS_DIR_METADATA_PKL,
    MODELS_TRAINED_DIR,
    SEQ2SEQ_CONFIG,
)


class GestureTransformer(nn.Module):
    """Clean transformer model for gesture sequence generation"""
    
    def __init__(
        self,
        feature_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 50
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Input/Output projections
        self.input_projection = nn.Linear(feature_dim, d_model)
        self.output_projection = nn.Linear(d_model, feature_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(d_model)
        
    def generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        """Generate mask to prevent looking ahead in sequence"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self, 
        src: torch.Tensor, 
        tgt: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Source sequence [batch_size, src_seq_len, feature_dim]
            tgt: Target sequence [batch_size, tgt_seq_len, feature_dim] 
            src_mask: Optional source mask
            tgt_mask: Optional target mask (causal mask for generation)
        """
        # Project to model dimension
        src_embedded = self.input_projection(src)
        tgt_embedded = self.input_projection(tgt)
        
        # Add positional encoding
        src_embedded = self.pos_encoding(src_embedded)
        tgt_embedded = self.pos_encoding(tgt_embedded)
        
        # Apply layer norm
        src_embedded = self.norm(src_embedded)
        tgt_embedded = self.norm(tgt_embedded)
        
        # Generate causal mask for target if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(
            src=src_embedded,
            tgt=tgt_embedded,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        
        # Project back to feature space
        output = self.output_projection(output)
        
        return output
    
    def generate_sequence(
        self, 
        start_frame: torch.Tensor,
        end_frame: torch.Tensor, 
        sequence_length: int,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """Generate interpolated sequence between start and end frames"""
        self.eval()
        device = start_frame.device
        batch_size = start_frame.size(0)
        
        # Use start and end frames as source context
        src = torch.stack([start_frame, end_frame], dim=1)  # [batch, 2, feature_dim]
        
        # Initialize target sequence with start frame
        generated = torch.zeros(batch_size, sequence_length, self.feature_dim).to(device)
        generated[:, 0] = start_frame
        
        # Generate sequence autoregressively
        with torch.no_grad():
            for i in range(1, sequence_length):
                # Current partial sequence
                current_seq = generated[:, :i]
                
                # Forward pass
                output = self.forward(src, current_seq)
                
                # Get next token prediction
                next_token = output[:, -1]  # Last position
                
                # Apply temperature for diversity
                if temperature != 1.0:
                    next_token = next_token / temperature
                
                generated[:, i] = next_token
        
        # Ensure end frame matches target (if sequence is long enough)
        if sequence_length > 1:
            generated[:, -1] = end_frame
            
        return generated


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings"""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class LandmarkProcessor:
    """Clean processor for gesture landmark data"""
    
    def __init__(self, metadata_path: str = None):
        self.metadata = self._load_metadata(metadata_path)
        self.feature_info = self.metadata.get("feature_info", {})
        self.feature_dim = self.feature_info.get("total_features", 333)
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        print(f"LandmarkProcessor initialized:")
        print(f"  Feature dimension: {self.feature_dim}")
        print(f"  Landmark types: {self.metadata.get('landmark_types', [])}")
    
    def _load_metadata(self, metadata_path: str = None) -> Dict:
        """Load extraction metadata"""
        if metadata_path is None:
            metadata_path = LANDMARKS_DIR_METADATA_PKL
            
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata not found at {metadata_path}. "
                "Run landmark extraction first."
            )
        
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
            
        print(f"Loaded metadata from: {metadata_path}")
        return metadata
    
    def extract_features(self, frame_data: Dict) -> np.ndarray:
        """Extract feature vector from frame data"""
        features = np.zeros(self.feature_dim, dtype=np.float32)
        current_idx = 0
        
        # Extract hand landmarks
        if "hands" in frame_data:
            hand_landmarks_dim = self.feature_info.get("hand_landmarks", 126)
            hand_features = self._extract_hand_features(frame_data["hands"])
            
            if len(hand_features) > 0:
                end_idx = current_idx + min(hand_landmarks_dim, len(hand_features))
                features[current_idx:end_idx] = hand_features[:end_idx-current_idx]
                current_idx = end_idx
        
        # Extract pose landmarks
        if "pose" in frame_data:
            pose_landmarks_dim = self.feature_info.get("pose_landmarks", 132)
            pose_features = self._extract_pose_features(frame_data["pose"])
            
            if len(pose_features) > 0:
                end_idx = current_idx + min(pose_landmarks_dim, len(pose_features))
                features[current_idx:end_idx] = pose_features[:end_idx-current_idx]
        
        return features
    
    def _extract_hand_features(self, hands_data: List) -> np.ndarray:
        """Extract hand landmark features"""
        max_hands = 2
        landmarks_per_hand = 21
        coords_per_landmark = 3
        
        hand_features = np.zeros(max_hands * landmarks_per_hand * coords_per_landmark)
        
        for i, hand_data in enumerate(hands_data[:max_hands]):
            landmarks = hand_data.get("landmarks", [])
            start_idx = i * landmarks_per_hand * coords_per_landmark
            
            for j, landmark in enumerate(landmarks[:landmarks_per_hand]):
                landmark_idx = start_idx + j * coords_per_landmark
                if landmark_idx + coords_per_landmark <= len(hand_features):
                    hand_features[landmark_idx:landmark_idx + coords_per_landmark] = landmark[:coords_per_landmark]
        
        return hand_features
    
    def _extract_pose_features(self, pose_data: Dict) -> np.ndarray:
        """Extract pose landmark features"""
        landmarks = pose_data.get("landmarks", [])
        max_landmarks = 33
        coords_per_landmark = 4
        
        pose_features = np.zeros(max_landmarks * coords_per_landmark)
        
        for i, landmark in enumerate(landmarks[:max_landmarks]):
            start_idx = i * coords_per_landmark
            if start_idx + coords_per_landmark <= len(pose_features):
                pose_features[start_idx:start_idx + coords_per_landmark] = landmark[:coords_per_landmark]
        
        return pose_features
    
    def fit_scaler(self, landmark_files: List[Path], sample_size: int = 2000):
        """Fit the feature scaler on sample data"""
        print(f"Fitting scaler on sample of {sample_size} frames...")
        
        sample_features = []
        files_to_sample = min(len(landmark_files), sample_size // 10)
        sampled_files = random.sample(landmark_files, files_to_sample)
        
        for pkl_file in sampled_files:
            try:
                with open(pkl_file, 'rb') as f:
                    landmark_data = pickle.load(f)
                
                frames = landmark_data.get("frames", [])
                for frame_data in frames[:10]:  # Max 10 frames per file
                    features = self.extract_features(frame_data)
                    sample_features.append(features)
                    
                    if len(sample_features) >= sample_size:
                        break
                
                if len(sample_features) >= sample_size:
                    break
                    
            except Exception as e:
                print(f"Warning: Error processing {pkl_file}: {e}")
                continue
        
        if len(sample_features) == 0:
            raise ValueError("No valid frames found for fitting scaler")
        
        sample_features = np.array(sample_features)
        self.scaler.fit(sample_features)
        self.is_fitted = True
        
        print(f"Scaler fitted on {len(sample_features)} frames")
        print(f"Feature statistics: mean={sample_features.mean():.3f}, std={sample_features.std():.3f}")
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features using fitted scaler"""
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before normalization")
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
            return self.scaler.transform(features).squeeze()
        else:
            return self.scaler.transform(features)


class GestureSequenceDataset(Dataset):
    """Dataset for gesture sequence generation"""
    
    def __init__(
        self,
        landmark_files: List[Path],
        processor: LandmarkProcessor,
        sequence_length: int = 8,
        gap_size: int = 4,
        max_samples: int = 10000
    ):
        self.landmark_files = landmark_files
        self.processor = processor
        self.sequence_length = sequence_length
        self.gap_size = gap_size
        self.max_samples = max_samples
        
        # Pre-load all valid sequences
        self.sequences = []
        self._load_sequences()
        
        print(f"GestureSequenceDataset ready with {len(self.sequences)} sequences")
    
    def _load_sequences(self):
        """Load all valid sequences from landmark files"""
        print("Loading sequences from landmark files...")
        
        for i, pkl_file in enumerate(self.landmark_files[:500]):  # Limit files for memory
            if i % 100 == 0:
                print(f"  Processing file {i+1}/{min(500, len(self.landmark_files))}")
            
            try:
                with open(pkl_file, 'rb') as f:
                    landmark_data = pickle.load(f)
                
                frames = landmark_data.get("frames", [])
                if len(frames) < self.sequence_length:
                    continue
                
                # Extract features for all frames
                frame_features = []
                for frame_data in frames:
                    try:
                        features = self.processor.extract_features(frame_data)
                        normalized = self.processor.normalize_features(features)
                        frame_features.append(normalized)
                    except Exception as e:
                        continue
                
                if len(frame_features) < self.sequence_length:
                    continue
                
                # Create sequences from this video
                frame_features = np.array(frame_features)
                max_start = len(frame_features) - self.sequence_length
                
                # Sample multiple sequences per video
                num_sequences = min(5, max_start + 1)
                if max_start > 0:
                    start_indices = np.random.choice(max_start + 1, size=num_sequences, replace=False)
                else:
                    start_indices = [0]
                
                for start_idx in start_indices:
                    sequence = frame_features[start_idx:start_idx + self.sequence_length]
                    
                    # Create source (start and end frames) and target (full sequence)
                    src_sequence = np.stack([sequence[0], sequence[-1]], axis=0)
                    
                    self.sequences.append({
                        'src': src_sequence,
                        'tgt': sequence,
                        'start_frame': sequence[0],
                        'end_frame': sequence[-1]
                    })
                    
                    if len(self.sequences) >= self.max_samples:
                        print(f"Reached max samples ({self.max_samples})")
                        return
                        
            except Exception as e:
                print(f"Warning: Error processing {pkl_file}: {e}")
                continue
        
        print(f"Loaded {len(self.sequences)} sequences")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sequences[idx]
        
        return {
            'src': torch.FloatTensor(sample['src']),
            'tgt': torch.FloatTensor(sample['tgt']),
            'start_frame': torch.FloatTensor(sample['start_frame']),
            'end_frame': torch.FloatTensor(sample['end_frame'])
        }


class GestureTransformerTrainer:
    """Trainer for gesture transformer model"""
    
    def __init__(
        self,
        model: GestureTransformer,
        device: str = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4
    ):
        self.model = model
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=1000,  # Will be updated in train()
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()
        
        # Training history
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': []
        }
        
        print(f"GestureTransformerTrainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            src = batch['src'].to(self.device)
            tgt = batch['tgt'].to(self.device)
            
            # Prepare decoder input (target shifted by one position)
            tgt_input = tgt[:, :-1]  # All but last
            tgt_output = tgt[:, 1:]  # All but first
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(src, tgt_input)
            
            # Calculate loss
            loss = self.criterion(output, tgt_output)
            
            # Add temporal consistency loss
            temporal_loss = self._temporal_consistency_loss(output)
            total_loss_batch = loss + 0.1 * temporal_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src'].to(self.device)
                tgt = batch['tgt'].to(self.device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = self.model(src, tgt_input)
                loss = self.criterion(output, tgt_output)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _temporal_consistency_loss(self, sequences: torch.Tensor) -> torch.Tensor:
        """Compute temporal smoothness loss"""
        if sequences.size(1) < 3:
            return torch.tensor(0.0, device=sequences.device)
        
        # Compute first and second derivatives
        first_diff = sequences[:, 1:] - sequences[:, :-1]
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        
        # Penalize large second derivatives (acceleration)
        return torch.mean(torch.norm(second_diff, dim=2))
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        patience: int = 10
    ):
        """Train the model"""
        print(f"Starting training for {num_epochs} epochs...")
        
        # Update scheduler total steps
        self.scheduler.total_steps = num_epochs * len(train_loader)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update history
            self.history['train_losses'].append(train_loss)
            self.history['val_losses'].append(val_loss)
            self.history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            
            # Logging
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Train Loss = {train_loss:.4f}, "
                      f"Val Loss = {val_loss:.4f}, "
                      f"LR = {self.scheduler.get_last_lr()[0]:.2e}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    def save_model(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history
        }
        
        save_path = Path(MODELS_TRAINED_DIR) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")
    
    def plot_training_curves(self):
        """Plot training progress"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        epochs = range(1, len(self.history['train_losses']) + 1)
        ax1.plot(epochs, self.history['train_losses'], label='Train Loss', alpha=0.8)
        ax1.plot(epochs, self.history['val_losses'], label='Val Loss', alpha=0.8)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Progress')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Learning rate
        ax2.plot(epochs, self.history['learning_rates'], alpha=0.8)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        plt.show()


def get_landmark_files(landmarks_dir: Path) -> List[Path]:
    """Get all landmark pickle files"""
    landmark_files = []
    
    for folder in landmarks_dir.iterdir():
        if folder.is_dir():
            pkl_files = list(folder.glob("*_landmarks.pkl"))
            landmark_files.extend(pkl_files)
    
    print(f"Found {len(landmark_files)} landmark files")
    return landmark_files


def create_config() -> Dict:
    """Create training configuration"""
    config = {
        'feature_dim': 333,  # Will be updated from metadata
        'd_model': 256,
        'nhead': 8,
        'num_encoder_layers': 4,
        'num_decoder_layers': 4,
        'sequence_length': 8,
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 50,
        'val_split': 0.2,
        'max_samples': 10000
    }
    return config


def main():
    """Main training pipeline"""
    print("=== Clean Transformer Gesture Sequence Generation ===")
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    try:
        # Load processor with metadata
        processor = LandmarkProcessor()
        
        # Create config and update feature dimension
        config = create_config()
        config['feature_dim'] = processor.feature_dim
        
        print(f"Configuration: {config}")
        
        # Get landmark files
        landmarks_dir = Path(LANDMARKS_DIR)
        landmark_files = get_landmark_files(landmarks_dir)
        
        if not landmark_files:
            raise ValueError("No landmark files found!")
        
        # Fit scaler
        processor.fit_scaler(landmark_files)
        
        # Create dataset
        dataset = GestureSequenceDataset(
            landmark_files=landmark_files,
            processor=processor,
            sequence_length=config['sequence_length'],
            max_samples=config['max_samples']
        )
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty!")
        
        # Train/validation split
        val_size = int(config['val_split'] * len(dataset))
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # Create model
        model = GestureTransformer(
            feature_dim=config['feature_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            num_decoder_layers=config['num_decoder_layers']
        )
        
        print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create trainer
        trainer = GestureTransformerTrainer(
            model=model,
            learning_rate=config['learning_rate']
        )
        
        # Train
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs']
        )
        
        # Plot results
        trainer.plot_training_curves()
        
        # Save final model and config
        trainer.save_model('final_transformer_model.pth')
        
        with open(Path(MODELS_TRAINED_DIR) / 'processor.pkl', 'wb') as f:
            pickle.dump(processor, f)
        
        with open(Path(MODELS_TRAINED_DIR) / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\nTraining complete! Models saved to {MODELS_TRAINED_DIR}")
        
        # Example generation
        print("\n=== Example Generation ===")
        sample = next(iter(val_loader))
        start_frame = sample['start_frame'][0:1]
        end_frame = sample['end_frame'][0:1]
        
        with torch.no_grad():
            generated = model.generate_sequence(
                start_frame.to(trainer.device),
                end_frame.to(trainer.device),
                sequence_length=config['sequence_length']
            )
        
        print(f"Generated sequence shape: {generated.shape}")
        print("Example generation successful!")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()