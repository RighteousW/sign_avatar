import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
import warnings

warnings.filterwarnings("ignore")


class GestureSequenceSimilarityMetrics:
    """Comprehensive similarity metrics specifically for gesture landmark sequences"""

    def __init__(self, feature_dim=333):
        self.feature_dim = feature_dim

        # Feature breakdown based on your landmark extractor
        self.hand_landmarks_dim = 21 * 3 * 2  # 126 (max 2 hands)
        self.hand_connections_dim = 20 * 2  # 40 (20 connections per hand, max 2 hands)
        self.pose_landmarks_dim = 33 * 4  # 132 (33 landmarks with x,y,z,visibility)
        self.pose_connections_dim = 35  # 35 pose connections

        # Feature indices for easy slicing
        self.hand_landmarks_start = 0
        self.hand_landmarks_end = self.hand_landmarks_dim
        self.hand_connections_start = self.hand_landmarks_end
        self.hand_connections_end = (
            self.hand_connections_start + self.hand_connections_dim
        )
        self.pose_landmarks_start = self.hand_connections_end
        self.pose_landmarks_end = self.pose_landmarks_start + self.pose_landmarks_dim
        self.pose_connections_start = self.pose_landmarks_end
        self.pose_connections_end = (
            self.pose_connections_start + self.pose_connections_dim
        )

        # Cache for statistical analysis
        self.real_sequences_cache = []
        self.fake_sequences_cache = []

    def to_numpy(self, tensor_or_array):
        """Convert tensor to numpy array if needed"""
        if isinstance(tensor_or_array, torch.Tensor):
            return tensor_or_array.detach().cpu().numpy()
        return tensor_or_array

    def extract_feature_components(self, sequences):
        """Extract different components of the feature vector"""
        sequences = self.to_numpy(sequences)

        # Handle both single sequences and batches
        if sequences.ndim == 2:  # Single sequence
            sequences = sequences.reshape(1, *sequences.shape)

        batch_size, seq_len, feat_dim = sequences.shape

        components = {
            "hand_landmarks": sequences[
                :, :, self.hand_landmarks_start : self.hand_landmarks_end
            ],
            "hand_connections": sequences[
                :, :, self.hand_connections_start : self.hand_connections_end
            ],
            "pose_landmarks": sequences[
                :, :, self.pose_landmarks_start : self.pose_landmarks_end
            ],
            "pose_connections": sequences[
                :, :, self.pose_connections_start : self.pose_connections_end
            ],
        }

        return components

    def compute_component_wise_mse(self, real_sequences, fake_sequences):
        """Compute MSE for each component separately"""
        real_comp = self.extract_feature_components(real_sequences)
        fake_comp = self.extract_feature_components(fake_sequences)

        mse_scores = {}
        for comp_name in real_comp.keys():
            mse = np.mean((real_comp[comp_name] - fake_comp[comp_name]) ** 2)
            mse_scores[f"{comp_name}_mse"] = mse

        # Overall MSE
        mse_scores["overall_mse"] = np.mean([mse_scores[k] for k in mse_scores.keys()])

        return mse_scores

    def compute_gesture_smoothness(self, sequences):
        """Measure smoothness specific to gesture data"""
        sequences = self.to_numpy(sequences)
        components = self.extract_feature_components(sequences)

        smoothness_scores = {}

        for comp_name, comp_data in components.items():
            # Calculate frame-to-frame differences
            diffs = np.diff(comp_data, axis=1)  # Shape: (batch, seq_len-1, feat_dim)

            # Average magnitude of changes per sequence
            smoothness_per_seq = np.mean(np.linalg.norm(diffs, axis=2), axis=1)

            smoothness_scores[f"{comp_name}_smoothness"] = {
                "mean": np.mean(smoothness_per_seq),
                "std": np.std(smoothness_per_seq),
                "median": np.median(smoothness_per_seq),
            }

        return smoothness_scores

    def compute_anatomical_plausibility(self, sequences):
        """Check if hand/pose configurations are anatomically plausible"""
        sequences = self.to_numpy(sequences)
        components = self.extract_feature_components(sequences)

        plausibility_scores = {}

        # Hand landmarks analysis
        hand_landmarks = components["hand_landmarks"]
        batch_size, seq_len, hand_dim = hand_landmarks.shape

        # Reshape to separate hands: (batch, seq_len, 2_hands, 21_landmarks, 3_coords)
        hand_landmarks_reshaped = hand_landmarks.reshape(batch_size, seq_len, 2, 21, 3)

        hand_span_scores = []
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                for hand_idx in range(2):
                    hand_points = hand_landmarks_reshaped[batch_idx, seq_idx, hand_idx]

                    # Skip if hand is not detected (all zeros)
                    if np.allclose(hand_points, 0):
                        continue

                    # Check hand span (distance between thumb and pinky)
                    thumb_tip = hand_points[4]  # Landmark 4 is thumb tip
                    pinky_tip = hand_points[20]  # Landmark 20 is pinky tip
                    hand_span = np.linalg.norm(thumb_tip - pinky_tip)
                    hand_span_scores.append(hand_span)

        plausibility_scores["hand_span"] = {
            "mean": np.mean(hand_span_scores) if hand_span_scores else 0,
            "std": np.std(hand_span_scores) if hand_span_scores else 0,
            "count": len(hand_span_scores),
        }

        # Hand connections analysis
        hand_connections = components["hand_connections"]
        connection_scores = hand_connections[
            hand_connections > 0
        ]  # Non-zero connections

        plausibility_scores["hand_connections"] = {
            "mean": np.mean(connection_scores) if len(connection_scores) > 0 else 0,
            "std": np.std(connection_scores) if len(connection_scores) > 0 else 0,
            "active_ratio": len(connection_scores)
            / (batch_size * seq_len * self.hand_connections_dim),
        }

        return plausibility_scores

    def compute_temporal_consistency(self, real_sequences, fake_sequences):
        """Measure how well fake sequences maintain temporal patterns from real data"""
        real_sequences = self.to_numpy(real_sequences)
        fake_sequences = self.to_numpy(fake_sequences)

        # Compute velocity patterns (frame-to-frame changes)
        real_velocities = np.diff(real_sequences, axis=1)
        fake_velocities = np.diff(fake_sequences, axis=1)

        # Compute acceleration patterns
        real_accelerations = np.diff(real_velocities, axis=1)
        fake_accelerations = np.diff(fake_velocities, axis=1)

        consistency_scores = {}

        # Velocity consistency
        velocity_mse = np.mean((real_velocities - fake_velocities) ** 2)
        consistency_scores["velocity_mse"] = velocity_mse

        # Acceleration consistency
        if real_accelerations.shape[1] > 0 and fake_accelerations.shape[1] > 0:
            acceleration_mse = np.mean((real_accelerations - fake_accelerations) ** 2)
            consistency_scores["acceleration_mse"] = acceleration_mse
        else:
            consistency_scores["acceleration_mse"] = 0.0

        # Temporal correlation
        correlations = []
        batch_size = real_sequences.shape[0]
        for i in range(batch_size):
            # Flatten sequences for correlation
            real_flat = real_sequences[i].flatten()
            fake_flat = fake_sequences[i].flatten()

            if len(real_flat) > 1 and len(fake_flat) > 1:
                corr = np.corrcoef(real_flat, fake_flat)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        consistency_scores["temporal_correlation"] = {
            "mean": np.mean(correlations) if correlations else 0,
            "std": np.std(correlations) if correlations else 0,
        }

        return consistency_scores

    def compute_feature_distribution_similarity(self, real_sequences, fake_sequences):
        """Compare statistical distributions of features"""
        real_sequences = self.to_numpy(real_sequences)
        fake_sequences = self.to_numpy(fake_sequences)

        # Flatten sequences for distribution analysis
        real_flat = real_sequences.reshape(-1, self.feature_dim)
        fake_flat = fake_sequences.reshape(-1, self.feature_dim)

        distribution_scores = {}

        # Mean and std comparison
        real_means = np.mean(real_flat, axis=0)
        fake_means = np.mean(fake_flat, axis=0)
        real_stds = np.std(real_flat, axis=0)
        fake_stds = np.std(real_flat, axis=0)

        distribution_scores["mean_difference"] = np.mean(
            np.abs(real_means - fake_means)
        )
        distribution_scores["std_difference"] = np.mean(np.abs(real_stds - fake_stds))

        # Per-component analysis
        real_comp = self.extract_feature_components(
            real_sequences.reshape(-1, 1, self.feature_dim)
        )
        fake_comp = self.extract_feature_components(
            fake_sequences.reshape(-1, 1, self.feature_dim)
        )

        for comp_name in real_comp.keys():
            real_comp_flat = real_comp[comp_name].reshape(-1)
            fake_comp_flat = fake_comp[comp_name].reshape(-1)

            # Remove zeros for non-zero components analysis
            real_nonzero = real_comp_flat[real_comp_flat != 0]
            fake_nonzero = fake_comp_flat[fake_comp_flat != 0]

            if len(real_nonzero) > 0 and len(fake_nonzero) > 0:
                try:
                    # Wasserstein distance (Earth Mover's Distance)
                    wasserstein_dist = wasserstein_distance(real_nonzero, fake_nonzero)
                    distribution_scores[f"{comp_name}_wasserstein"] = wasserstein_dist
                except:
                    distribution_scores[f"{comp_name}_wasserstein"] = float("inf")
            else:
                distribution_scores[f"{comp_name}_wasserstein"] = 0.0

        return distribution_scores

    def compute_comprehensive_similarity(self, real_sequences, fake_sequences):
        """Compute all similarity metrics and return comprehensive report"""

        # Ensure inputs are the same shape
        real_sequences = self.to_numpy(real_sequences)
        fake_sequences = self.to_numpy(fake_sequences)

        if real_sequences.shape != fake_sequences.shape:
            min_batch = min(real_sequences.shape[0], fake_sequences.shape[0])
            real_sequences = real_sequences[:min_batch]
            fake_sequences = fake_sequences[:min_batch]

        similarity_report = {}

        try:
            # Component-wise MSE
            similarity_report["component_mse"] = self.compute_component_wise_mse(
                real_sequences, fake_sequences
            )

            # Smoothness analysis
            real_smoothness = self.compute_gesture_smoothness(real_sequences)
            fake_smoothness = self.compute_gesture_smoothness(fake_sequences)

            similarity_report["smoothness_comparison"] = {
                "real": real_smoothness,
                "fake": fake_smoothness,
            }

            # Anatomical plausibility
            real_plausibility = self.compute_anatomical_plausibility(real_sequences)
            fake_plausibility = self.compute_anatomical_plausibility(fake_sequences)

            similarity_report["plausibility_comparison"] = {
                "real": real_plausibility,
                "fake": fake_plausibility,
            }

            # Temporal consistency
            similarity_report["temporal_consistency"] = (
                self.compute_temporal_consistency(real_sequences, fake_sequences)
            )

            # Distribution similarity
            similarity_report["distribution_similarity"] = (
                self.compute_feature_distribution_similarity(
                    real_sequences, fake_sequences
                )
            )

        except Exception as e:
            similarity_report["error"] = str(e)

        return similarity_report

    def summarize_similarity_scores(self, similarity_report):
        """Create a summary of key similarity metrics"""
        if "error" in similarity_report:
            return {"error": similarity_report["error"]}

        summary = {}

        # Overall quality score (lower is better for MSE-based metrics)
        try:
            overall_mse = similarity_report["component_mse"]["overall_mse"]
            summary["overall_mse"] = overall_mse
            summary["overall_quality"] = 1.0 / (
                1.0 + overall_mse
            )  # Convert to 0-1 scale
        except:
            summary["overall_mse"] = float("inf")
            summary["overall_quality"] = 0.0

        # Temporal quality
        try:
            temp_corr = similarity_report["temporal_consistency"][
                "temporal_correlation"
            ]["mean"]
            summary["temporal_correlation"] = temp_corr
        except:
            summary["temporal_correlation"] = 0.0

        # Component breakdown
        try:
            comp_mse = similarity_report["component_mse"]
            summary["hand_landmarks_quality"] = 1.0 / (
                1.0 + comp_mse.get("hand_landmarks_mse", float("inf"))
            )
            summary["hand_connections_quality"] = 1.0 / (
                1.0 + comp_mse.get("hand_connections_mse", float("inf"))
            )
            summary["pose_landmarks_quality"] = 1.0 / (
                1.0 + comp_mse.get("pose_landmarks_mse", float("inf"))
            )
            summary["pose_connections_quality"] = 1.0 / (
                1.0 + comp_mse.get("pose_connections_mse", float("inf"))
            )
        except:
            summary["hand_landmarks_quality"] = 0.0
            summary["hand_connections_quality"] = 0.0
            summary["pose_landmarks_quality"] = 0.0
            summary["pose_connections_quality"] = 0.0

        return summary
