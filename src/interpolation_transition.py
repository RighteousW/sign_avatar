import os
import pickle
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
from collections import defaultdict
import re

from constants import LANDMARKS_DIR, OUTPUT_DIR
from landmark_visualization import LandmarkVisualizer


class GestureTransitionPreprocessor:
    """Preprocesses gesture files and creates all possible transitions"""

    def __init__(self, handedness_filter: Optional[str] = None):
        self.visualizer = LandmarkVisualizer()
        self.handedness_filter = handedness_filter  # 'left', 'right', 'both', or None
        
    def extract_gloss_from_filename(self, filename: str) -> str:
        """Extract gloss from filename by removing timestamp and suffixes"""
        # Remove _landmarks.pkl
        base_name = filename.replace('_landmarks.pkl', '')
        
        # Remove _flipped if present
        if base_name.endswith('_flipped'):
            base_name = base_name[:-8]  # Remove '_flipped'
        
        # Find timestamp pattern YYYYMMDD_HHMMSS
        timestamp_pattern = r'_(\d{8}_\d{6})$'
        match = re.search(timestamp_pattern, base_name)
        
        if match:
            # Remove timestamp from the end
            gloss = base_name[:match.start()]
        else:
            # Fallback: split by underscore and take all but last two parts
            parts = base_name.split('_')
            if len(parts) >= 3:
                gloss = '_'.join(parts[:-2])  # Remove last two parts (assumed date/time)
            else:
                gloss = base_name
        
        return gloss

    def determine_handedness(self, frames: List[Dict]) -> str:
        """Determine handedness based on first and last 25% of frames"""
        if not frames:
            return "unknown"
        
        num_frames = len(frames)
        quarter_size = max(1, num_frames // 4)
        
        # Get first and last 25% of frames
        important_frames = frames[:quarter_size] + frames[-quarter_size:]
        
        left_count = 0
        right_count = 0
        both_count = 0
        
        for frame in important_frames:
            hands_present = set()
            
            if 'hands' in frame and frame['hands']:
                for hand in frame['hands']:
                    hands_present.add(hand['handedness'])
            
            if 'Left' in hands_present and 'Right' in hands_present:
                both_count += 1
            elif 'Left' in hands_present:
                left_count += 1
            elif 'Right' in hands_present:
                right_count += 1
        
        # Determine predominant handedness
        total_frames = len(important_frames)
        left_ratio = left_count / total_frames if total_frames > 0 else 0
        right_ratio = right_count / total_frames if total_frames > 0 else 0
        both_ratio = both_count / total_frames if total_frames > 0 else 0
        
        # Classification logic
        if both_ratio > 0.3:  # If >30% of frames have both hands
            return "both"
        elif left_ratio > right_ratio:
            return "left"
        elif right_ratio > left_ratio:
            return "right"
        else:
            return "unknown"

    def get_gesture_files(self) -> Dict[str, List[Tuple[str, str]]]:
        """Get all gesture files organized by gloss with handedness"""
        gesture_files = defaultdict(list)
        
        for root, dirs, files in os.walk(LANDMARKS_DIR):
            for file in files:
                if file.endswith('_landmarks.pkl'):
                    file_path = os.path.join(root, file)
                    
                    try:
                        # Load file to determine handedness
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                        
                        handedness = self.determine_handedness(data['frames'])
                        
                        # Apply handedness filter
                        if self.handedness_filter and handedness != self.handedness_filter:
                            continue
                        
                        # Extract gloss
                        gloss = self.extract_gloss_from_filename(file)
                        
                        gesture_files[gloss].append((file_path, handedness))
                        
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
                        continue
        
        return gesture_files

    def calculate_hand_consistency(self, frames: List[Dict]) -> Dict[str, float]:
        """Calculate hand consistency metrics to detect flickering"""
        if not frames:
            return {"left_consistency": 0.0, "right_consistency": 0.0, "overall_penalty": 1.0}
        
        # Track hand presence per frame
        left_present = []
        right_present = []
        
        for frame in frames:
            left_found = False
            right_found = False
            
            if 'hands' in frame and frame['hands']:
                for hand in frame['hands']:
                    if hand['handedness'] == 'Left':
                        left_found = True
                    elif hand['handedness'] == 'Right':
                        right_found = True
            
            left_present.append(left_found)
            right_present.append(right_found)
        
        # Calculate consistency metrics
        def calculate_consistency_score(presence_list: List[bool]) -> float:
            if len(presence_list) <= 1:
                return 1.0
            
            # Count transitions (flickering events)
            transitions = 0
            for i in range(1, len(presence_list)):
                if presence_list[i] != presence_list[i-1]:
                    transitions += 1
            
            # Penalize frequent transitions
            # Good: few transitions, Bad: many transitions
            max_transitions = len(presence_list) - 1
            if max_transitions == 0:
                return 1.0
            
            transition_ratio = transitions / max_transitions
            
            # Convert to consistency score (lower transitions = higher consistency)
            consistency = max(0.0, 1.0 - transition_ratio)
            
            return consistency
        
        def calculate_single_frame_penalty(presence_list: List[bool]) -> float:
            """Penalize hands that appear for only single frames"""
            if len(presence_list) <= 2:
                return 1.0
            
            single_frame_count = 0
            total_appearances = 0
            
            i = 0
            while i < len(presence_list):
                if presence_list[i]:  # Hand is present
                    # Count consecutive frames where hand is present
                    consecutive = 1
                    j = i + 1
                    while j < len(presence_list) and presence_list[j]:
                        consecutive += 1
                        j += 1
                    
                    total_appearances += 1
                    if consecutive == 1:  # Single frame appearance
                        single_frame_count += 1
                    
                    i = j  # Move past this appearance
                else:
                    i += 1
            
            if total_appearances == 0:
                return 1.0
            
            # Penalty based on ratio of single-frame appearances
            single_frame_ratio = single_frame_count / total_appearances
            penalty = max(0.1, 1.0 - single_frame_ratio)  # Minimum 0.1, maximum 1.0
            
            return penalty
        
        left_consistency = calculate_consistency_score(left_present)
        right_consistency = calculate_consistency_score(right_present)
        
        left_single_penalty = calculate_single_frame_penalty(left_present)
        right_single_penalty = calculate_single_frame_penalty(right_present)
        
        # Calculate overall penalty
        # Average the consistency scores, weighted by presence
        left_weight = sum(left_present) / len(left_present) if left_present else 0
        right_weight = sum(right_present) / len(right_present) if right_present else 0
        
        if left_weight == 0 and right_weight == 0:
            overall_penalty = 0.1  # Very bad if no hands detected
        elif left_weight == 0:
            overall_penalty = right_consistency * right_single_penalty
        elif right_weight == 0:
            overall_penalty = left_consistency * left_single_penalty
        else:
            # Both hands present, weight by their relative presence
            total_weight = left_weight + right_weight
            overall_penalty = (
                (left_weight / total_weight) * left_consistency * left_single_penalty +
                (right_weight / total_weight) * right_consistency * right_single_penalty
            )
        
        return {
            "left_consistency": left_consistency,
            "right_consistency": right_consistency,
            "left_single_penalty": left_single_penalty,
            "right_single_penalty": right_single_penalty,
            "left_presence_ratio": left_weight,
            "right_presence_ratio": right_weight,
            "overall_penalty": overall_penalty
        }

    def select_representative_file(self, files: List[Tuple[str, str]], gloss: str) -> Tuple[str, str]:
        """Select the most representative file for a gloss"""
        if len(files) == 1:
            return files[0]
        
        # Load all files and analyze them
        file_stats = []
        for file_path, handedness in files:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                frames = data['frames']
                num_frames = len(frames)
                
                # Calculate hand coverage in transition-critical areas (first/last 25%)
                quarter_size = max(1, num_frames // 4)
                important_frames = frames[:quarter_size] + frames[-quarter_size:]
                
                hand_frames = sum(1 for frame in important_frames if frame.get('hands'))
                hand_coverage = hand_frames / len(important_frames) if important_frames else 0
                
                # Calculate hand consistency (flickering penalty)
                consistency_metrics = self.calculate_hand_consistency(frames)
                consistency_penalty = consistency_metrics["overall_penalty"]
                
                # Prefer reasonable length (10-40 frames) and good coverage
                length_score = 1.0
                if num_frames < 10:
                    length_score = 0.5
                elif num_frames > 40:
                    length_score = 0.8
                
                # Combined score with flickering penalty
                base_score = hand_coverage * length_score * min(num_frames, 30)
                total_score = base_score * consistency_penalty  # Multiply by penalty (0.1-1.0)
                
                file_stats.append({
                    'path': file_path,
                    'handedness': handedness,
                    'frames': num_frames,
                    'hand_coverage': hand_coverage,
                    'consistency_penalty': consistency_penalty,
                    'consistency_metrics': consistency_metrics,
                    'base_score': base_score,
                    'score': total_score
                })
                
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                continue
        
        if not file_stats:
            return files[0]
        
        # Select file with highest score
        best_file = max(file_stats, key=lambda x: x['score'])
        
        # Print detailed selection info
        print(f"Gloss '{gloss}' ({best_file['handedness']}): {len(files)} files, "
              f"selected {os.path.basename(best_file['path'])}")
        print(f"  Frames: {best_file['frames']}, Coverage: {best_file['hand_coverage']:.2f}, "
              f"Consistency: {best_file['consistency_penalty']:.2f}")
        
        # Show consistency breakdown
        metrics = best_file['consistency_metrics']
        print(f"  Left: {metrics['left_presence_ratio']:.2f} presence, "
              f"{metrics['left_consistency']:.2f} consistency, "
              f"{metrics['left_single_penalty']:.2f} single-frame penalty")
        print(f"  Right: {metrics['right_presence_ratio']:.2f} presence, "
              f"{metrics['right_consistency']:.2f} consistency, "
              f"{metrics['right_single_penalty']:.2f} single-frame penalty")
        
        # If the selected file has poor consistency, warn about it
        if best_file['consistency_penalty'] < 0.7:
            print(f"  WARNING: Selected file has poor hand consistency (flickering detected)")
        
        return best_file['path'], best_file['handedness']

    def analyze_hand_presence(self, frames: List[Dict]) -> Dict[str, List[int]]:
        """Analyze which frames have which hands present"""
        left_frames = []
        right_frames = []
        
        for i, frame in enumerate(frames):
            if 'hands' in frame and frame['hands']:
                for hand in frame['hands']:
                    if hand['handedness'] == 'Left':
                        left_frames.append(i)
                    elif hand['handedness'] == 'Right':
                        right_frames.append(i)
        
        return {
            'Left': sorted(list(set(left_frames))),
            'Right': sorted(list(set(right_frames)))
        }

    def find_optimal_cut_points(self, first_data: Dict, second_data: Dict) -> Tuple[int, int]:
        """Find optimal cut points to avoid hand appearance/disappearance issues"""
        first_frames = first_data['frames']
        second_frames = second_data['frames']
        
        # Analyze hand presence
        first_presence = self.analyze_hand_presence(first_frames)
        second_presence = self.analyze_hand_presence(second_frames)
        
        # Find where hands first appear and last appear
        first_cut = len(first_frames) - 1  # Default to end of first sequence
        second_cut = 0  # Default to start of second sequence
        
        # For first gesture, find the latest point where all hands that will be needed are present
        hands_needed_in_second = set()
        if second_presence['Left']:
            hands_needed_in_second.add('Left')
        if second_presence['Right']:
            hands_needed_in_second.add('Right')
        
        # Find the latest frame in first gesture where all needed hands are present
        for i in range(len(first_frames) - 1, -1, -1):
            frame = first_frames[i]
            present_hands = set()
            
            if 'hands' in frame and frame['hands']:
                for hand in frame['hands']:
                    present_hands.add(hand['handedness'])
            
            if hands_needed_in_second.issubset(present_hands) or not hands_needed_in_second:
                first_cut = i
                break
        
        # For second gesture, find earliest point where hands that exist in first are present
        hands_available_in_first = set()
        if first_presence['Left']:
            hands_available_in_first.add('Left')
        if first_presence['Right']:
            hands_available_in_first.add('Right')
        
        # Find the earliest frame in second gesture where available hands appear
        for i, frame in enumerate(second_frames):
            present_hands = set()
            
            if 'hands' in frame and frame['hands']:
                for hand in frame['hands']:
                    present_hands.add(hand['handedness'])
            
            if hands_available_in_first.intersection(present_hands) or not hands_available_in_first:
                second_cut = i
                break
        
        return first_cut, second_cut

    def calculate_frame_similarity(self, frame1: Dict, frame2: Dict) -> float:
        """Calculate similarity between two frames based on hand landmarks"""
        # Extract hand positions for both frames
        pos1 = np.zeros((2, 21, 3))
        pos2 = np.zeros((2, 21, 3))
        
        # Fill positions for frame1
        if 'hands' in frame1 and frame1['hands']:
            for hand in frame1['hands']:
                hand_idx = 0 if hand['handedness'] == 'Left' else 1
                landmarks = np.array(hand['landmarks'])
                if landmarks.shape == (21, 3):
                    pos1[hand_idx] = landmarks
        
        # Fill positions for frame2
        if 'hands' in frame2 and frame2['hands']:
            for hand in frame2['hands']:
                hand_idx = 0 if hand['handedness'] == 'Left' else 1
                landmarks = np.array(hand['landmarks'])
                if landmarks.shape == (21, 3):
                    pos2[hand_idx] = landmarks
        
        # Calculate normalized difference
        diff = np.linalg.norm(pos1 - pos2)
        max_possible_diff = np.sqrt(2 * 21 * 3)  # Maximum possible difference
        
        # Return similarity as percentage (1.0 = identical, 0.0 = completely different)
        similarity = max(0, 1.0 - (diff / max_possible_diff))
        return similarity

    def reduce_similar_frames(self, frames: List[Dict], similarity_threshold: float = 0.95, 
                             min_consecutive: int = 5) -> List[Dict]:
        """Reduce consecutive similar frames to representative samples"""
        if len(frames) <= min_consecutive:
            return frames
        
        reduced_frames = []
        i = 0
        
        while i < len(frames):
            current_frame = frames[i]
            reduced_frames.append(current_frame)
            
            # Look ahead to find consecutive similar frames
            consecutive_count = 1
            j = i + 1
            
            while j < len(frames):
                similarity = self.calculate_frame_similarity(current_frame, frames[j])
                if similarity >= similarity_threshold:
                    consecutive_count += 1
                    j += 1
                else:
                    break
            
            # If we found enough consecutive similar frames, subsample them
            if consecutive_count >= min_consecutive:
                # Take every 5th frame from the similar sequence
                # For 30 similar frames: take frames 1, 6, 11, 16, 21, 26
                similar_start = i
                similar_end = i + consecutive_count - 1
                
                # Sample every 5th frame from the similar sequence
                sample_indices = list(range(similar_start + 5, similar_end + 1, 5))
                
                for idx in sample_indices:
                    if idx < len(frames):
                        reduced_frames.append(frames[idx])
                
                print(f"  Reduced {consecutive_count} similar frames (indices {similar_start}-{similar_end}) "
                      f"to {1 + len(sample_indices)} frames")
                
                # Skip past the similar sequence
                i = similar_end + 1
            else:
                # No long similar sequence, move to next frame
                i += 1
        
        # Renumber frames
        for idx, frame in enumerate(reduced_frames):
            frame['frame_number'] = idx
        
        return reduced_frames

    def preprocess_gesture(self, data: Dict, start_frame: int = None, end_frame: int = None, 
                          reduce_similar: bool = True) -> Dict:
        """Preprocess gesture data by trimming frames and reducing similar sequences"""
        original_frame_count = len(data['frames'])
        
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = len(data['frames']) - 1
        
        # Ensure valid range
        start_frame = max(0, start_frame)
        end_frame = min(len(data['frames']) - 1, end_frame)
        
        if start_frame >= end_frame:
            # Fallback to using the whole sequence
            start_frame = 0
            end_frame = len(data['frames']) - 1
        
        # Create new data with trimmed frames
        trimmed_data = data.copy()
        trimmed_data['frames'] = data['frames'][start_frame:end_frame + 1]
        trimmed_frame_count = len(trimmed_data['frames'])
        
        # Reduce similar frames if requested
        final_frame_count = trimmed_frame_count
        if reduce_similar and len(trimmed_data['frames']) > 5:
            print(f"  Reducing similar frames in sequence...")
            reduced_frames = self.reduce_similar_frames(trimmed_data['frames'])
            trimmed_data['frames'] = reduced_frames
            final_frame_count = len(reduced_frames)
        
        trimmed_data['preprocessing_info'] = {
            'original_frames': original_frame_count,
            'trimmed_start': start_frame,
            'trimmed_end': end_frame,
            'frames_after_trimming': trimmed_frame_count,
            'frames_after_similarity_reduction': final_frame_count,
            'similarity_reduction_enabled': reduce_similar,
            'total_reduction_ratio': final_frame_count / original_frame_count if original_frame_count > 0 else 1.0
        }
        
        return trimmed_data

    def extract_hand_positions(self, frames: List[Dict]) -> np.ndarray:
        """Extract hand positions from frames into a consistent format"""
        positions = []
        
        for frame in frames:
            # Initialize frame with zeros for 2 hands * 21 landmarks * 3 coords
            frame_positions = np.zeros((2, 21, 3))
            
            if "hands" in frame and frame["hands"]:
                for hand in frame["hands"]:
                    hand_idx = 0 if hand["handedness"] == "Left" else 1
                    landmarks = np.array(hand["landmarks"])
                    if landmarks.shape == (21, 3):
                        frame_positions[hand_idx] = landmarks
            
            positions.append(frame_positions)
        
        return np.array(positions)

    def extract_pose_positions(self, frames: List[Dict]) -> np.ndarray:
        """Extract pose positions from frames into a consistent format"""
        positions = []
        
        for frame in frames:
            # Initialize frame with zeros for 33 landmarks * 4 coords (x, y, z, visibility)
            frame_positions = np.zeros((33, 4))
            
            if "pose" in frame and frame["pose"] and frame["pose"]["landmarks"]:
                landmarks = np.array(frame["pose"]["landmarks"])
                if landmarks.shape == (33, 4):
                    frame_positions = landmarks
            
            positions.append(frame_positions)
        
        return np.array(positions)

    def interpolate_positions(self, start_pos: np.ndarray, end_pos: np.ndarray, 
                            num_frames: int) -> np.ndarray:
        """Interpolate between start and end positions"""
        if start_pos.shape != end_pos.shape:
            raise ValueError("Start and end positions must have the same shape")
        
        # For simple linear interpolation between two points
        original_shape = start_pos.shape
        start_flat = start_pos.flatten()
        end_flat = end_pos.flatten()
        
        # Create interpolation weights
        weights = np.linspace(0, 1, num_frames)
        
        # Interpolate each dimension using simple linear interpolation
        interpolated = []
        for i in range(num_frames):
            # Linear interpolation: start + weight * (end - start)
            frame_values = start_flat + weights[i] * (end_flat - start_flat)
            interpolated.append(frame_values)
        
        # Reshape back to original format
        interpolated = np.array(interpolated)
        return interpolated.reshape((num_frames,) + original_shape)

    def positions_to_frames(self, hand_positions: np.ndarray, pose_positions: np.ndarray = None) -> List[Dict]:
        """Convert position arrays back to frame format"""
        frames = []
        num_frames = hand_positions.shape[0]
        
        for i in range(num_frames):
            frame_data = {"frame_number": i, "hands": []}
            
            # Process hands
            for hand_idx in range(2):
                hand_landmarks = hand_positions[i, hand_idx]
                # Check if hand has any non-zero landmarks
                if np.any(hand_landmarks != 0):
                    hand_data = {
                        "hand_index": hand_idx,
                        "handedness": "Left" if hand_idx == 0 else "Right",
                        "landmarks": hand_landmarks.tolist()
                    }
                    frame_data["hands"].append(hand_data)
            
            # Process pose if available
            if pose_positions is not None:
                pose_landmarks = pose_positions[i]
                if np.any(pose_landmarks != 0):
                    frame_data["pose"] = {
                        "landmarks": pose_landmarks.tolist()
                    }
                else:
                    frame_data["pose"] = None
            
            frames.append(frame_data)
        
        return frames

    def generate_transition(self, first_data: Dict, second_data: Dict, 
                          transition_length: int = 6) -> Dict[str, Any]:
        """Generate smooth transition between two gestures"""
        first_frames = first_data["frames"]
        second_frames = second_data["frames"]
        
        if not first_frames or not second_frames:
            raise ValueError("Both gestures must have frames")
        
        # Extract positions
        first_hand_positions = self.extract_hand_positions(first_frames)
        second_hand_positions = self.extract_hand_positions(second_frames)
        
        # Get last frame of first gesture and first frame of second gesture
        start_hand_pos = first_hand_positions[-1]
        end_hand_pos = second_hand_positions[0]
        
        # Interpolate hand positions
        transition_hand_positions = self.interpolate_positions(
            start_hand_pos, end_hand_pos, transition_length
        )
        
        # Handle pose if available in both gestures
        transition_pose_positions = None
        if ("pose" in first_data.get("landmark_types", []) and 
            "pose" in second_data.get("landmark_types", [])):
            
            first_pose_positions = self.extract_pose_positions(first_frames)
            second_pose_positions = self.extract_pose_positions(second_frames)
            
            start_pose_pos = first_pose_positions[-1]
            end_pose_pos = second_pose_positions[0]
            
            transition_pose_positions = self.interpolate_positions(
                start_pose_pos, end_pose_pos, transition_length
            )
        
        # Convert back to frame format
        transition_frames = self.positions_to_frames(
            transition_hand_positions, transition_pose_positions
        )
        
        # Create combined sequence
        combined_frames = (
            first_frames[:-1] +  # All but last frame of first gesture
            transition_frames +   # Generated transition
            second_frames[1:]     # All but first frame of second gesture
        )
        
        # Re-number frames
        for i, frame in enumerate(combined_frames):
            frame["frame_number"] = i
        
        # Create combined data structure
        combined_data = {
            "video_path": "generated_transition",
            "timestamp": f"transition_{len(combined_frames)}_frames",
            "frames": combined_frames,
            "landmark_types": first_data.get("landmark_types", ["hand_landmarks"]),
            "feature_info": first_data.get("feature_info"),
            "max_feature_vector_size": first_data.get("max_feature_vector_size"),
            "transition_info": {
                "first_gesture": {
                    "original_frames": len(first_frames),
                    "used_frames": len(first_frames) - 1,
                    "preprocessing": first_data.get("preprocessing_info")
                },
                "transition_frames": transition_length,
                "second_gesture": {
                    "original_frames": len(second_frames),
                    "used_frames": len(second_frames) - 1,
                    "preprocessing": second_data.get("preprocessing_info")
                },
                "total_frames": len(combined_frames)
            }
        }
        
        return combined_data

    def create_transition_matrix(self, transition_length: int = 6) -> Dict[str, Any]:
        """Create all possible transitions between representative gestures"""
        print("=== Creating Gesture Transition Matrix ===")
        
        # Get gesture files organized by gloss
        gesture_files = self.get_gesture_files()
        print(f"Found {len(gesture_files)} unique gestures")
        
        if self.handedness_filter:
            print(f"Filtering for {self.handedness_filter} handedness only")
        
        # Select representative files
        representatives = {}
        handedness_counts = defaultdict(int)
        
        for gloss, files in gesture_files.items():
            if files:  # Only process glosses that have files
                rep_file, handedness = self.select_representative_file(files, gloss)
                representatives[gloss] = (rep_file, handedness)
                handedness_counts[handedness] += 1
        
        print(f"\nSelected {len(representatives)} representative gestures:")
        for handedness, count in handedness_counts.items():
            print(f"  {handedness}: {count} gestures")
        
        # Create output directory
        output_dir = Path(OUTPUT_DIR) / "transition_matrix"
        if self.handedness_filter:
            output_dir = output_dir / f"{self.handedness_filter}_handed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate all possible transitions (only within same handedness)
        transitions = {}
        glosses = list(representatives.keys())
        
        # Group glosses by handedness for transition generation
        handedness_groups = defaultdict(list)
        for gloss in glosses:
            _, handedness = representatives[gloss]
            handedness_groups[handedness].append(gloss)
        
        total_transitions = sum(len(group) * len(group) for group in handedness_groups.values())
        current = 0
        
        print(f"\nGenerating {total_transitions} transitions (within same handedness)...")
        
        for handedness, group_glosses in handedness_groups.items():
            print(f"\nProcessing {handedness} handed gestures ({len(group_glosses)} gestures)...")
            
            for first_gloss in group_glosses:
                for second_gloss in group_glosses:
                    current += 1
                    print(f"[{current}/{total_transitions}] {first_gloss} → {second_gloss} ({handedness})")
                    
                    try:
                        # Load gesture data
                        first_file, _ = representatives[first_gloss]
                        second_file, _ = representatives[second_gloss]
                        
                        with open(first_file, 'rb') as f:
                            first_data = pickle.load(f)
                        with open(second_file, 'rb') as f:
                            second_data = pickle.load(f)
                        
                        # Find optimal cut points
                        first_cut, second_cut = self.find_optimal_cut_points(first_data, second_data)
                        
                        # Preprocess gestures (with similarity reduction)
                        first_processed = self.preprocess_gesture(first_data, end_frame=first_cut, reduce_similar=True)
                        second_processed = self.preprocess_gesture(second_data, start_frame=second_cut, reduce_similar=True)
                        
                        # Generate transition
                        transition_data = self.generate_transition(
                            first_processed, second_processed, transition_length
                        )
                        
                        # Add metadata
                        transition_data['transition_metadata'] = {
                            'from_gloss': first_gloss,
                            'to_gloss': second_gloss,
                            'handedness': handedness,
                            'from_file': first_file,
                            'to_file': second_file,
                            'transition_length': transition_length,
                            'preprocessing': {
                                'first_cut_point': first_cut,
                                'second_cut_point': second_cut
                            }
                        }
                        
                        # Save individual transition
                        transition_file = output_dir / f"{first_gloss}_to_{second_gloss}_{handedness}.pkl"
                        with open(transition_file, 'wb') as f:
                            pickle.dump(transition_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                        
                        # Store in matrix
                        transitions[f"{first_gloss}_to_{second_gloss}_{handedness}"] = {
                            'file': str(transition_file),
                            'from_gloss': first_gloss,
                            'to_gloss': second_gloss,
                            'handedness': handedness,
                            'frames': len(transition_data['frames']),
                            'transition_info': transition_data['transition_info']
                        }
                        
                    except Exception as e:
                        print(f"  Error: {e}")
                        transitions[f"{first_gloss}_to_{second_gloss}_{handedness}"] = {
                            'error': str(e),
                            'from_gloss': first_gloss,
                            'to_gloss': second_gloss,
                            'handedness': handedness
                        }
        
        # Create summary data
        summary = {
            'created_at': str(Path().cwd()),
            'handedness_filter': self.handedness_filter,
            'transition_length': transition_length,
            'total_glosses': len(glosses),
            'handedness_distribution': dict(handedness_counts),
            'total_transitions': total_transitions,
            'successful_transitions': len([t for t in transitions.values() if 'error' not in t]),
            'failed_transitions': len([t for t in transitions.values() if 'error' in t]),
            'glosses_by_handedness': dict(handedness_groups),
            'representatives': {g: {'file': os.path.basename(f), 'handedness': h} 
                              for g, (f, h) in representatives.items()},
            'transitions': transitions
        }
        
        # Save summary
        summary_file = output_dir / "transition_matrix_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save summary as pickle too
        summary_pkl = output_dir / "transition_matrix_summary.pkl"
        with open(summary_pkl, 'wb') as f:
            pickle.dump(summary, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"\n=== Transition Matrix Complete ===")
        print(f"Generated {summary['successful_transitions']}/{total_transitions} transitions")
        print(f"Failed: {summary['failed_transitions']}")
        print("Handedness distribution:")
        for handedness, count in handedness_counts.items():
            print(f"  {handedness}: {count} gestures")
        print(f"Output directory: {output_dir}")
        print(f"Summary file: {summary_file}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="Create comprehensive gesture transition matrix")
    parser.add_argument(
        "--handedness",
        choices=["left", "right", "both"],
        default="right",
        help="Filter by handedness (left, right, or both)",
    )
    parser.add_argument("--length", type=int, default=6, 
                       help="Transition length in frames")

    args = parser.parse_args()

    print("=== Gesture Transition Matrix Generator ===")
    print("Pre-rendering all possible transitions between representative gestures")
    print("Analyzing first/last 25% of frames to determine handedness")

    if args.handedness:
        print(f"Filtering for {args.handedness} handedness only")

    try:
        processor = GestureTransitionPreprocessor(handedness_filter=args.handedness)
        summary = processor.create_transition_matrix(transition_length=args.length)

        print(f"\nMatrix creation complete!")
        print(f"Check the output directory for all transition files and summary")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
