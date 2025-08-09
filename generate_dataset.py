"""
This script generates synthetic anomaly videos from the Charades dataset to test plausibility priors for video LLMs.
It applies random temporal and spatial anomalies such as frame reversal, frame shuffling, object insertions,
and also combines clips from two different activities. The goal is to test if models are "trapped by training expectations"
when encountering anomalies in both space and time.

Key failure modes tested:
- Temporal anomalies: frame reversal, shuffling, speed changes
- Spatial anomalies: object insertions, color shifts, geometric distortions
- Combined anomalies: mixing activities from different scenes/actions
"""

import cv2
import os
import numpy as np
import csv
import random
import pandas as pd
from pathlib import Path
import shutil
from typing import List, Tuple, Dict
import json

class AnomalyVideoGenerator:
    def __init__(self, input_video_dir: str = 'sampled_videos', 
                 output_video_dir: str = 'anomaly_videos',
                 csv_file: str = 'sampled_train.csv'):
        """
        Initialize the anomaly video generator.
        
        Args:
            input_video_dir: Directory containing original videos
            output_video_dir: Directory to save anomaly videos
            csv_file: CSV file with video metadata
        """
        self.input_video_dir = Path(input_video_dir)
        self.output_video_dir = Path(output_video_dir)
        self.csv_file = csv_file
        
        # Create output directories
        self.output_video_dir.mkdir(exist_ok=True)
        self.synthetic_objects_dir = Path('synthetic_objects')
        self.synthetic_objects_dir.mkdir(exist_ok=True)
        
        # Load video metadata
        self.video_metadata = pd.read_csv(csv_file)
        self.video_files = list(self.input_video_dir.glob('*.mp4'))
        
        # Load class mappings
        self.class_mappings = self._load_class_mappings()
        
        # Generate synthetic objects for insertion
        self._generate_synthetic_objects()
        
        # Anomaly types and their probabilities
        self.anomaly_types = {
            'temporal': {
                'frame_reversal': 0.3,
                'frame_shuffling': 0.4,
                'speed_change': 0.3,
                'temporal_cut': 0.2
            },
            'spatial': {
                'object_insertion': 0.4,
                'color_shift': 0.3,
                'geometric_distortion': 0.2,
                'noise_injection': 0.3
            },
            'semantic': {
                'activity_mixing': 0.3,
                'scene_mixing': 0.2,
                'action_reversal': 0.2
            }
        }
    
    def _load_class_mappings(self) -> Dict[str, str]:
        """Load class mappings from Charades_v1_classes.txt"""
        class_mappings = {}
        classes_file = Path('Charades/Charades_v1_classes.txt')
        
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and ' ' in line:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            class_id, description = parts
                            class_mappings[class_id] = description
        
        return class_mappings
    
    def _decode_actions(self, actions_str: str) -> List[Dict]:
        """Decode action string into readable format with timestamps"""
        if pd.isna(actions_str) or not actions_str:
            return []
        
        decoded_actions = []
        actions = actions_str.split(';')
        
        for action in actions:
            action = action.strip()
            if action:
                parts = action.split()
                if len(parts) >= 3:
                    class_id = parts[0]
                    start_time = float(parts[1])
                    end_time = float(parts[2])
                    
                    description = self.class_mappings.get(class_id, f"Unknown action ({class_id})")
                    
                    decoded_actions.append({
                        'class_id': class_id,
                        'description': description,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': end_time - start_time
                    })
        
        return decoded_actions
    
    def _generate_synthetic_objects(self):
        """Generate synthetic objects for insertion anomalies."""
        object_shapes = [
            (50, 50), (30, 70), (70, 30), (40, 40)
        ]
        
        for i in range(5):
            shape = random.choice(object_shapes)
            obj = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
            
            # Random color
            color = np.random.randint(0, 255, size=3)
            obj[:, :, :3] = color
            
            # Alpha channel for transparency
            obj[:, :, 3] = np.random.randint(200, 255)
            
            # Add some pattern
            if random.random() < 0.5:
                obj[::2, ::2, 3] = 255  # Checkerboard pattern
            
            cv2.imwrite(str(self.synthetic_objects_dir / f'object_{i}.png'), obj)
    
    def _apply_temporal_anomaly(self, frames: List[np.ndarray], anomaly_type: str, fps: float = 30.0) -> Tuple[List[np.ndarray], List[Dict]]:
        """Apply temporal anomalies to frames and return timestamps."""
        anomaly_timestamps = []
        
        if anomaly_type == 'frame_reversal':
            anomaly_timestamps.append({
                'anomaly_type': 'frame_reversal',
                'start_time': 0.0,
                'end_time': len(frames) / fps,
                'description': 'Entire video played backwards'
            })
            return frames[::-1], anomaly_timestamps
        
        elif anomaly_type == 'frame_shuffling':
            # Shuffle frames in chunks to maintain some temporal coherence
            chunk_size = max(1, len(frames) // 10)
            shuffled_frames = []
            for i in range(0, len(frames), chunk_size):
                chunk = frames[i:i+chunk_size]
                start_time = i / fps
                end_time = min((i + len(chunk)) / fps, len(frames) / fps)
                
                random.shuffle(chunk)
                shuffled_frames.extend(chunk)
                
                anomaly_timestamps.append({
                    'anomaly_type': 'frame_shuffling',
                    'start_time': start_time,
                    'end_time': end_time,
                    'description': f'Frames shuffled between {start_time:.2f}s and {end_time:.2f}s'
                })
            return shuffled_frames, anomaly_timestamps
        
        elif anomaly_type == 'speed_change':
            # Change playback speed by skipping or duplicating frames
            if random.random() < 0.5:
                # Speed up: skip frames
                anomaly_timestamps.append({
                    'anomaly_type': 'speed_change',
                    'start_time': 0.0,
                    'end_time': len(frames) / fps,
                    'description': 'Video speed increased (frames skipped)'
                })
                return frames[::2], anomaly_timestamps
            else:
                # Slow down: duplicate frames
                slowed_frames = []
                for frame in frames:
                    slowed_frames.extend([frame] * 2)
                
                anomaly_timestamps.append({
                    'anomaly_type': 'speed_change',
                    'start_time': 0.0,
                    'end_time': len(frames) / fps,
                    'description': 'Video speed decreased (frames duplicated)'
                })
                return slowed_frames, anomaly_timestamps
        
        elif anomaly_type == 'temporal_cut':
            # Cut out a middle portion
            cut_start = len(frames) // 4
            cut_end = 3 * len(frames) // 4
            start_time = cut_start / fps
            end_time = cut_end / fps
            
            anomaly_timestamps.append({
                'anomaly_type': 'temporal_cut',
                'start_time': start_time,
                'end_time': end_time,
                'description': f'Middle portion cut from {start_time:.2f}s to {end_time:.2f}s'
            })
            return frames[:cut_start] + frames[cut_end:], anomaly_timestamps
        
        return frames, anomaly_timestamps
    
    def _apply_spatial_anomaly(self, frame: np.ndarray, frame_idx: int, fps: float, anomaly_type: str) -> Tuple[np.ndarray, List[Dict]]:
        """Apply spatial anomalies to a single frame and return timestamps."""
        anomaly_timestamps = []
        h, w, _ = frame.shape
        
        if anomaly_type == 'object_insertion':
            # Insert synthetic object
            obj_files = list(self.synthetic_objects_dir.glob('*.png'))
            if obj_files:
                obj_path = random.choice(obj_files)
                obj = cv2.imread(str(obj_path), cv2.IMREAD_UNCHANGED)
                
                if obj is not None and obj.shape[2] == 4:
                    obj_h, obj_w = obj.shape[:2]
                    x = random.randint(0, max(0, w - obj_w))
                    y = random.randint(0, max(0, h - obj_h))
                    
                    # Blend object with frame
                    alpha = obj[:, :, 3] / 255.0
                    for c in range(3):
                        frame[y:y+obj_h, x:x+obj_w, c] = (
                            (1-alpha) * frame[y:y+obj_h, x:x+obj_w, c] + 
                            alpha * obj[:, :, c]
                        )
                    
                    timestamp = frame_idx / fps
                    anomaly_timestamps.append({
                        'anomaly_type': 'object_insertion',
                        'start_time': timestamp,
                        'end_time': timestamp + 1.0,  # Assume object stays for 1 second
                        'description': f'Synthetic object inserted at {timestamp:.2f}s at position ({x}, {y})'
                    })
        
        elif anomaly_type == 'color_shift':
            # Apply color shift
            shift = np.random.randint(-50, 50, 3)
            frame = np.clip(frame.astype(np.int16) + shift, 0, 255).astype(np.uint8)
            
            timestamp = frame_idx / fps
            anomaly_timestamps.append({
                'anomaly_type': 'color_shift',
                'start_time': timestamp,
                'end_time': timestamp + 1.0,
                'description': f'Color shift applied at {timestamp:.2f}s (RGB shift: {shift})'
            })
        
        elif anomaly_type == 'geometric_distortion':
            # Apply perspective transform
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            offset = random.randint(-w//8, w//8)
            pts2 = np.float32([[offset, 0], [w-offset, 0], [0, h], [w, h]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(frame, matrix, (w, h))
            
            timestamp = frame_idx / fps
            anomaly_timestamps.append({
                'anomaly_type': 'geometric_distortion',
                'start_time': timestamp,
                'end_time': timestamp + 1.0,
                'description': f'Geometric distortion applied at {timestamp:.2f}s'
            })
        
        elif anomaly_type == 'noise_injection':
            # Add noise
            noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            timestamp = frame_idx / fps
            anomaly_timestamps.append({
                'anomaly_type': 'noise_injection',
                'start_time': timestamp,
                'end_time': timestamp + 1.0,
                'description': f'Noise injected at {timestamp:.2f}s'
            })
        
        return frame, anomaly_timestamps
    
    def _combine_videos(self, video1_path: Path, video2_path: Path) -> Tuple[List[np.ndarray], List[Dict], List[Dict]]:
        """Combine two videos to create semantic anomalies."""
        frames = []
        anomalies = []
        all_anomaly_timestamps = []
        
        # Read first video
        cap1 = cv2.VideoCapture(str(video1_path))
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        frames1 = []
        while True:
            ret, frame = cap1.read()
            if not ret:
                break
            frames1.append(frame)
        cap1.release()
        
        # Read second video
        cap2 = cv2.VideoCapture(str(video2_path))
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        frames2 = []
        while True:
            ret, frame = cap2.read()
            if not ret:
                break
            frames2.append(frame)
        cap2.release()
        
        # Combine frames
        frames = frames1 + frames2
        
        # Get metadata for anomaly description
        vid1_id = video1_path.stem
        vid2_id = video2_path.stem
        
        vid1_meta = self.video_metadata[self.video_metadata['id'] == vid1_id]
        vid2_meta = self.video_metadata[self.video_metadata['id'] == vid2_id]
        
        if not vid1_meta.empty and not vid2_meta.empty:
            scene1 = vid1_meta.iloc[0]['scene']
            scene2 = vid2_meta.iloc[0]['scene']
            script1 = vid1_meta.iloc[0]['script']
            script2 = vid2_meta.iloc[0]['script']
            
            # Add scene combination anomaly
            anomalies.append(f'combined_scenes_{scene1}_{scene2}')
            anomalies.append(f'combined_videos_{vid1_id}_{vid2_id}')
            
            # Add timestamps for the combination
            duration1 = len(frames1) / fps1
            all_anomaly_timestamps.append({
                'anomaly_type': 'scene_mixing',
                'start_time': duration1,
                'end_time': len(frames) / fps1,
                'description': f'Scene transition from {scene1} to {scene2} at {duration1:.2f}s'
            })
            
            all_anomaly_timestamps.append({
                'anomaly_type': 'video_combination',
                'start_time': 0.0,
                'end_time': len(frames) / fps1,
                'description': f'Combined videos: {vid1_id} ({scene1}) + {vid2_id} ({scene2})'
            })
        
        return frames, anomalies, all_anomaly_timestamps
    
    def generate_anomaly_video(self, video_path: Path, sample_idx: int) -> Dict:
        """Generate a single anomaly video."""
        print(f"Processing {video_path.name}...")
        
        # Read original video
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if not frames:
            return None
        
        anomalies = []
        all_anomaly_timestamps = []
        h, w, _ = frames[0].shape
        
        # Get original video metadata
        video_id = video_path.stem
        video_meta = self.video_metadata[self.video_metadata['id'] == video_id]
        
        original_script = ""
        combined_script = ""
        combined_video_id = ""
        
        if not video_meta.empty:
            original_script = video_meta.iloc[0]['script']
        
        # Apply semantic anomalies (combine with another video)
        if random.random() < self.anomaly_types['semantic']['activity_mixing']:
            other_video = random.choice(self.video_files)
            if other_video != video_path:
                frames, semantic_anomalies, semantic_timestamps = self._combine_videos(video_path, other_video)
                anomalies.extend(semantic_anomalies)
                all_anomaly_timestamps.extend(semantic_timestamps)
                
                # Get combined video metadata
                other_video_id = other_video.stem
                other_video_meta = self.video_metadata[self.video_metadata['id'] == other_video_id]
                if not other_video_meta.empty:
                    combined_script = other_video_meta.iloc[0]['script']
                    combined_video_id = other_video_id
        
        # Apply temporal anomalies
        for anomaly_type, prob in self.anomaly_types['temporal'].items():
            if random.random() < prob:
                frames, temporal_timestamps = self._apply_temporal_anomaly(frames, anomaly_type, fps)
                anomalies.append(anomaly_type)
                all_anomaly_timestamps.extend(temporal_timestamps)
        
        # Apply spatial anomalies to random frames
        for i, frame in enumerate(frames):
            for anomaly_type, prob in self.anomaly_types['spatial'].items():
                if random.random() < prob * 0.1:  # Lower probability per frame
                    frames[i], spatial_timestamps = self._apply_spatial_anomaly(frame, i, fps, anomaly_type)
                    all_anomaly_timestamps.extend(spatial_timestamps)
                    if anomaly_type not in anomalies:
                        anomalies.append(anomaly_type)
        
        # Save anomaly video
        output_name = f"anomaly_video_{sample_idx:04d}.mp4"
        output_path = self.output_video_dir / output_name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
        
        for frame in frames:
            out.write(frame)
        out.release()
        
        # Decode actions
        original_actions = []
        if not video_meta.empty:
            actions_str = video_meta.iloc[0]['actions']
            original_actions = self._decode_actions(actions_str)
        
        # Format anomaly timestamps for CSV
        anomaly_timestamps_str = ""
        if all_anomaly_timestamps:
            timestamp_parts = []
            for ts in all_anomaly_timestamps:
                timestamp_parts.append(f"{ts['anomaly_type']} {ts['start_time']:.2f} {ts['end_time']:.2f}")
            anomaly_timestamps_str = ";".join(timestamp_parts)
        
        # Format decoded actions for CSV
        decoded_actions_str = ""
        if original_actions:
            action_parts = []
            for action in original_actions:
                action_parts.append(f"{action['class_id']} {action['start_time']:.2f} {action['end_time']:.2f} ({action['description']})")
            decoded_actions_str = ";".join(action_parts)
        
        result = {
            'anomaly_video_name': output_name,
            'original_video_id': video_id,
            'original_scene': video_meta.iloc[0]['scene'] if not video_meta.empty else 'unknown',
            'original_script': original_script,
            'combined_video_id': combined_video_id,
            'combined_script': combined_script,
            'original_actions': video_meta.iloc[0]['actions'] if not video_meta.empty else 'unknown',
            'decoded_actions': decoded_actions_str,
            'anomalies_applied': ';'.join(set(anomalies)),
            'anomaly_timestamps': anomaly_timestamps_str,
            'anomaly_count': len(set(anomalies))
        }
        
        return result
    
    def generate_dataset(self, n_samples: int = 100):
        """Generate the complete anomaly dataset."""
        print(f"Generating {n_samples} anomaly videos...")
        
        results = []
        
        for i in range(n_samples):
            video_path = random.choice(self.video_files)
            result = self.generate_anomaly_video(video_path, i)
            
            if result:
                results.append(result)
                print(f"Generated {result['anomaly_video_name']} with anomalies: {result['anomalies_applied']}")
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_video_dir / 'anomaly_summary.csv', index=False)
        
        # Save detailed metadata
        metadata = {
            'dataset_info': {
                'total_anomaly_videos': len(results),
                'original_videos_used': len(set(r['original_video_id'] for r in results)),
                'anomaly_types_used': list(set(
                    anomaly for r in results 
                    for anomaly in r['anomalies_applied'].split(';')
                ))
            },
            'anomaly_distribution': results_df['anomaly_count'].value_counts().to_dict(),
            'scene_distribution': results_df['original_scene'].value_counts().to_dict()
        }
        
        with open(self.output_video_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset generation complete!")
        print(f"Generated {len(results)} anomaly videos")
        print(f"Results saved to: {self.output_video_dir}")
        print(f"Summary CSV: {self.output_video_dir / 'anomaly_summary.csv'}")
        print(f"Metadata: {self.output_video_dir / 'dataset_metadata.json'}")
        
        return results

def main():
    """Main function to run the anomaly video generation."""
    generator = AnomalyVideoGenerator()
    
    # Generate 100 anomaly videos
    results = generator.generate_dataset(n_samples=100)
    
    # Print some statistics
    print("\n=== Dataset Statistics ===")
    anomaly_counts = [r['anomaly_count'] for r in results]
    print(f"Average anomalies per video: {np.mean(anomaly_counts):.2f}")
    print(f"Max anomalies in a video: {max(anomaly_counts)}")
    print(f"Min anomalies in a video: {min(anomaly_counts)}")
    
    # Show most common anomalies
    all_anomalies = []
    for r in results:
        all_anomalies.extend(r['anomalies_applied'].split(';'))
    
    from collections import Counter
    anomaly_freq = Counter(all_anomalies)
    print(f"\nMost common anomalies:")
    for anomaly, count in anomaly_freq.most_common(5):
        print(f"  {anomaly}: {count}")

if __name__ == "__main__":
    main()
