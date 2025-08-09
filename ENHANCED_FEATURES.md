# Enhanced Features for Plausibility Priors Testing System

## Overview

The plausibility priors testing system has been significantly enhanced to provide more detailed and comprehensive information for testing video language models. These enhancements address the need for better understanding of how models respond to anomalies in both space and time.

## Key Enhancements

### 1. Combined Video Information

**Feature**: When videos are combined to create semantic anomalies, the system now tracks:
- **Combined Video ID**: The ID of the second video used in the combination
- **Combined Script**: The script description of the second video
- **Scene Transition Information**: Details about how scenes change from one video to another

**Example**:
```
combined_video_id: FQA3W
combined_script: "A person in their bedroom is playing with their shoes. Once they have put their shoes on, they start briskly running towards the door and grab onto the doorknob."
```

### 2. Decoded Action Classes

**Feature**: The system now decodes action class IDs into human-readable descriptions using the Charades class mappings.

**Before**:
```
actions: c134 0.00 6.90;c020 12.00 31.00;c063 16.80 22.30
```

**After**:
```
decoded_actions: c134 0.00 6.90 (Lying on a bed);c020 12.00 31.00 (Holding a bag);c063 16.80 22.30 (Taking food from somewhere)
```

### 3. Anomaly Timestamps

**Feature**: Each anomaly now includes precise timestamps indicating when it occurs in the video.

**Format**: `anomaly_type start_time end_time`

**Example**:
```
anomaly_timestamps: scene_mixing 31.95 56.99;video_combination 0.00 56.99;noise_injection 0.65 1.65;color_shift 0.85 1.85
```

### 4. Enhanced Anomaly Types

**New Anomaly Categories**:

#### Temporal Anomalies
- **frame_reversal**: Entire video played backwards
- **frame_shuffling**: Frames shuffled in chunks to maintain some coherence
- **speed_change**: Video speed increased (frames skipped) or decreased (frames duplicated)
- **temporal_cut**: Middle portion of video removed

#### Spatial Anomalies
- **object_insertion**: Synthetic objects inserted at random positions
- **color_shift**: RGB color values shifted randomly
- **geometric_distortion**: Perspective transforms applied
- **noise_injection**: Random noise added to frames

#### Semantic Anomalies
- **scene_mixing**: Combining videos from different scenes
- **video_combination**: Combining two different videos
- **activity_mixing**: Mixing different activities

### 5. Detailed Metadata Tracking

**Enhanced CSV Columns**:
- `original_script`: Script description of the original video
- `combined_video_id`: ID of the second video (if combined)
- `combined_script`: Script description of the second video (if combined)
- `decoded_actions`: Human-readable action descriptions with timestamps
- `anomaly_timestamps`: Precise timestamps for each anomaly
- `anomaly_count`: Number of different anomaly types applied

## File Structure

### Enhanced Files

1. **`sampled_train.csv`**
   - Now includes `decoded_actions` column
   - All action classes are translated to human-readable descriptions

2. **`anomaly_videos/anomaly_summary.csv`**
   - Enhanced with all new columns
   - Comprehensive tracking of combined videos
   - Detailed anomaly timestamps

3. **`generate_dataset.py`**
   - Enhanced anomaly generation with timestamp tracking
   - Improved video combination logic
   - Better metadata extraction

## Usage Examples

### Analyzing Combined Videos

```python
import pandas as pd

# Load the enhanced anomaly summary
df = pd.read_csv('anomaly_videos/anomaly_summary.csv')

# Find videos with scene mixing
scene_mixing_videos = df[df['anomalies_applied'].str.contains('scene_mixing', na=False)]

# Get details of combined videos
for _, row in scene_mixing_videos.iterrows():
    print(f"Original: {row['original_video_id']} ({row['original_scene']})")
    print(f"Combined: {row['combined_video_id']} ({row['combined_script']})")
    print(f"Anomaly: {row['anomaly_timestamps']}")
    print("---")
```

### Analyzing Anomaly Timestamps

```python
# Parse anomaly timestamps
def parse_anomaly_timestamps(timestamp_str):
    if pd.isna(timestamp_str):
        return []
    
    anomalies = []
    for part in timestamp_str.split(';'):
        if ' ' in part:
            anomaly_type, start_time, end_time = part.split(' ')[:3]
            anomalies.append({
                'type': anomaly_type,
                'start': float(start_time),
                'end': float(end_time)
            })
    return anomalies

# Analyze temporal distribution of anomalies
df['parsed_timestamps'] = df['anomaly_timestamps'].apply(parse_anomaly_timestamps)
```

### Understanding Decoded Actions

```python
# Compare original vs decoded actions
for _, row in df.iterrows():
    print(f"Video: {row['anomaly_video_name']}")
    print(f"Original actions: {row['original_actions']}")
    print(f"Decoded actions: {row['decoded_actions']}")
    print("---")
```

## Research Applications

### 1. Temporal Anomaly Analysis
- Track when specific anomalies occur in videos
- Analyze model responses to temporal inconsistencies
- Study how models handle reversed or shuffled content

### 2. Spatial Anomaly Detection
- Monitor model sensitivity to inserted objects
- Analyze responses to color and geometric distortions
- Study spatial reasoning capabilities

### 3. Semantic Anomaly Understanding
- Test model understanding of scene transitions
- Analyze responses to impossible combinations
- Study semantic reasoning in video understanding

### 4. Comprehensive Model Evaluation
- Compare model performance across different anomaly types
- Analyze confidence levels for different anomaly categories
- Study hallucination patterns in response to anomalies

## Technical Implementation

### Class Mapping System
```python
def _load_class_mappings(self):
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
```

### Timestamp Tracking
```python
def _apply_temporal_anomaly(self, frames, anomaly_type, fps=30.0):
    """Apply temporal anomalies and return timestamps."""
    anomaly_timestamps = []
    
    if anomaly_type == 'frame_reversal':
        anomaly_timestamps.append({
            'anomaly_type': 'frame_reversal',
            'start_time': 0.0,
            'end_time': len(frames) / fps,
            'description': 'Entire video played backwards'
        })
        return frames[::-1], anomaly_timestamps
```

## Benefits for Video LLM Testing

1. **Precise Anomaly Localization**: Know exactly when anomalies occur
2. **Comprehensive Metadata**: Full context for each anomaly video
3. **Human-Readable Actions**: Easy interpretation of video content
4. **Combined Video Tracking**: Understand semantic mixing effects
5. **Detailed Analysis**: Support for sophisticated model evaluation

## Future Enhancements

1. **Anomaly Severity Scoring**: Quantify the "strangeness" of anomalies
2. **Temporal Coherence Metrics**: Measure how well anomalies blend
3. **Semantic Consistency Checks**: Validate anomaly plausibility
4. **Interactive Analysis Tools**: Web-based visualization of anomalies
5. **Model Response Analysis**: Automated evaluation of model outputs

This enhanced system provides a comprehensive foundation for testing video language models' understanding of plausibility priors in both spatial and temporal dimensions.
