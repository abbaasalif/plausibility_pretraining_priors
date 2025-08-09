# Plausibility Priors Testing System - Summary

## What Was Created

We've successfully created a comprehensive system for testing plausibility priors in video language models using the Charades dataset. The system addresses the key insight that "Video is the only modality where anomalies can be present in both space and time simultaneously."

## Files Created

### Core Scripts
1. **`random_sample.py`** - Samples 100 videos from the Charades dataset
2. **`generate_dataset.py`** - Generates synthetic anomaly videos for testing
3. **`requirements.txt`** - Dependencies for the system
4. **`README.md`** - Comprehensive documentation
5. **`SUMMARY.md`** - This summary document

### Generated Data
- **`sampled_videos/`** - 100 randomly sampled videos from Charades
- **`sampled_train.csv`** - Metadata for sampled videos
- **`anomaly_videos/`** - 100 synthetic anomaly videos
- **`anomaly_summary.csv`** - Detailed anomaly information
- **`dataset_metadata.json`** - Dataset statistics
- **`synthetic_objects/`** - Synthetic objects for insertion anomalies

## Key Features

### Anomaly Types Implemented

#### Temporal Anomalies
- **Frame Reversal**: Playing video backwards
- **Frame Shuffling**: Random temporal order
- **Speed Changes**: Unnatural playback speeds
- **Temporal Cuts**: Missing middle portions

#### Spatial Anomalies
- **Object Insertion**: Synthetic objects in scenes
- **Color Shifts**: Unnatural color changes
- **Geometric Distortion**: Perspective changes
- **Noise Injection**: Artificial noise

#### Semantic Anomalies
- **Activity Mixing**: Combining different activities
- **Scene Mixing**: Combining different scenes
- **Action Reversal**: Impossible action sequences

### Dataset Statistics
- **100 anomaly videos** generated
- **61 original videos** used as sources
- **Average 5.4 anomalies** per video
- **8 different scene types** represented
- **Multiple anomaly combinations** tested

## How to Use

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Sample Videos (if needed)
```bash
python random_sample.py
```

### Step 3: Generate Anomaly Videos
```bash
python generate_dataset.py
```

### Step 4: Test Video LLMs
Use the generated anomaly videos to test video language models and compare their responses to:
- Original videos (baseline)
- Anomaly videos (test cases)

## Research Applications

This system enables testing of:

1. **Anomaly Detection**: Can models identify when something is wrong?
2. **Explanation Quality**: How well do models explain anomalies?
3. **Confidence Calibration**: Do models express uncertainty about anomalies?
4. **Hallucination Rate**: Do models make up explanations for anomalies?

## Expected Failure Modes

Models should struggle with:
- **Temporal inconsistencies**: "The person is walking backwards" (when video is reversed)
- **Spatial impossibilities**: "There's a floating red square" (inserted objects)
- **Semantic contradictions**: "Someone is cooking in a bedroom" (scene mixing)

## Key Insights

The system tests the hypothesis that video LLMs are "trapped by training expectations" when encountering:
- **Temporal anomalies**: Events happening in impossible sequences
- **Spatial anomalies**: Objects or scenes that violate physical laws
- **Semantic anomalies**: Activities that don't make sense in context

This creates unique failure modes where model elements that wouldn't necessarily align with the general sequence of events in both space and time can cause the model to fail.

## Next Steps

1. **Test with Video LLMs**: Use the generated videos to test existing video language models
2. **Analyze Responses**: Compare model responses to original vs. anomaly videos
3. **Identify Failure Patterns**: Document specific failure modes
4. **Improve Models**: Use insights to improve video LLM robustness
5. **Extend Anomaly Types**: Add more sophisticated anomaly types

## Files Structure
```
charades_pretraining_priors/
├── random_sample.py
├── generate_dataset.py
├── requirements.txt
├── README.md
├── SUMMARY.md
├── sampled_videos/          # 100 sampled videos
├── sampled_train.csv        # Sample metadata
├── anomaly_videos/          # 100 anomaly videos
│   ├── anomaly_video_0000.mp4
│   ├── ...
│   ├── anomaly_summary.csv
│   └── dataset_metadata.json
└── synthetic_objects/       # Synthetic objects for insertion
    ├── object_0.png
    └── ...
```

The system is now ready for testing video language models and investigating plausibility priors in video understanding!
