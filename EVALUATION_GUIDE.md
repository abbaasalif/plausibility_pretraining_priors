# Anomaly Video Evaluation Guide for Qwen2.5-VL

## Overview

This guide explains how to use the enhanced anomaly video evaluation system to test Qwen2.5-VL's ability to detect and describe anomalies in videos. The system is designed to comprehensively evaluate how well video language models handle temporal, spatial, and semantic anomalies.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_enhanced.txt
```

### 2. Run Simple Evaluation (Recommended for Testing)

```bash
python simple_anomaly_evaluation.py
```

This will:
- Load Qwen2.5-VL model
- Evaluate 5 videos by default
- Generate basic scores
- Save results to JSON file

### 3. Run Enhanced Evaluation (Full Analysis)

```bash
python enhanced_anomaly_evaluation.py
```

This will:
- Evaluate all videos (or configurable subset)
- Use 6 different question types
- Calculate 4 scoring metrics
- Generate visualizations
- Create comprehensive reports

## Evaluation System Architecture

### Question Types

The enhanced system uses 6 different question types to test various aspects:

1. **Basic Description**
   - Question: "Describe what you see in this video in detail. What activities are happening?"
   - Purpose: Test basic video understanding and content description

2. **Anomaly Detection**
   - Question: "Does anything seem unusual or out of place in this video? Are there any visual distortions or anomalies you notice?"
   - Purpose: Test general anomaly awareness

3. **Temporal Analysis**
   - Question: "Does the video appear to be playing normally or are there any temporal issues like reversed playback or shuffled frames?"
   - Purpose: Test temporal anomaly detection

4. **Spatial Analysis**
   - Question: "Do you see any objects that don't belong in this scene? Are there any color distortions or visual artifacts?"
   - Purpose: Test spatial anomaly detection

5. **Semantic Analysis**
   - Question: "Does the sequence of events make logical sense? Are there any scene transitions that seem abrupt or unnatural?"
   - Purpose: Test semantic anomaly detection

6. **Confidence Assessment**
   - Question: "How confident are you in your description? Are there any parts you're uncertain about?"
   - Purpose: Test confidence calibration

### Scoring Metrics

The system calculates 4 different scores for each video:

1. **Content Accuracy (0-10)**
   - Measures how well the model describes the original content
   - Based on keyword overlap with ground truth script and actions
   - Higher scores indicate better content understanding

2. **Anomaly Detection (0-10)**
   - Measures how accurately the model identifies anomalies
   - Based on detection of expected anomalies in the video
   - Higher scores indicate better anomaly awareness

3. **Confidence Calibration (0-10)**
   - Measures how well the model expresses uncertainty
   - Based on balance of confidence vs uncertainty indicators
   - Higher scores indicate better self-awareness

4. **Hallucination Score (0-10)**
   - Measures how much the model makes up explanations
   - Based on objects/actions mentioned but not in ground truth
   - Higher scores indicate less hallucination

## Configuration Options

### Simple Evaluation Configuration

Edit `simple_anomaly_evaluation.py`:

```python
# Configuration
CSV_PATH = "anomaly_videos/anomaly_summary.csv"
VIDEO_DIR = "anomaly_videos"
MAX_VIDEOS = 5  # Set to None to evaluate all videos
```

### Enhanced Evaluation Configuration

Edit `enhanced_anomaly_evaluation.py`:

```python
# Configuration
CSV_PATH = "anomaly_videos/anomaly_summary.csv"
VIDEO_DIR = "anomaly_videos"
MAX_VIDEOS = 10  # Set to None to evaluate all videos
```

## Output Files

### Simple Evaluation Output

- `simple_evaluation_results_YYYYMMDD_HHMMSS.json`
  - Contains detailed results for each video
  - Includes responses and scores
  - Easy to parse and analyze

### Enhanced Evaluation Output

- `evaluation_results/detailed_results_YYYYMMDD_HHMMSS.json`
  - Complete evaluation data
  - All responses and scores
  - Anomaly type analysis

- `evaluation_results/summary_scores_YYYYMMDD_HHMMSS.csv`
  - Tabular summary of all scores
  - Easy to import into analysis tools

- `evaluation_results/evaluation_plots_YYYYMMDD_HHMMSS.png`
  - Visualizations of results
  - Score distributions and correlations
  - Performance by anomaly type

## Interpreting Results

### Score Interpretation

- **8-10**: Excellent performance
- **6-8**: Good performance
- **4-6**: Moderate performance
- **2-4**: Poor performance
- **0-2**: Very poor performance

### Expected Patterns

1. **Content Accuracy**: Should be high for videos without severe anomalies
2. **Anomaly Detection**: Should correlate with anomaly complexity
3. **Confidence Calibration**: Should be moderate (not overconfident)
4. **Hallucination Score**: Should be high (few made-up elements)

### Anomaly Type Analysis

The system categorizes anomalies into three types:

1. **Temporal Anomalies**
   - Frame reversal, shuffling, speed changes
   - Models should detect temporal inconsistencies

2. **Spatial Anomalies**
   - Object insertions, color shifts, geometric distortions
   - Models should detect visual artifacts

3. **Semantic Anomalies**
   - Scene mixing, video combinations
   - Models should detect logical inconsistencies

## Customization

### Adding New Question Types

Edit the `questions` dictionary in `AnomalyVideoEvaluator.__init__()`:

```python
self.questions = {
    "your_new_question": "Your question text here",
    # ... existing questions
}
```

### Modifying Scoring Metrics

Edit the scoring functions in the evaluator:

```python
def calculate_your_score(self, response, ground_truth):
    # Your scoring logic here
    return score
```

### Adding New Anomaly Types

Update the `categorize_anomalies` function:

```python
def categorize_anomalies(self, anomalies):
    categories = {
        "temporal": [],
        "spatial": [],
        "semantic": [],
        "your_new_category": []  # Add new category
    }
    # ... categorization logic
    return categories
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Ensure sufficient GPU memory
   - Check CUDA installation
   - Try reducing model precision

2. **Video Processing Errors**
   - Verify video file paths
   - Check video format compatibility
   - Ensure sufficient disk space

3. **Memory Issues**
   - Reduce `max_videos` parameter
   - Process videos in smaller batches
   - Use CPU offloading if available

### Performance Optimization

1. **Faster Evaluation**
   - Reduce `max_new_tokens` in generation
   - Use lower video resolution
   - Process videos in parallel (if memory allows)

2. **Better Results**
   - Increase `max_new_tokens` for longer responses
   - Use higher video resolution
   - Add more question types

## Research Applications

### Model Comparison

Use the evaluation system to compare different models:

1. Run evaluation on same videos with different models
2. Compare scores across models
3. Analyze strengths and weaknesses

### Ablation Studies

Test different components:

1. Remove specific question types
2. Modify scoring metrics
3. Test different video preprocessing

### Dataset Analysis

Analyze the anomaly dataset:

1. Which anomaly types are hardest to detect?
2. How do different scenes affect performance?
3. What are the most challenging videos?

## Citation

If you use this evaluation system in your research, please cite:

```bibtex
@misc{anomaly_video_evaluation_2024,
  title={Enhanced Anomaly Video Evaluation for Video Language Models},
  author={Your Name},
  year={2024},
  note={Comprehensive evaluation system for testing video LLM robustness to temporal, spatial, and semantic anomalies}
}
```

## Support

For questions or issues:

1. Check the troubleshooting section
2. Review the code comments
3. Examine the example outputs
4. Modify the configuration as needed

The system is designed to be modular and extensible, so you can easily adapt it to your specific research needs.


