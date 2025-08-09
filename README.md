# Plausibility Priors Testing for Video LLMs

This project generates synthetic anomaly videos to test whether video language models are "trapped by training expectations" when encountering anomalies in both space and time. The system provides comprehensive evaluation tools for testing video LLMs like Qwen2.5-VL on temporal, spatial, and semantic anomalies.

## 🎯 Overview

Video is the only modality where anomalies can be present in both space and time simultaneously. This creates unique failure modes for video LLMs where model elements that wouldn't necessarily align with the general sequence of events in both dimensions can cause the model to fail.

### Key Research Question
**Are video language models "trapped by training expectations" when encountering anomalies in both space and time?**

## 🔍 Key Concepts

### Plausibility Priors
Models trained on large video datasets develop strong priors about what constitutes "normal" video content. These priors can trap models when they encounter:
- **Temporal anomalies**: Events happening in impossible sequences
- **Spatial anomalies**: Objects or scenes that violate physical laws
- **Semantic anomalies**: Activities that don't make sense in context

### Failure Modes Tested

1. **Temporal Anomalies**:
   - Frame reversal (playing video backwards)
   - Frame shuffling (random temporal order)
   - Speed changes (unnatural playback speeds)
   - Temporal cuts (missing middle portions)

2. **Spatial Anomalies**:
   - Object insertions (synthetic objects in scenes)
   - Color shifts (unnatural color changes)
   - Geometric distortions (perspective changes)
   - Noise injection (artificial noise)

3. **Semantic Anomalies**:
   - Activity mixing (combining different activities)
   - Scene mixing (combining different scenes)
   - Action reversal (impossible action sequences)

## 🚀 Enhanced Features

### 1. Combined Video Information
When videos are combined to create semantic anomalies, the system tracks:
- **Combined Video ID**: The ID of the second video used in the combination
- **Combined Script**: The script description of the second video
- **Scene Transition Information**: Details about how scenes change from one video to another

**Example**:
```
combined_video_id: FQA3W
combined_script: "A person in their bedroom is playing with their shoes. Once they have put their shoes on, they start briskly running towards the door and grab onto the doorknob."
```

### 2. Decoded Action Classes
The system decodes action class IDs into human-readable descriptions using the Charades class mappings.

**Before**:
```
actions: c134 0.00 6.90;c020 12.00 31.00;c063 16.80 22.30
```

**After**:
```
decoded_actions: c134 0.00 6.90 (Lying on a bed);c020 12.00 31.00 (Holding a bag);c063 16.80 22.30 (Taking food from somewhere)
```

### 3. Anomaly Timestamps
Each anomaly includes precise timestamps indicating when it occurs in the video.

**Format**: `anomaly_type start_time end_time`

**Example**:
```
anomaly_timestamps: scene_mixing 31.95 56.99;video_combination 0.00 56.99;noise_injection 0.65 1.65;color_shift 0.85 1.85
```

### 4. Detailed Metadata Tracking
Enhanced CSV columns include:
- `original_script`: Script description of the original video
- `combined_video_id`: ID of the second video (if combined)
- `combined_script`: Script description of the second video (if combined)
- `decoded_actions`: Human-readable action descriptions with timestamps
- `anomaly_timestamps`: Precise timestamps for each anomaly
- `anomaly_count`: Number of different anomaly types applied

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for video processing)
- Sufficient disk space (at least 10GB for full dataset)

### Install Dependencies

```bash
# Install all dependencies (includes both basic and enhanced features)
pip install -r requirements.txt
```

### Environment Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd charades_pretraining_priors
```

2. **Set up environment variables** (optional):
```bash
# Create .env file for OpenAI API (for enhanced evaluation)
echo "OPENAI_API_KEY=your_api_key_here" > .env
echo "OPENAI_JUDGE_MODEL=gpt-4o-mini" >> .env
```

3. **Download Charades dataset**:
   - Place `Charades_v1_480.zip` in the project root
   - Extract `Charades/` directory with metadata files

## 🎬 Usage

### Step 1: Generate Sample Videos
First, run the random sampling script to create a smaller dataset:

```bash
python random_sample.py
```

This will:
- Randomly sample 100 videos from the Charades dataset
- Extract them to `sampled_videos/` directory
- Create `sampled_train.csv` with metadata including decoded actions
- Show progress and summary statistics

### Step 2: Generate Anomaly Videos
Run the anomaly generation script:

```bash
python generate_dataset.py
```

This will:
- Generate 100 anomaly videos in `anomaly_videos/` directory
- Create synthetic objects for insertion
- Apply various temporal, spatial, and semantic anomalies
- Save detailed metadata and summaries with timestamps
- Show progress and final statistics

### Step 3: Evaluate Video LLMs

#### Option A: Simple Evaluation (Quick Testing)
```bash
python simple_anomaly_evaluation.py
```

**Features**:
- Basic description and anomaly detection questions
- Automated scoring based on content accuracy and anomaly detection
- JSON output with detailed results
- Configurable number of videos to test
- **Incremental saving**: Results saved after each video to prevent data loss
- **Resume capability**: Automatically resumes from where it left off if interrupted
- **Progress monitoring**: Real-time progress updates and statistics
- **Individual response files**: Each video's responses saved separately for detailed inspection

**Output Structure**:
```
simple_evaluation_results/
├── incremental_YYYYMMDD_HHMMSS/
│   ├── incremental_results.json          # Complete results after each video
│   ├── progress_summary.json             # Current progress statistics
│   ├── current_summary.csv               # CSV summary of all completed videos
│   ├── simple_evaluation_results_YYYYMMDD_HHMMSS.json  # Final results
│   └── video_responses/                  # Individual video response files
│       ├── anomaly_video_0000_responses.json
│       ├── anomaly_video_0001_responses.json
│       └── ...
```

**Key Benefits**:
- **No data loss**: Results saved after each video evaluation
- **Automatic resume**: Script detects existing results and continues from where it left off
- **Progress monitoring**: Check progress without running the full evaluation
- **Detailed inspection**: Individual response files for each video
- **Real-time statistics**: Live updates of average scores and progress

#### Option B: Enhanced Evaluation (Comprehensive Analysis)
```bash
# Test mode (single video for debugging)
python enhanced_anomaly_evaluation.py --test

# Normal mode with verbose output
python enhanced_anomaly_evaluation.py --verbose --max-videos 10

# Full evaluation
python enhanced_anomaly_evaluation.py
```

**Features**:
- **Multiple Question Types**: 6 different question categories
- **Automated Scoring**: 4 different scoring metrics
- **Visualization**: Automatic plot generation
- **Detailed Analysis**: Performance breakdown by anomaly type
- **Comprehensive Reporting**: JSON, CSV, and visual outputs
- **Incremental Saving**: Results saved after each video to prevent data loss
- **Resume Capability**: Automatically resumes from where it left off if interrupted
- **Individual Response Files**: Each video's responses saved separately for detailed inspection

#### Option C: Progress Monitoring
```bash
python check_progress.py
```

**Features**:
- Display current progress and statistics
- Show best and worst performing videos
- List recently completed videos
- Check average scores across all metrics

## 📊 Output Structure

```
charades_pretraining_priors/
├── random_sample.py                    # Video sampling script
├── generate_dataset.py                 # Anomaly generation script
├── simple_anomaly_evaluation.py        # Simple evaluation
├── enhanced_anomaly_evaluation.py      # Enhanced evaluation
├── check_progress.py                   # Progress monitoring
├── requirements.txt                    # All dependencies (basic + enhanced)
├── README.md                          # This file
├── ENHANCED_FEATURES.md               # Enhanced features documentation
├── EVALUATION_GUIDE.md                # Evaluation guide
├── SUMMARY.md                         # Project summary
│
├── sampled_videos/                    # 100 sampled videos
│   ├── TETZ7.mp4
│   ├── FQA3W.mp4
│   └── ...
├── sampled_train.csv                  # Enhanced metadata with decoded actions
│
├── anomaly_videos/                    # 100 synthetic anomaly videos
│   ├── anomaly_video_0000.mp4
│   ├── anomaly_video_0001.mp4
│   ├── ...
│   ├── anomaly_summary.csv            # Enhanced anomaly information with timestamps
│   └── dataset_metadata.json          # Dataset statistics and metadata
│
├── synthetic_objects/                 # Synthetic objects for insertion
│   ├── object_0.png
│   ├── object_1.png
│   └── ...
│
├── simple_evaluation_results/         # Simple evaluation outputs
│   ├── incremental_YYYYMMDD_HHMMSS/   # Incremental results
│   │   ├── incremental_results.json
│   │   ├── progress_summary.json
│   │   ├── current_summary.csv
│   │   ├── simple_evaluation_results_YYYYMMDD_HHMMSS.json  # Final results
│   │   └── video_responses/
│   │       ├── anomaly_video_0000_responses.json
│   │       └── ...
│   └── ...
│
└── evaluation_results/                # Enhanced evaluation outputs
    ├── incremental_YYYYMMDD_HHMMSS/   # Incremental results
    │   ├── incremental_results.json
    │   ├── progress_summary.json
    │   ├── current_summary.csv
    │   └── video_responses/
    │       ├── anomaly_video_0000_responses.json
    │       └── ...
    └── final_results_YYYYMMDD_HHMMSS/
        ├── detailed_results.json
        ├── summary_scores.csv
        └── evaluation_plots.png
```

## 🔍 Dataset Files

### anomaly_summary.csv
Contains comprehensive information about each generated anomaly video:
- `anomaly_video_name`: Name of the generated video
- `original_video_id`: ID of the source video
- `original_scene`: Scene type (bedroom, kitchen, etc.)
- `original_script`: Script description of the original video
- `combined_video_id`: ID of the second video (if combined)
- `combined_script`: Script description of the second video (if combined)
- `original_actions`: Original action annotations
- `decoded_actions`: Human-readable action descriptions with timestamps
- `anomalies_applied`: List of applied anomalies
- `anomaly_timestamps`: Precise timestamps for each anomaly
- `anomaly_count`: Number of different anomaly types applied

### sampled_train.csv
Enhanced metadata for sampled videos:
- All original Charades columns
- `decoded_actions`: Human-readable action descriptions with timestamps

### dataset_metadata.json
Contains dataset-level statistics:
- Total number of anomaly videos
- Distribution of anomaly types
- Scene distribution
- Original videos used
- Anomaly type statistics

## 🧠 Enhanced Evaluation System

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
   - Based on ChatGPT evaluation of content coverage, accuracy, and detail level
   - Higher scores indicate better content understanding

2. **Anomaly Detection (0-10)**
   - Measures how accurately the model identifies anomalies
   - Based on ChatGPT evaluation of anomaly detection, accuracy, and false positives
   - Higher scores indicate better anomaly awareness

3. **Confidence Calibration (0-10)**
   - Measures how well the model expresses uncertainty
   - Based on ChatGPT evaluation of uncertainty recognition, confidence appropriateness, and calibration
   - Higher scores indicate better self-awareness

4. **Hallucination Score (0-10)**
   - Measures how much the model makes up explanations
   - Based on ChatGPT evaluation of hallucination severity, factual accuracy, and speculation control
   - Higher scores indicate less hallucination

### ChatGPT Judge System

The enhanced evaluation uses **ChatGPT as the sole judge** for all scoring:

- **No manual or keyword-based scoring**
- **Comprehensive JSON-based evaluation**
- **Robust error handling and retries**
- **Detailed assessment feedback**

#### Judge Prompts
Each scoring metric uses specialized prompts that:
- Provide clear evaluation criteria
- Request specific JSON format responses
- Include ground truth information
- Generate detailed assessments

#### Error Handling
- Automatic retries with exponential backoff
- Graceful fallbacks for failed evaluations
- Detailed error logging and debugging
- JSON parsing with cleanup for common issues

## 🔧 Debugging and Testing

### Test Mode
Run a single video evaluation for debugging:
```bash
python enhanced_anomaly_evaluation.py --test
```

**Features**:
- Isolated testing with one video
- Full response printing for debugging
- Detailed JSON parsing steps
- Error tracing and diagnostics

### Debug Features
- **Full Response Printing**: All model responses printed in plain text
- **JSON Parsing Debug**: Shows raw JSON responses and parsing steps
- **Error Tracing**: Detailed error messages with tracebacks
- **Progress Monitoring**: Real-time progress updates
- **Incremental Saving**: Results saved after each video

### Simple Evaluation Debugging
The simple evaluation script also includes robust debugging features:
- **Automatic resume**: Continues from where it left off if interrupted
- **Progress tracking**: Real-time progress updates and statistics
- **Error handling**: Graceful error handling with detailed logging
- **Incremental saving**: Results saved after each video to prevent data loss
- **Individual response files**: Each video's responses saved separately for inspection

### Common Issues and Solutions

1. **Model Loading Errors**
   - Ensure sufficient GPU memory (at least 8GB recommended)
   - Check CUDA installation and compatibility
   - Try reducing model precision or using CPU offloading

2. **Video Processing Errors**
   - Verify video file paths and permissions
   - Check video format compatibility (MP4 recommended)
   - Ensure sufficient disk space for processing

3. **Memory Issues**
   - Reduce `max_videos` parameter
   - Process videos in smaller batches
   - Use CPU offloading if available
   - Close other applications to free memory

4. **JSON Parsing Errors**
   - Check OpenAI API key and quota
   - Verify internet connection
   - Review error logs for specific issues
   - Use test mode for debugging

## 📈 Research Applications

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

### 5. Model Comparison Studies
- Compare different video LLMs on the same anomaly videos
- Analyze strengths and weaknesses across models
- Identify common failure patterns

## 🎯 Expected Failure Modes

Models should struggle with:
- **Temporal inconsistencies**: "The person is walking backwards" (when video is reversed)
- **Spatial impossibilities**: "There's a floating red square" (inserted objects)
- **Semantic contradictions**: "Someone is cooking in a bedroom" (scene mixing)

## 📊 Evaluation Metrics

1. **Anomaly Detection**: Can the model identify when something is wrong?
2. **Explanation Quality**: How well does the model explain anomalies?
3. **Confidence Calibration**: Does the model express uncertainty about anomalies?
4. **Hallucination Rate**: Does the model make up explanations for anomalies?
5. **Temporal Awareness**: Can the model identify when anomalies occur in time?
6. **Spatial Reasoning**: Can the model identify where anomalies occur in space?

## 🔄 Incremental Saving and Resume Feature

Both evaluation systems include robust incremental saving to prevent data loss and enable resuming from interruptions:

### Simple Evaluation System
```
simple_evaluation_results/
├── incremental_YYYYMMDD_HHMMSS/
│   ├── incremental_results.json          # Complete results after each video
│   ├── progress_summary.json             # Current progress statistics
│   ├── current_summary.csv               # CSV summary of all completed videos
│   ├── simple_evaluation_results_YYYYMMDD_HHMMSS.json  # Final results
│   └── video_responses/                  # Individual video response files
│       ├── anomaly_video_0000_responses.json
│       ├── anomaly_video_0001_responses.json
│       └── ...
```

### Enhanced Evaluation System
```
evaluation_results/
├── incremental_YYYYMMDD_HHMMSS/
│   ├── incremental_results.json          # Complete results after each video
│   ├── progress_summary.json             # Current progress statistics
│   ├── current_summary.csv               # CSV summary of all completed videos
│   └── video_responses/                  # Individual video response files
│       ├── anomaly_video_0000_responses.json
│       ├── anomaly_video_0001_responses.json
│       └── ...
└── final_results_YYYYMMDD_HHMMSS/
    ├── detailed_results.json
    ├── summary_scores.csv
    └── evaluation_plots.png
```

### Key Benefits
- **No Data Loss**: Results saved after each video evaluation
- **Automatic Resume**: Script detects existing results and continues from where it left off
- **Progress Monitoring**: Check progress without running the full evaluation
- **Detailed Inspection**: Individual response files for each video
- **Real-time Statistics**: Live updates of average scores and progress

### Usage Examples
```bash
# Start simple evaluation (will create incremental directory)
python simple_anomaly_evaluation.py

# Start enhanced evaluation (will create incremental directory)
python enhanced_anomaly_evaluation.py

# Check progress without running evaluation
python check_progress.py

# Resume evaluation (automatically detected)
python simple_anomaly_evaluation.py
python enhanced_anomaly_evaluation.py

# Start fresh (delete incremental directory first)
rm -rf simple_evaluation_results/incremental_*
rm -rf evaluation_results/incremental_*
python simple_anomaly_evaluation.py
python enhanced_anomaly_evaluation.py
```

## 🛠️ Customization

### Modifying Anomaly Types
Edit the `anomaly_types` dictionary in `AnomalyVideoGenerator.__init__()`:

```python
self.anomaly_types = {
    'temporal': {
        'frame_reversal': 0.3,      # Probability of applying this anomaly
        'frame_shuffling': 0.4,
        # Add new temporal anomalies here
    },
    'spatial': {
        'object_insertion': 0.4,
        # Add new spatial anomalies here
    },
    'semantic': {
        'activity_mixing': 0.3,
        # Add new semantic anomalies here
    }
}
```

### Adding New Anomaly Types
1. Add the anomaly type to the appropriate category in `anomaly_types`
2. Implement the anomaly logic in `_apply_temporal_anomaly()` or `_apply_spatial_anomaly()`
3. Update the documentation

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

## 🔬 Technical Implementation

### Class Mapping System
The system automatically loads class mappings from `Charades/Charades_v1_classes.txt` to decode action IDs into human-readable descriptions.

### Timestamp Tracking
Each anomaly application includes precise timestamp tracking to know exactly when anomalies occur in the video timeline.

### Video Combination Logic
When combining videos for semantic anomalies, the system:
1. Loads two different videos
2. Combines their frames
3. Tracks metadata from both videos
4. Records transition points and timestamps

### JSON Parsing and Error Handling
- Robust JSON parsing with cleanup for common issues
- Automatic retries with exponential backoff
- Graceful fallbacks for failed evaluations
- Detailed error logging and debugging

## 📚 Usage Examples

### Running Simple Evaluation
```bash
# First run (creates new incremental directory)
python simple_anomaly_evaluation.py

# Resume after interruption (automatically detects existing results)
python simple_anomaly_evaluation.py

# Start fresh (delete existing results first)
rm -rf simple_evaluation_results/incremental_*
python simple_anomaly_evaluation.py
```

### Running Enhanced Evaluation
```bash
# Test mode (single video for debugging)
python enhanced_anomaly_evaluation.py --test

# Normal mode with verbose output
python enhanced_anomaly_evaluation.py --verbose --max-videos 10

# Full evaluation
python enhanced_anomaly_evaluation.py

# Resume after interruption
python enhanced_anomaly_evaluation.py
```

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

## 🎯 Benefits for Video LLM Testing

1. **Precise Anomaly Localization**: Know exactly when anomalies occur
2. **Comprehensive Metadata**: Full context for each anomaly video
3. **Human-Readable Actions**: Easy interpretation of video content
4. **Combined Video Tracking**: Understand semantic mixing effects
5. **Detailed Analysis**: Support for sophisticated model evaluation
6. **Temporal Awareness**: Track when anomalies occur in time
7. **Spatial Understanding**: Know where anomalies occur in space
8. **Robust Evaluation Systems**: Both simple and enhanced evaluation with incremental saving
9. **Automatic Resume**: Continue from where you left off if interrupted
10. **Progress Monitoring**: Real-time progress tracking and statistics
11. **Individual Response Files**: Detailed inspection of each video's responses
12. **No Data Loss**: Results saved after each video evaluation

## 🔮 Future Enhancements

1. **Anomaly Severity Scoring**: Quantify the "strangeness" of anomalies
2. **Temporal Coherence Metrics**: Measure how well anomalies blend
3. **Semantic Consistency Checks**: Validate anomaly plausibility
4. **Interactive Analysis Tools**: Web-based visualization of anomalies
5. **Model Response Analysis**: Automated evaluation of model outputs
6. **Multi-Modal Evaluation**: Test audio-visual models
7. **Real-time Evaluation**: Live evaluation during video processing

## 📄 Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{plausibility_priors_video_llms,
  title={Plausibility Priors Testing for Video Language Models},
  author={Your Name},
  year={2024},
  note={Dataset for testing video LLM robustness to temporal and spatial anomalies}
}
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions or issues:

1. Check the troubleshooting section
2. Review the code comments
3. Examine the example outputs
4. Modify the configuration as needed
5. Open an issue on GitHub

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Charades dataset for providing the base video content
- OpenAI for providing the ChatGPT API for evaluation
- The video processing community for tools and libraries
- Contributors and researchers who provided feedback

---

**Note**: This system is designed to be modular and extensible, so you can easily adapt it to your specific research needs. The comprehensive documentation and debugging features make it suitable for both research and educational purposes.

## 📊 EVALUATION SUMMARY

### Simple Evaluation System
- **Purpose**: Quick testing and validation of video LLMs
- **Features**: Basic description and anomaly detection questions
- **Output**: JSON results with incremental saving
- **Resume**: Automatic resume from interruptions
- **Progress**: Real-time progress monitoring
- **Directory**: `simple_evaluation_results/`

### Enhanced Evaluation System
- **Purpose**: Comprehensive analysis and research
- **Features**: 6 question types, 4 scoring metrics, ChatGPT judge
- **Output**: JSON, CSV, and visual outputs
- **Resume**: Automatic resume from interruptions
- **Progress**: Real-time progress monitoring
- **Directory**: `evaluation_results/`

### Key Metrics
1. **Content Accuracy (0-10)**: How well the model describes the original content
2. **Anomaly Detection (0-10)**: How accurately the model identifies anomalies
3. **Confidence Calibration (0-10)**: How well the model expresses uncertainty
4. **Hallucination Score (0-10)**: How much the model makes up explanations
