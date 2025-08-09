"""
Simple Anomaly Video Evaluation for Qwen2.5-VL
==============================================

A simplified version of the enhanced evaluation script for testing Qwen2.5-VL
on anomaly videos. This version focuses on core functionality and can be easily
modified and extended.

Usage:
    python simple_anomaly_evaluation.py
"""

import torch
import pandas as pd
import numpy as np
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
import re
import json
import time
from datetime import datetime, timedelta

def normalize_text(text):
    """Normalize text for comparison."""
    return re.sub(r"[^a-z0-9]", " ", text.lower()).strip()

def load_model():
    """Load Qwen2.5-VL model."""
    print("🔄 Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("✅ Model loaded successfully")
    return model, processor

def generate_response(model, processor, video_path, question):
    """Generate response from Qwen2.5-VL."""
    try:
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video_path),
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": question},
            ],
        }]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        
        return output_text.strip()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error: {str(e)}"

def calculate_scores(response, ground_truth):
    """Calculate evaluation scores."""
    scores = {}
    response_lower = response.lower()
    
    # Content Accuracy (0-10)
    content_score = 0.0
    if ground_truth.get('original_script'):
        script_keywords = set(normalize_text(ground_truth['original_script']).split())
        response_words = set(normalize_text(response).split())
        script_overlap = len(script_keywords.intersection(response_words))
        if len(script_keywords) > 0:
            content_score = (script_overlap / len(script_keywords)) * 10.0
    
    scores['content_accuracy'] = min(content_score, 10.0)
    
    # Anomaly Detection (0-10)
    anomaly_score = 0.0
    expected_anomalies = []
    if ground_truth.get('anomalies_applied'):
        expected_anomalies = ground_truth['anomalies_applied'].split(';')
    
    if expected_anomalies:
        detected_count = 0
        for anomaly in expected_anomalies:
            anomaly_clean = normalize_text(anomaly)
            if anomaly_clean in response_lower or any(keyword in response_lower for keyword in anomaly_clean.split('_')):
                detected_count += 1
        
        anomaly_score = (detected_count / len(expected_anomalies)) * 10.0
    
    scores['anomaly_detection'] = anomaly_score
    
    # Overall Score
    scores['overall'] = np.mean([scores['content_accuracy'], scores['anomaly_detection']])
    
    return scores

def evaluate_video(model, processor, video_path, ground_truth):
    """Evaluate a single video."""
    print(f"🔍 Evaluating: {ground_truth['anomaly_video_name']}")
    
    # Question 1: Basic description
    question1 = "Describe what you see in this video in detail. What activities are happening?"
    response1 = generate_response(model, processor, video_path, question1)
    print(f"  📝 Basic description: {response1[:100]}...")
    
    # Question 2: Anomaly detection
    question2 = "Does anything seem unusual or out of place in this video? Are there any visual distortions or anomalies you notice?"
    response2 = generate_response(model, processor, video_path, question2)
    print(f"  🔍 Anomaly detection: {response2[:100]}...")
    
    # Calculate scores based on anomaly detection response
    scores = calculate_scores(response2, ground_truth)
    
    result = {
        'video_name': ground_truth['anomaly_video_name'],
        'original_script': ground_truth.get('original_script', ''),
        'expected_anomalies': ground_truth.get('anomalies_applied', ''),
        'basic_description': response1,
        'anomaly_detection': response2,
        'scores': scores
    }
    
    return result

def save_incremental_results(results, output_dir, current_idx, total_videos):
    """Save incremental results to prevent data loss."""
    try:
        # Save complete results
        with open(output_dir / "incremental_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        if results:
            # Save progress summary
            progress_summary = {
                'timestamp': datetime.now().isoformat(),
                'progress': f"{current_idx}/{total_videos}",
                'completed_videos': len(results),
                'latest_video': results[-1]['video_name'],
                'latest_scores': results[-1]['scores'],
                'average_overall_score': float(np.mean([r['scores']['overall'] for r in results])),
                'average_content_accuracy': float(np.mean([r['scores']['content_accuracy'] for r in results])),
                'average_anomaly_detection': float(np.mean([r['scores']['anomaly_detection'] for r in results]))
            }
            
            with open(output_dir / "progress_summary.json", 'w') as f:
                json.dump(progress_summary, f, indent=2)
            
            # Save CSV summary
            summary_data = [{
                'video_name': r['video_name'],
                'content_accuracy': r['scores']['content_accuracy'],
                'anomaly_detection': r['scores']['anomaly_detection'],
                'overall_score': r['scores']['overall']
            } for r in results]
            
            pd.DataFrame(summary_data).to_csv(output_dir / "current_summary.csv", index=False)
            
            # Save individual video response
            latest = results[-1]
            vr_dir = output_dir / "video_responses"
            vr_dir.mkdir(exist_ok=True)
            
            with open(vr_dir / f"{latest['video_name'].replace('.mp4','')}_responses.json", 'w') as f:
                json.dump({
                    'video_name': latest['video_name'],
                    'original_script': latest['original_script'],
                    'expected_anomalies': latest['expected_anomalies'],
                    'basic_description': latest['basic_description'],
                    'anomaly_detection': latest['anomaly_detection'],
                    'scores': latest['scores'],
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
        
        print(f"💾 Saved incremental results ({len(results)} videos completed)")
        
    except Exception as e:
        print(f"⚠️  Error saving incremental results: {e}")

def find_latest_incremental_results():
    """Find the latest incremental results directory."""
    eval_dir = Path("simple_evaluation_results")
    if not eval_dir.exists():
        return None
    
    incremental_dirs = [d for d in eval_dir.iterdir() if d.is_dir() and d.name.startswith("incremental_")]
    return max(incremental_dirs, key=lambda x: x.stat().st_mtime) if incremental_dirs else None

def display_progress_summary(incremental_dir):
    """Display progress summary from existing results."""
    progress_file = incremental_dir / "progress_summary.json"
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            print(f"📊 Current Progress:")
            print(f"  Progress: {progress['progress']}")
            print(f"  Completed videos: {progress['completed_videos']}")
            print(f"  Latest video: {progress['latest_video']}")
            print(f"  Average overall score: {progress['average_overall_score']:.2f}")
            print(f"  Average content accuracy: {progress['average_content_accuracy']:.2f}")
            print(f"  Average anomaly detection: {progress['average_anomaly_detection']:.2f}")
            print(f"  Last updated: {progress['timestamp']}")
        except Exception as e:
            print(f"⚠️  Could not read progress summary: {e}")

def main():
    """Main evaluation function."""
    # Configuration
    CSV_PATH = "anomaly_videos/anomaly_summary.csv"
    VIDEO_DIR = "anomaly_videos"
    MAX_VIDEOS = 5  # Start with a small number for testing
    
    print("🎬 Simple Anomaly Video Evaluation for Qwen2.5-VL")
    print("=" * 60)
    
    # Check for existing incremental results
    latest_incremental = find_latest_incremental_results()
    existing_results = []
    
    if latest_incremental:
        print(f"📂 Found existing incremental results: {latest_incremental}")
        display_progress_summary(latest_incremental)
        print("The script will automatically resume from where it left off.")
        print("To start fresh, delete the incremental results directory.")
        
        # Load existing results
        try:
            with open(latest_incremental / "incremental_results.json", 'r') as f:
                existing_results = json.load(f)
            print(f"📂 Found {len(existing_results)} existing results, resuming...")
        except Exception as e:
            print(f"⚠️  Could not load existing results: {e}")
            existing_results = []
    
    # Load model
    model, processor = load_model()
    
    # Load data
    print(f"📊 Loading data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    if MAX_VIDEOS:
        df = df.head(MAX_VIDEOS)
    
    print(f"📹 Found {len(df)} videos to evaluate")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"simple_evaluation_results/incremental_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy existing results to new directory if resuming
    if existing_results:
        with open(output_dir / "incremental_results.json", 'w') as f:
            json.dump(existing_results, f, indent=2)
        print(f"📋 Copied existing results to new directory: {output_dir}")
    
    all_results = existing_results.copy()
    processed_videos = {r['video_name'] for r in all_results}
    video_dir_path = Path(VIDEO_DIR)
    
    # Evaluate videos
    for idx, row in df.iterrows():
        video_name = row['anomaly_video_name']
        video_path = video_dir_path / video_name
        
        if video_name in processed_videos:
            print(f"⏭️  Skipping {video_name} (already processed)")
            continue
        
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}")
            continue
        
        try:
            result = evaluate_video(model, processor, video_path, row.to_dict())
            all_results.append(result)
            
            # Save incremental results
            save_incremental_results(all_results, output_dir, idx + 1, len(df))
            
            print(f"✅ Completed {idx + 1}/{len(df)}: {video_name}")
            print(f"   Scores: Content={result['scores']['content_accuracy']:.1f}, Anomaly={result['scores']['anomaly_detection']:.1f}, Overall={result['scores']['overall']:.1f}")
            
            # Add delay to avoid rate limiting
            time.sleep(2)
            
        except Exception as e:
            print(f"❌ Error evaluating {video_name}: {e}")
            save_incremental_results(all_results, output_dir, idx + 1, len(df))
            continue
    
    # Save final results
    final_output_file = output_dir / f"simple_evaluation_results_{timestamp}.json"
    with open(final_output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    if all_results:
        avg_content = np.mean([r['scores']['content_accuracy'] for r in all_results])
        avg_anomaly = np.mean([r['scores']['anomaly_detection'] for r in all_results])
        avg_overall = np.mean([r['scores']['overall'] for r in all_results])
        
        print(f"Total videos evaluated: {len(all_results)}")
        print(f"Average Content Accuracy: {avg_content:.2f}/10")
        print(f"Average Anomaly Detection: {avg_anomaly:.2f}/10")
        print(f"Average Overall Score: {avg_overall:.2f}/10")
        
        # Show some examples
        print(f"\n📋 Sample Results:")
        for i, result in enumerate(all_results[:3]):
            print(f"\nVideo {i+1}: {result['video_name']}")
            print(f"  Expected anomalies: {result['expected_anomalies']}")
            print(f"  Anomaly detection response: {result['anomaly_detection'][:150]}...")
            print(f"  Scores: {result['scores']}")
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"📄 Final results: {final_output_file}")
    print("✅ Evaluation complete!")

if __name__ == "__main__":
    main()


