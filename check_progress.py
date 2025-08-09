#!/usr/bin/env python3
"""
Check Progress Script for Enhanced Anomaly Video Evaluation
==========================================================

This script allows you to check the current progress of the evaluation
without running the full evaluation process.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

def find_latest_incremental_results():
    """Find the most recent incremental results directory."""
    eval_dir = Path("evaluation_results")
    if not eval_dir.exists():
        return None
    
    incremental_dirs = [d for d in eval_dir.iterdir() if d.is_dir() and d.name.startswith("incremental_")]
    if not incremental_dirs:
        return None
    
    # Return the most recent directory
    return max(incremental_dirs, key=lambda x: x.stat().st_mtime)

def display_progress_summary(incremental_dir):
    """Display current progress from incremental results."""
    print(f"📂 Incremental Results Directory: {incremental_dir}")
    print("=" * 60)
    
    # Check progress summary
    progress_file = incremental_dir / "progress_summary.json"
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
            
            print(f"📊 Progress Summary:")
            print(f"  Progress: {progress['progress']}")
            print(f"  Completed videos: {progress['completed_videos']}")
            print(f"  Latest video: {progress['latest_video']}")
            print(f"  Average overall score: {progress['average_overall_score']:.2f}")
            print(f"  Last updated: {progress['timestamp']}")
            
        except Exception as e:
            print(f"⚠️  Could not read progress summary: {e}")
    
    # Check current summary CSV
    summary_file = incremental_dir / "current_summary.csv"
    if summary_file.exists():
        try:
            df = pd.read_csv(summary_file)
            print(f"\n📈 Score Statistics:")
            print(f"  Total videos: {len(df)}")
            print(f"  Average Content Accuracy: {df['content_accuracy'].mean():.2f}")
            print(f"  Average Anomaly Detection: {df['anomaly_detection'].mean():.2f}")
            print(f"  Average Confidence Calibration: {df['confidence_calibration'].mean():.2f}")
            print(f"  Average Hallucination Score: {df['hallucination'].mean():.2f}")
            print(f"  Average Overall Score: {df['overall_score'].mean():.2f}")
            
            print(f"\n🏆 Best Performing Videos:")
            best_videos = df.nlargest(3, 'overall_score')[['video_name', 'overall_score']]
            for _, row in best_videos.iterrows():
                print(f"  {row['video_name']}: {row['overall_score']:.2f}")
            
            print(f"\n📉 Worst Performing Videos:")
            worst_videos = df.nsmallest(3, 'overall_score')[['video_name', 'overall_score']]
            for _, row in worst_videos.iterrows():
                print(f"  {row['video_name']}: {row['overall_score']:.2f}")
                
        except Exception as e:
            print(f"⚠️  Could not read summary CSV: {e}")
    
    # Check video responses directory
    responses_dir = incremental_dir / "video_responses"
    if responses_dir.exists():
        response_files = list(responses_dir.glob("*_responses.json"))
        print(f"\n📝 Individual Video Responses:")
        print(f"  Total response files: {len(response_files)}")
        
        # Show latest 3 responses
        latest_responses = sorted(response_files, key=lambda x: x.stat().st_mtime, reverse=True)[:3]
        for resp_file in latest_responses:
            try:
                with open(resp_file, 'r') as f:
                    data = json.load(f)
                print(f"  {data['video_name']}: {data['scores']['overall']:.2f} (last updated: {data['timestamp']})")
            except Exception as e:
                print(f"  {resp_file.name}: Error reading file")

def main():
    """Main function to check progress."""
    print("🔍 Checking Evaluation Progress")
    print("=" * 60)
    
    latest_incremental = find_latest_incremental_results()
    if latest_incremental:
        display_progress_summary(latest_incremental)
    else:
        print("❌ No incremental results found.")
        print("Run the evaluation script first to generate results.")

if __name__ == "__main__":
    main()

