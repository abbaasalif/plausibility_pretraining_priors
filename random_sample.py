import pandas as pd
import zipfile
import os
import random
import shutil
from pathlib import Path

def load_class_mappings():
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

def decode_actions(actions_str, class_mappings):
    """Decode action string into readable format with timestamps"""
    if pd.isna(actions_str) or not actions_str:
        return ""
    
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
                
                description = class_mappings.get(class_id, f"Unknown action ({class_id})")
                
                decoded_actions.append(f"{class_id} {start_time:.2f} {end_time:.2f} ({description})")
    
    return ";".join(decoded_actions)

def random_sample_videos():
    """
    Randomly sample 100 videos from the Charades dataset, extract them from the zip file,
    and create a new CSV with the sampled data.
    """
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Load class mappings
    class_mappings = load_class_mappings()
    
    # Read the training CSV file
    print("Reading training CSV file...")
    train_df = pd.read_csv('Charades/Charades_v1_train.csv')
    print(f"Total videos in training set: {len(train_df)}")
    
    # Randomly sample 100 videos
    print("Randomly sampling 100 videos...")
    sampled_df = train_df.sample(n=100, random_state=42)
    sampled_video_ids = sampled_df['id'].tolist()
    
    # Decode actions for sampled videos
    print("Decoding actions...")
    sampled_df['decoded_actions'] = sampled_df['actions'].apply(
        lambda x: decode_actions(x, class_mappings)
    )
    
    # Create output directory
    output_dir = Path('sampled_videos')
    output_dir.mkdir(exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Extract sampled videos from zip file
    print("Extracting sampled videos from zip file...")
    zip_path = 'Charades_v1_480.zip'
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of all files in zip
        all_files = zip_ref.namelist()
        
        # Filter for our sampled video IDs
        extracted_count = 0
        for video_id in sampled_video_ids:
            # Look for the video file in the zip
            video_filename = f"Charades_v1_480/{video_id}.mp4"
            
            if video_filename in all_files:
                # Extract the video
                zip_ref.extract(video_filename, output_dir)
                # Move from subdirectory to main output directory
                src_path = output_dir / video_filename
                dst_path = output_dir / f"{video_id}.mp4"
                shutil.move(str(src_path), str(dst_path))
                extracted_count += 1
                print(f"Extracted: {video_id}.mp4")
            else:
                print(f"Warning: Video {video_id}.mp4 not found in zip file")
    
    # Remove the empty Charades_v1_480 directory
    empty_dir = output_dir / "Charades_v1_480"
    if empty_dir.exists():
        empty_dir.rmdir()
    
    print(f"Successfully extracted {extracted_count} videos")
    
    # Save the sampled CSV
    print("Saving sampled CSV file...")
    sampled_df.to_csv('sampled_train.csv', index=False)
    print("Saved: sampled_train.csv")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total videos in original training set: {len(train_df)}")
    print(f"Videos sampled: {len(sampled_df)}")
    print(f"Videos successfully extracted: {extracted_count}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Sampled CSV: sampled_train.csv")
    
    # Show first few rows of sampled data
    print("\n=== First 5 rows of sampled data ===")
    print(sampled_df[['id', 'subject', 'scene', 'length']].head())
    
    # Show example of decoded actions
    print("\n=== Example of decoded actions ===")
    example_row = sampled_df.iloc[0]
    print(f"Video ID: {example_row['id']}")
    print(f"Original actions: {example_row['actions']}")
    print(f"Decoded actions: {example_row['decoded_actions']}")

if __name__ == "__main__":
    random_sample_videos() 