"""
Enhanced Anomaly Video Evaluation for Gemini 2.5
================================================

This script evaluates Gemini 2.5's ability to detect and describe anomalies in videos.
It uses ChatGPT (OpenAI) as the judge—no manual/keyword scoring remains.

Requirements:
- pip install openai google-generativeai python-dotenv pandas numpy matplotlib seaborn
- .env file with OPENAI_API_KEY=... and GOOGLE_API_KEY=... (OPTIONAL: OPENAI_JUDGE_MODEL, default 'gpt-4o-mini')

DEBUGGING IMPROVEMENTS:
- Added comprehensive debug printing for responses and JSON parsing
- Enhanced error handling with fallback JSON structures
- Added test mode with --test flag
- Improved JSON parsing with cleanup for common issues
- Added full response printing for debugging
- Better error messages and traceback printing
"""

import os
import json
import time
import re
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # optional, used only for styling

from dotenv import load_dotenv

# ----------------------------
# Gemini client
# ----------------------------
try:
    import google.generativeai as genai
except ImportError as e:
    raise ImportError(
        "The 'google-generativeai' package is required. Install with: pip install google-generativeai"
    ) from e

# ----------------------------
# OpenAI client (ChatGPT judge)
# ----------------------------
try:
    from openai import OpenAI
except ImportError as e:
    raise ImportError(
        "The 'openai' package is required. Install with: pip install openai"
    ) from e

# Load environment variables (expects OPENAI_API_KEY and GOOGLE_API_KEY in .env)
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ----------------------------
# LLM Judge using ChatGPT API
# ----------------------------
class LLMJudge:
    """Uses OpenAI ChatGPT as the sole judge (no manual fallback)."""

    def __init__(self, model_name: str = None, request_timeout: int = 60, max_retries: int = 3, retry_backoff: float = 1.5):
        self.model_name = model_name or os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
        self.client = OpenAI()
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Prompt templates (updated to force JSON format)
        self.judge_prompts = {
            "anomaly_detection": """Evaluate if the model correctly identified video anomalies.

Ground Truth: Script: {original_script} | Actions: {decoded_actions} | Scene: {original_scene} | Anomalies: {applied_anomalies}

Model Response: {model_response}

Rate 0-10: anomaly_detection (detected actual anomalies), accuracy (description quality), false_positive (avoided fake claims).

You MUST return ONLY valid JSON in this exact format:
{{
    "anomaly_detection_score": <number 0-10>,
    "accuracy_score": <number 0-10>,
    "false_positive_score": <number 0-10>,
    "overall_assessment": "<brief text>",
    "detected_anomalies": ["item1", "item2"],
    "missed_anomalies": ["item1", "item2"],
    "false_positives": ["item1", "item2"]
}}""",

            "content_accuracy": """Evaluate video content description accuracy.

Ground Truth: Script: {original_script} | Actions: {decoded_actions} | Scene: {original_scene} | Anomalies: {applied_anomalies}

Model Response: {model_response}

Rate 0-10: content_coverage (described actual content), accuracy (description quality), detail_level (specificity).

You MUST return ONLY valid JSON in this exact format:
{{
    "content_coverage_score": <number 0-10>,
    "accuracy_score": <number 0-10>,
    "detail_level_score": <number 0-10>,
    "overall_assessment": "<brief text>",
    "correctly_described_elements": ["item1", "item2"],
    "missing_elements": ["item1", "item2"],
    "incorrect_descriptions": ["item1", "item2"]
}}""",

            "confidence_calibration": """Evaluate confidence calibration in video analysis.

Ground Truth: Script: {original_script} | Actions: {decoded_actions} | Scene: {original_scene} | Anomalies: {applied_anomalies}

Model Response: {model_response}

Rate 0-10: uncertainty_recognition (recognizes uncertainty), confidence_appropriateness (appropriate confidence), calibration (overall quality).

You MUST return ONLY valid JSON in this exact format:
{{
    "uncertainty_recognition_score": <number 0-10>,
    "confidence_appropriateness_score": <number 0-10>,
    "calibration_score": <number 0-10>,
    "overall_assessment": "<brief text>",
    "uncertainty_indicators": ["item1", "item2"],
    "confidence_indicators": ["item1", "item2"],
    "calibration_issues": ["item1", "item2"]
}}""",

            "hallucination_detection": """Evaluate hallucination in video analysis.

Ground Truth: Script: {original_script} | Actions: {decoded_actions} | Scene: {original_scene} | Anomalies: {applied_anomalies}

Model Response: {model_response}

Rate 0-10: hallucination_severity (made-up content), factual_accuracy (truthfulness), speculation_control (uncertainty marking).

You MUST return ONLY valid JSON in this exact format:
{{
    "hallucination_severity_score": <number 0-10>,
    "factual_accuracy_score": <number 0-10>,
    "speculation_control_score": <number 0-10>,
    "overall_assessment": "<brief text>",
    "hallucinated_elements": ["item1", "item2"],
    "factual_errors": ["item1", "item2"],
    "speculative_content": ["item1", "item2"]
}}"""
        }

    def _post(self, prompt: str) -> str:
        """Call OpenAI with retries; return text content."""
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": "You are a strict evaluation judge. You MUST return ONLY valid JSON. No other text, no explanations, just JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    timeout=self.request_timeout,
                )
                content = resp.choices[0].message.content or ""
                return content.strip()
            except Exception as e:
                last_err = e
                error_str = str(e).lower()
                
                # Handle rate limits specifically
                if "rate limit" in error_str or "429" in error_str:
                    # Exponential backoff for rate limits
                    sleep_s = min(60, self.retry_backoff ** (attempt - 1) * 10)  # Cap at 60 seconds
                    print(f"⚠️ Rate limit hit (attempt {attempt}/{self.max_retries}). Waiting {sleep_s:.1f}s...")
                elif "quota" in error_str:
                    print(f"❌ Quota exceeded. Please check your OpenAI API quota.")
                    raise RuntimeError("OpenAI API quota exceeded. Please upgrade your plan or wait for quota reset.")
                else:
                    sleep_s = self.retry_backoff ** (attempt - 1)
                    print(f"⚠️ OpenAI judge call failed (attempt {attempt}/{self.max_retries}): {e}. Retrying in {sleep_s:.1f}s")
                
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI judge call failed after {self.max_retries} attempts: {last_err}")

    def generate_judgment(self, prompt_type: str, model_response: str, ground_truth: Dict) -> Dict:
        """Get judgment JSON from ChatGPT; raises on failure."""
        if prompt_type not in self.judge_prompts:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        prompt = self.judge_prompts[prompt_type].format(
            original_script=ground_truth.get('original_script', 'N/A'),
            decoded_actions=ground_truth.get('decoded_actions', 'N/A'),
            original_scene=ground_truth.get('original_scene', 'N/A'),
            applied_anomalies=ground_truth.get('anomalies_applied', 'N/A'),
            model_response=model_response,
        )
        
        print(f"🤖 Sending prompt to ChatGPT judge ({prompt_type})...")
        text = self._post(prompt)
        print(f"📄 Raw judge response: {text[:500]}...")

        # Extract/parse JSON (strict request, but be defensive)
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start == -1 or json_end <= json_start:
            print(f"❌ Judge did not return JSON: {text[:200]}")
            # Return a default structure
            return {
                "error": "No JSON found in response",
                "raw_response": text[:200],
                "overall_assessment": "Failed to parse judge response"
            }

        payload = text[json_start:json_end]
        try:
            result = json.loads(payload)
            print(f"✅ Successfully parsed JSON: {list(result.keys())}")
            return result
        except json.JSONDecodeError as e:
            print(f"❌ Judge JSON parsing error: {e}; text: {text[:200]}")
            # Try to fix common JSON issues
            try:
                # Remove any trailing commas or extra characters
                payload = re.sub(r',\s*}', '}', payload)
                payload = re.sub(r',\s*]', ']', payload)
                result = json.loads(payload)
                print(f"✅ Successfully parsed JSON after cleanup: {list(result.keys())}")
                return result
            except json.JSONDecodeError as e2:
                print(f"❌ Still failed after cleanup: {e2}")
                # Return a default structure
                return {
                    "error": f"JSON parsing error: {e}",
                    "raw_response": text[:200],
                    "overall_assessment": "Failed to parse judge response"
                }

# ----------------------------
# Evaluator
# ----------------------------
class AnomalyVideoEvaluator:
    def __init__(self, model_name: str = "gemini-2.0-flash-exp", verbose: bool = False):
        self.verbose = verbose

        print(f"🔄 Loading Gemini model: {model_name}")
        try:
            self.model = genai.GenerativeModel(model_name)
            print("✅ Gemini model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Gemini model: {e}")
            raise

        print("🔄 Initializing ChatGPT LLM judge...")
        self.judge = LLMJudge()
        print("✅ LLM judge initialized successfully")
        
        # Add delay between video evaluations to avoid rate limits
        self.inter_video_delay = 2.0  # seconds between videos

        self.questions = {
            "basic_description": "Describe what you see in this video in detail. What activities are happening?",
            "anomaly_detection": "Does anything seem unusual or out of place in this video? Are there any visual distortions or anomalies you notice?",
            "temporal_analysis": "Does the video appear to be playing normally or are there any temporal issues like reversed playback or shuffled frames?",
            "spatial_analysis": "Do you see any objects that don't belong in this scene? Are there any color distortions or visual artifacts?",
            "semantic_analysis": "Does the sequence of events make logical sense? Are there any scene transitions that seem abrupt or unnatural?",
            "confidence_assessment": "How confident are you in your description? Are there any parts you're uncertain about?"
        }

    # ----------------------------
    # Core inference for Gemini
    # ----------------------------
    def generate_response(self, video_path: str, question: str) -> str:
        try:
            print(f"    🎥 Processing video: {Path(video_path).name}")
            print(f"    ❓ Question: {question[:100]}...")
            
            # Load video file
            with open(video_path, 'rb') as video_file:
                video_data = video_file.read()
            
            print(f"    🔄 Generating response with Gemini...")
            
            # Create the prompt with video
            prompt = f"{question}\n\nPlease analyze this video and provide a detailed response."
            
            # Generate response using Gemini with retries
            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.model.generate_content([prompt, {"mime_type": "video/mp4", "data": video_data}])
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate limit" in error_str or "quota" in error_str or "429" in error_str:
                        if attempt < max_retries:
                            sleep_s = min(30, 5 * attempt)  # Progressive backoff: 5s, 10s, 15s
                            print(f"    ⚠️ Gemini rate limit hit (attempt {attempt}/{max_retries}). Waiting {sleep_s}s...")
                            time.sleep(sleep_s)
                            continue
                        else:
                            print(f"    ❌ Gemini rate limit exceeded after {max_retries} attempts")
                            return f"Error: Rate limit exceeded - {str(e)}"
                    else:
                        # Non-rate-limit error, don't retry
                        print(f"    ❌ Gemini error: {e}")
                        return f"Error: {str(e)}"
            
            # Extract text from response
            if hasattr(response, 'text'):
                output_text = response.text
            else:
                output_text = str(response)
            
            response_text = output_text.strip()
            print(f"    📝 Response: {response_text[:200]}...")
            return response_text

        except Exception as e:
            error_msg = f"Error generating response: {e}"
            print(f"    ❌ {error_msg}")
            return f"Error: {str(e)}"

    # ----------------------------
    # Scoring via ChatGPT only
    # ----------------------------
    def _score_content_accuracy(self, response: str, ground_truth: Dict) -> Tuple[float, Dict]:
        print(f"🔍 Scoring content accuracy...")
        j = self.judge.generate_judgment("content_accuracy", response, ground_truth)
        
        # Handle error cases
        if "error" in j:
            print(f"⚠️ Content accuracy scoring failed: {j['error']}")
            return 0.0, j
        
        scores = []
        weights = []
        if "content_coverage_score" in j:
            scores.append(j["content_coverage_score"]); weights.append(0.4)
        if "accuracy_score" in j:
            scores.append(j["accuracy_score"]); weights.append(0.4)
        if "detail_level_score" in j:
            scores.append(j["detail_level_score"]); weights.append(0.2)
        final = float(np.average(scores, weights=weights)) if scores else 0.0
        print(f"📊 Content accuracy score: {final:.2f}")
        return final, j

    def _score_anomaly_detection(self, response: str, ground_truth: Dict) -> Tuple[float, Dict]:
        print(f"🔍 Scoring anomaly detection...")
        j = self.judge.generate_judgment("anomaly_detection", response, ground_truth)
        
        # Handle error cases
        if "error" in j:
            print(f"⚠️ Anomaly detection scoring failed: {j['error']}")
            return 0.0, j
        
        scores = []
        weights = []
        if "anomaly_detection_score" in j:
            scores.append(j["anomaly_detection_score"]); weights.append(0.5)
        if "accuracy_score" in j:
            scores.append(j["accuracy_score"]); weights.append(0.3)
        if "false_positive_score" in j:
            scores.append(j["false_positive_score"]); weights.append(0.2)
        final = float(np.average(scores, weights=weights)) if scores else 0.0
        print(f"📊 Anomaly detection score: {final:.2f}")
        return final, j

    def _score_confidence_calibration(self, response: str, ground_truth: Dict) -> Tuple[float, Dict]:
        print(f"🔍 Scoring confidence calibration...")
        j = self.judge.generate_judgment("confidence_calibration", response, ground_truth)
        
        # Handle error cases
        if "error" in j:
            print(f"⚠️ Confidence calibration scoring failed: {j['error']}")
            return 0.0, j
        
        scores = []
        weights = []
        if "uncertainty_recognition_score" in j:
            scores.append(j["uncertainty_recognition_score"]); weights.append(0.4)
        if "confidence_appropriateness_score" in j:
            scores.append(j["confidence_appropriateness_score"]); weights.append(0.4)
        if "calibration_score" in j:
            scores.append(j["calibration_score"]); weights.append(0.2)
        final = float(np.average(scores, weights=weights)) if scores else 0.0
        print(f"📊 Confidence calibration score: {final:.2f}")
        return final, j

    def _score_hallucination(self, response: str, ground_truth: Dict) -> Tuple[float, Dict]:
        print(f"🔍 Scoring hallucination detection...")
        j = self.judge.generate_judgment("hallucination_detection", response, ground_truth)
        
        # Handle error cases
        if "error" in j:
            print(f"⚠️ Hallucination detection scoring failed: {j['error']}")
            return 0.0, j
        
        # invert hallucination_severity: lower severity -> higher score
        scores = []
        weights = []
        if "hallucination_severity_score" in j:
            scores.append(10.0 - j["hallucination_severity_score"]); weights.append(0.5)
        if "factual_accuracy_score" in j:
            scores.append(j["factual_accuracy_score"]); weights.append(0.3)
        if "speculation_control_score" in j:
            scores.append(j["speculation_control_score"]); weights.append(0.2)
        final = float(np.average(scores, weights=weights)) if scores else 0.0
        print(f"📊 Hallucination detection score: {final:.2f}")
        return final, j

    # ----------------------------
    # Evaluation Loop
    # ----------------------------
    def evaluate_video(self, video_path: str, ground_truth: Dict) -> Dict:
        results = {
            'video_name': ground_truth['anomaly_video_name'],
            'ground_truth': ground_truth,
            'responses': {},
            'scores': {},
            'judgments': {}
        }

        print(f"🔍 Evaluating: {ground_truth['anomaly_video_name']}")
        print(f"   📊 Ground truth: Script='{ground_truth.get('original_script', 'N/A')[:100]}...'")
        print(f"   🎯 Anomalies: {ground_truth.get('anomalies_applied', 'N/A')}")

        for qtype, question in self.questions.items():
            print(f"  📝 {qtype}...")
            response = self.generate_response(video_path, question)
            results['responses'][qtype] = response
            
            # Print full response for debugging
            print(f"    📄 Full {qtype} response:")
            print(f"    {'='*50}")
            print(f"    {response}")
            print(f"    {'='*50}")
            
            time.sleep(1.0)  # increased pacing to avoid rate limits

        # Score using ChatGPT judge
        basic_response = results['responses']['basic_description']
        anomaly_response = results['responses']['anomaly_detection']

        print("  🧠 Getting ChatGPT judge assessments...")
        try:
            content_score, content_j = self._score_content_accuracy(basic_response, ground_truth)
            anomaly_score, anomaly_j = self._score_anomaly_detection(anomaly_response, ground_truth)
            conf_score, conf_j = self._score_confidence_calibration(basic_response, ground_truth)
            halluc_score, halluc_j = self._score_hallucination(basic_response, ground_truth)
        except Exception as e:
            print(f"  ❌ Error during scoring: {e}")
            import traceback
            traceback.print_exc()
            # Set default scores
            content_score, content_j = 0.0, {"error": str(e)}
            anomaly_score, anomaly_j = 0.0, {"error": str(e)}
            conf_score, conf_j = 0.0, {"error": str(e)}
            halluc_score, halluc_j = 0.0, {"error": str(e)}

        results['judgments'] = {
            'content_accuracy': content_j,
            'anomaly_detection': anomaly_j,
            'confidence_calibration': conf_j,
            'hallucination_detection': halluc_j
        }

        results['scores'] = {
            'content_accuracy': content_score,
            'anomaly_detection': anomaly_score,
            'confidence_calibration': conf_score,
            'hallucination': halluc_score
        }
        results['scores']['overall'] = float(np.mean(list(results['scores'].values())))

        print(f"  📊 Final scores: Content={content_score:.2f}, Anomaly={anomaly_score:.2f}, Confidence={conf_score:.2f}, Hallucination={halluc_score:.2f}, Overall={results['scores']['overall']:.2f}")

        if self.verbose:
            self.display_judge_assessment(ground_truth['anomaly_video_name'], results['judgments'])

        return results

    # ----------------------------
    # Reporting / Saving
    # ----------------------------
    def run_evaluation(self, csv_path: str, video_dir: str, max_videos: int = None) -> Dict:
        print(f"🚀 Starting comprehensive anomaly video evaluation with Gemini 2.5")
        print(f"📊 Loading data from: {csv_path}")

        df = pd.read_csv(csv_path)
        if max_videos:
            df = df.head(max_videos)

        print(f"📹 Found {len(df)} videos to evaluate")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        incremental_dir = Path(f"evaluation_results/incremental_{timestamp}")
        incremental_dir.mkdir(parents=True, exist_ok=True)

        existing_results = []
        existing_file = incremental_dir / "incremental_results.json"
        if existing_file.exists():
            try:
                with open(existing_file, 'r') as f:
                    existing_results = json.load(f)
                print(f"📂 Found {len(existing_results)} existing results, resuming...")
            except Exception as e:
                print(f"⚠️  Could not load existing results: {e}")

        all_results = existing_results.copy()
        processed_videos = {r['video_name'] for r in all_results}
        video_dir_path = Path(video_dir)

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
                result = self.evaluate_video(video_path, row.to_dict())
                all_results.append(result)
                self.save_incremental_results(all_results, incremental_dir, idx + 1, len(df))
                print(f"✅ Completed {idx + 1}/{len(df)}: {video_name}")
                
                # Add delay between videos to avoid rate limits
                if idx + 1 < len(df):
                    print(f"⏳ Waiting {self.inter_video_delay}s before next video...")
                    time.sleep(self.inter_video_delay)
                    
            except Exception as e:
                print(f"❌ Error evaluating {video_name}: {e}")
                self.save_incremental_results(all_results, incremental_dir, idx + 1, len(df))
                continue

        return self.compile_results(all_results)

    def compile_results(self, all_results: List[Dict]) -> Dict:
        print("📈 Compiling results...")

        scores_df = pd.DataFrame([
            {
                'video_name': r['video_name'],
                'content_accuracy': r['scores']['content_accuracy'],
                'anomaly_detection': r['scores']['anomaly_detection'],
                'confidence_calibration': r['scores']['confidence_calibration'],
                'hallucination': r['scores']['hallucination'],
                'overall': r['scores']['overall']
            }
            for r in all_results
        ])

        numeric_columns = scores_df.select_dtypes(include=[np.number]).columns
        stats = {
            'total_videos': len(all_results),
            'average_scores': scores_df[numeric_columns].mean().to_dict(),
            'score_std': scores_df[numeric_columns].std().to_dict(),
            'score_ranges': {
                col: {'min': float(scores_df[col].min()), 'max': float(scores_df[col].max())}
                for col in numeric_columns
            }
        }

        anomaly_analysis = self.analyze_by_anomaly_type(all_results)
        judge_analysis = self.analyze_judge_assessments(all_results)

        return {
            'summary': stats,
            'detailed_results': all_results,
            'anomaly_analysis': anomaly_analysis,
            'judge_analysis': judge_analysis,
            'timestamp': datetime.now().isoformat()
        }

    def save_incremental_results(self, results: List[Dict], output_dir: Path, current_idx: int, total_videos: int):
        try:
            with open(output_dir / "incremental_results.json", 'w') as f:
                json.dump(results, f, indent=2)

            if results:
                progress_summary = {
                    'timestamp': datetime.now().isoformat(),
                    'progress': f"{current_idx}/{total_videos}",
                    'completed_videos': len(results),
                    'latest_video': results[-1]['video_name'],
                    'latest_scores': results[-1]['scores'],
                    'average_overall_score': float(np.mean([r['scores']['overall'] for r in results])),
                    'latest_judgments': results[-1].get('judgments', {})
                }
                with open(output_dir / "progress_summary.json", 'w') as f:
                    json.dump(progress_summary, f, indent=2)

                summary_data = [{
                    'video_name': r['video_name'],
                    'content_accuracy': r['scores']['content_accuracy'],
                    'anomaly_detection': r['scores']['anomaly_detection'],
                    'confidence_calibration': r['scores']['confidence_calibration'],
                    'hallucination': r['scores']['hallucination'],
                    'overall_score': r['scores']['overall']
                } for r in results]
                pd.DataFrame(summary_data).to_csv(output_dir / "current_summary.csv", index=False)

                latest = results[-1]
                vr_dir = output_dir / "video_responses"
                vr_dir.mkdir(exist_ok=True)
                with open(vr_dir / f"{latest['video_name'].replace('.mp4','')}_responses.json", 'w') as f:
                    json.dump({
                        'video_name': latest['video_name'],
                        'ground_truth': latest['ground_truth'],
                        'responses': latest['responses'],
                        'scores': latest['scores'],
                        'judgments': latest.get('judgments', {}),
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)

            print(f"💾 Saved incremental results ({len(results)} videos completed)")
        except Exception as e:
            print(f"⚠️  Error saving incremental results: {e}")

    def find_latest_incremental_results(self) -> Path:
        eval_dir = Path("evaluation_results")
        if not eval_dir.exists():
            return None
        incremental_dirs = [d for d in eval_dir.iterdir() if d.is_dir() and d.name.startswith("incremental_")]
        return max(incremental_dirs, key=lambda x: x.stat().st_mtime) if incremental_dirs else None

    def display_progress_summary(self, incremental_dir: Path):
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
                print(f"  Last updated: {progress['timestamp']}")
                if 'latest_judgments' in progress:
                    print(f"\n🧠 Latest Judge Assessments:")
                    for metric, judgment in progress['latest_judgments'].items():
                        if isinstance(judgment, dict) and 'overall_assessment' in judgment:
                            print(f"    {metric}: {judgment['overall_assessment'][:100]}...")
            except Exception as e:
                print(f"⚠️  Could not read progress summary: {e}")

    def categorize_anomalies(self, anomalies: List[str]) -> Dict[str, List[str]]:
        cats = {"temporal": [], "spatial": [], "semantic": []}
        for a in anomalies:
            if any(k in a for k in ["frame_reversal", "frame_shuffling", "speed_change", "temporal_cut"]):
                cats["temporal"].append(a)
            elif any(k in a for k in ["object_insertion", "color_shift", "geometric_distortion", "noise_injection"]):
                cats["spatial"].append(a)
            elif any(k in a for k in ["scene_mixing", "video_combination", "combined_scenes", "combined_videos"]):
                cats["semantic"].append(a)
        return cats

    def analyze_by_anomaly_type(self, results: List[Dict]) -> Dict:
        analysis = {
            'temporal': {'count': 0, 'avg_score': 0, 'videos': []},
            'spatial': {'count': 0, 'avg_score': 0, 'videos': []},
            'semantic': {'count': 0, 'avg_score': 0, 'videos': []}
        }
        for r in results:
            gt = r['ground_truth']
            anomalies = gt.get('anomalies_applied', '').split(';')
            categorized = self.categorize_anomalies(anomalies)
            for cat, a_list in categorized.items():
                if a_list:
                    analysis[cat]['count'] += 1
                    analysis[cat]['videos'].append({
                        'video_name': r['video_name'],
                        'anomalies': a_list,
                        'scores': r['scores']
                    })
        for cat in analysis:
            if analysis[cat]['count'] > 0:
                scores = [v['scores']['overall'] for v in analysis[cat]['videos']]
                analysis[cat]['avg_score'] = float(np.mean(scores))
        return analysis

    def analyze_judge_assessments(self, results: List[Dict]) -> Dict:
        analysis = {
            'judgment_quality': {'successful_judgments': 0, 'failed_judgments': 0, 'error_types': {}},
            'assessment_patterns': {
                'content_accuracy': {'avg_score': 0, 'common_issues': []},
                'anomaly_detection': {'avg_score': 0, 'common_issues': []},
                'confidence_calibration': {'avg_score': 0, 'common_issues': []},
                'hallucination_detection': {'avg_score': 0, 'common_issues': []}
            }
        }
        total_j = 0
        success = 0
        for r in results:
            jud = r.get('judgments', {})
            for metric, j in jud.items():
                total_j += 1
                if isinstance(j, dict) and 'overall_assessment' in j:
                    success += 1
                    analysis['assessment_patterns'][metric]['common_issues'].append(j['overall_assessment'])
                else:
                    analysis['judgment_quality']['failed_judgments'] += 1
                    err = j.get('error', 'unknown') if isinstance(j, dict) else 'unknown'
                    analysis['judgment_quality']['error_types'][err] = analysis['judgment_quality']['error_types'].get(err, 0) + 1
        analysis['judgment_quality']['successful_judgments'] = success
        analysis['judgment_quality']['success_rate'] = (success / total_j) if total_j else 0.0
        return analysis

    def display_judge_assessment(self, video_name: str, judgments: Dict):
        print(f"\n🧠 LLM Judge Assessment for {video_name}:")
        print("-" * 50)
        for metric, j in judgments.items():
            print(f"\n📊 {metric.replace('_', ' ').title()}:")
            if isinstance(j, dict):
                for k, v in j.items():
                    if isinstance(v, (int, float)) and 'score' in k:
                        print(f"  {k.replace('_', ' ').title()}: {v:.1f}/10")
                if 'overall_assessment' in j:
                    print(f"  Assessment: {j['overall_assessment']}")
                for key in ['detected_anomalies', 'missed_anomalies', 'false_positives',
                            'correctly_described_elements', 'missing_elements', 'incorrect_descriptions',
                            'uncertainty_indicators', 'confidence_indicators', 'calibration_issues',
                            'hallucinated_elements', 'factual_errors', 'speculative_content']:
                    if key in j and j[key]:
                        val = j[key]
                        if isinstance(val, list):
                            print(f"  {key.replace('_', ' ').title()}: {', '.join(val)}")
            else:
                print("  (non-dict judgment)")

    def save_results(self, results: Dict, output_dir: str = "evaluation_results") -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        detailed_file = output_path / f"detailed_results_{ts}.json"
        with open(detailed_file, 'w') as f:
            json.dump(results, f, indent=2)

        summary_data = [{
            'video_name': r['video_name'],
            'content_accuracy': r['scores']['content_accuracy'],
            'anomaly_detection': r['scores']['anomaly_detection'],
            'confidence_calibration': r['scores']['confidence_calibration'],
            'hallucination': r['scores']['hallucination'],
            'overall_score': r['scores']['overall']
        } for r in results['detailed_results']]
        summary_df = pd.DataFrame(summary_data)
        summary_file = output_path / f"summary_scores_{ts}.csv"
        summary_df.to_csv(summary_file, index=False)

        self.generate_plots(results, output_path, ts)

        print(f"💾 Results saved to: {output_path}")
        print(f"📄 Detailed results: {detailed_file}")
        print(f"📊 Summary scores: {summary_file}")
        return output_path

    def generate_plots(self, results: Dict, output_path: Path, timestamp: str):
        scores_df = pd.DataFrame([
            {
                'video_name': r['video_name'],
                'content_accuracy': r['scores']['content_accuracy'],
                'anomaly_detection': r['scores']['anomaly_detection'],
                'confidence_calibration': r['scores']['confidence_calibration'],
                'hallucination': r['scores']['hallucination'],
                'overall': r['scores']['overall']
            }
            for r in results['detailed_results']
        ])

        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gemini 2.5 Anomaly Video Evaluation Results', fontsize=16, fontweight='bold')

        axes[0, 0].hist(scores_df['overall'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Overall Score Distribution')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].axvline(scores_df['overall'].mean(), color='red', linestyle='--', label=f"Mean: {scores_df['overall'].mean():.2f}")
        axes[0, 0].legend()

        score_cols = ['content_accuracy', 'anomaly_detection', 'confidence_calibration', 'hallucination']
        avg_scores = [scores_df[col].mean() for col in score_cols]
        axes[0, 1].bar(score_cols, avg_scores, color=['lightcoral', 'lightgreen', 'lightblue', 'lightyellow'])
        axes[0, 1].set_title('Average Scores by Category')
        axes[0, 1].set_ylabel('Average Score')
        axes[0, 1].tick_params(axis='x', rotation=45)

        corr = scores_df[score_cols].corr()
        im = axes[1, 0].imshow(corr, cmap='coolwarm', aspect='auto')
        axes[1, 0].set_title('Score Correlation Matrix')
        axes[1, 0].set_xticks(range(len(score_cols)))
        axes[1, 0].set_yticks(range(len(score_cols)))
        axes[1, 0].set_xticklabels(score_cols, rotation=45)
        axes[1, 0].set_yticklabels(score_cols)
        plt.colorbar(im, ax=axes[1, 0])

        categories = list(results['anomaly_analysis'].keys())
        counts = [results['anomaly_analysis'][c]['count'] for c in categories]
        avg_by_cat = [results['anomaly_analysis'][c]['avg_score'] for c in categories]
        x = np.arange(len(categories))
        width = 0.35
        ax2 = axes[1, 1].twinx()
        axes[1, 1].bar(x - width/2, counts, width, label='Video Count', alpha=0.7)
        ax2.bar(x + width/2, avg_by_cat, width, label='Avg Score', alpha=0.7, color='orange')
        axes[1, 1].set_title('Performance by Anomaly Type')
        axes[1, 1].set_xlabel('Anomaly Type')
        axes[1, 1].set_ylabel('Video Count')
        ax2.set_ylabel('Average Score')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(categories)
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plot_file = output_path / f"evaluation_plots_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Plots saved: {plot_file}")


def test_single_video():
    """Test function to debug a single video evaluation."""
    print("🧪 Testing single video evaluation with Gemini 2.5...")
    
    # Initialize evaluator
    evaluator = AnomalyVideoEvaluator(verbose=True)
    
    # Test with a simple ground truth
    test_ground_truth = {
        'anomaly_video_name': 'anomaly_video_0000.mp4',
        'original_script': 'A person walks through the entryway, laughing and eating from a bag of snacks.',
        'decoded_actions': 'walking, eating, laughing',
        'original_scene': 'entryway',
        'anomalies_applied': 'object_insertion;color_shift'
    }
    
    video_path = "anomaly_videos/anomaly_video_0000.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    try:
        result = evaluator.evaluate_video(video_path, test_ground_truth)
        print(f"✅ Test completed successfully!")
        print(f"📊 Final scores: {result['scores']}")
        return result
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Enhanced Anomaly Video Evaluation with Gemini 2.5 (ChatGPT judge only)")
    parser.add_argument("--max-videos", type=int, default=50, help="Maximum number of videos to evaluate")
    parser.add_argument("--csv-path", default="anomaly_videos/anomaly_summary.csv", help="Path to anomaly summary CSV")
    parser.add_argument("--video-dir", default="anomaly_videos", help="Directory containing anomaly videos")
    parser.add_argument("--verbose", action="store_true", help="Print detailed judge assessments")
    parser.add_argument("--test", action="store_true", help="Run in test mode (single video)")
    parser.add_argument("--model", default="gemini-2.0-flash-exp", help="Gemini model to use")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between videos in seconds (default: 2.0)")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum retries for API calls (default: 3)")
    args = parser.parse_args()

    print("🎬 Enhanced Anomaly Video Evaluation for Gemini 2.5 (ChatGPT Judge)")
    print("=" * 60)
    
    if args.test:
        print("🧪 Running in TEST MODE")
        test_single_video()
        return
    
    print(f"📹 Evaluating up to {args.max_videos} videos with {args.model}")
    print(f"⏱️  Delay between videos: {args.delay}s")
    print(f"🔄 Max retries per API call: {args.max_retries}")

    evaluator = AnomalyVideoEvaluator(model_name=args.model, verbose=args.verbose)
    evaluator.inter_video_delay = args.delay
    evaluator.judge.max_retries = args.max_retries

    latest_incremental = evaluator.find_latest_incremental_results()
    if latest_incremental:
        print(f"📂 Found existing incremental results: {latest_incremental}")
        evaluator.display_progress_summary(latest_incremental)
        print("The script will automatically resume from where it left off.")
        print("To start fresh, delete the incremental results directory.")

    results = evaluator.run_evaluation(args.csv_path, args.video_dir, args.max_videos)
    output_path = evaluator.save_results(results)

    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    summary = results['summary']
    print(f"Total videos evaluated: {summary['total_videos']}")
    print(f"\nAverage Scores:")
    for metric, score in summary['average_scores'].items():
        if metric != 'video_name':
            print(f"  {metric.replace('_', ' ').title()}: {score:.2f}/10")

    print(f"\nAnomaly Type Analysis:")
    for category, data in results['anomaly_analysis'].items():
        if data['count'] > 0:
            print(f"  {category.title()}: {data['count']} videos, avg score: {data['avg_score']:.2f}")

    print(f"\n🧠 LLM Judge Analysis:")
    judge_analysis = results.get('judge_analysis', {})
    if judge_analysis:
        quality = judge_analysis.get('judgment_quality', {})
        success_rate = quality.get('success_rate', 0) * 100
        print(f"  Judge Success Rate: {success_rate:.1f}%")
        print(f"  Successful Judgments: {quality.get('successful_judgments', 0)}")
        print(f"  Failed Judgments: {quality.get('failed_judgments', 0)}")

    print(f"\n✅ Evaluation complete! Results saved to: {output_path}")
    print(f"📂 Incremental results saved to: evaluation_results/incremental_*/")
    print(f"🧠 ChatGPT judge provided all scoring (no manual/keyword scoring).")
    print(f"🤖 Gemini 2.5 processed all videos.")


if __name__ == "__main__":
    main()
