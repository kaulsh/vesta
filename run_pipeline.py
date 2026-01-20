#!/usr/bin/env python3
"""
Vesta Pipeline Runner
=====================
Master script to run the complete Vesta training pipeline.

Usage:
    python run_pipeline.py [--skip-preprocess] [--skip-train] [--skip-eval]
"""

import sys
import os
import argparse
import subprocess
from datetime import datetime


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70 + "\n")


def run_command(command, description):
    """
    Run a shell command and handle errors.
    
    Args:
        command (str): Command to run
        description (str): Description of what the command does
        
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"➤ {description}...")
    print(f"  Command: {command}\n")
    
    result = subprocess.run(
        command,
        shell=True,
        cwd=os.path.join(os.path.dirname(__file__), 'scripts'),
        capture_output=False
    )
    
    if result.returncode == 0:
        print(f"✓ {description} completed successfully\n")
        return True
    else:
        print(f"✗ {description} failed with exit code {result.returncode}\n")
        return False


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description='Run the Vesta training pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python run_pipeline.py
  
  # Skip preprocessing (if data already processed)
  python run_pipeline.py --skip-preprocess
  
  # Only run evaluation (if models already trained)
  python run_pipeline.py --skip-preprocess --skip-train
        """
    )
    
    parser.add_argument('--skip-preprocess', action='store_true',
                        help='Skip data preprocessing step')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip model training step')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation step')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: Reduce Optuna trials and GridSearch params')
    
    args = parser.parse_args()
    
    # Print banner
    print_section("VESTA TRAINING PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    if args.fast:
        print("⚡ Fast mode enabled (reduced hyperparameter search)")
    
    print()
    
    # Step 1: Preprocessing
    if not args.skip_preprocess:
        print_section("STEP 1: DATA PREPROCESSING")
        if not run_command("python preprocess.py", "Data preprocessing"):
            print("Pipeline failed at preprocessing step.")
            sys.exit(1)
    else:
        print_section("STEP 1: DATA PREPROCESSING [SKIPPED]")
    
    # Step 2: Training
    if not args.skip_train:
        print_section("STEP 2: MODEL TRAINING")
        
        train_cmd = "python train.py"
        if args.fast:
            # Modify training script behavior for fast mode
            # This would require changes to train.py to accept CLI args
            print("⚡ Using fast training mode")
            train_cmd += " --fast"  # Note: train.py would need to support this
        
        if not run_command(train_cmd, "Model training (RF + LSTM)"):
            print("Pipeline failed at training step.")
            sys.exit(1)
    else:
        print_section("STEP 2: MODEL TRAINING [SKIPPED]")
    
    # Step 3: Evaluation
    if not args.skip_eval:
        print_section("STEP 3: MODEL EVALUATION")
        if not run_command("python evaluate.py", "Model evaluation and visualization"):
            print("Pipeline failed at evaluation step.")
            sys.exit(1)
    else:
        print_section("STEP 3: MODEL EVALUATION [SKIPPED]")
    
    # Summary
    print_section("PIPELINE COMPLETE!")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Results:")
    print("  - Processed data: data/processed/")
    print("  - Trained models: models/")
    print("  - Evaluation results: results/")
    print()
    print("Next steps:")
    print("  1. Check results/metrics.json for model performance")
    print("  2. View visualizations in results/*.png")
    print("  3. Explore data in notebooks/exploration.ipynb")
    print()


if __name__ == '__main__':
    main()
