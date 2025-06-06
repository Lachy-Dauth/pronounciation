"""
Script to train both G2P and P2G models sequentially
"""
import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"STARTING: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✓ COMPLETED: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ FAILED: {description}")
        print(f"Error: {e}")
        return False

def main():
    """Train both models"""
    print("Training both G2P and P2G models...")
    
    # Check if data files exist
    if not os.path.exists('cmudict.dict'):
        print("Error: cmudict.dict not found!")
        print("Please download the CMU Pronunciation Dictionary")
        return
    
    # Train G2P model
    if not run_command("python3 g2p_transformer.py", "G2P Model Training"):
        print("G2P training failed. Stopping.")
        return
    
    # Train P2G model  
    if not run_command("python3 p2g_transformer.py", "P2G Model Training"):
        print("P2G training failed.")
        return
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("Both models have been trained successfully.")
    print("You can now test them using:")
    print("  python3 test_models.py")
    print("="*60)

if __name__ == "__main__":
    main()