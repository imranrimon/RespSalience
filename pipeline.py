import argparse
import sys
import subprocess
import os

def main():
    parser = argparse.ArgumentParser(description="Infant Respiratory Monitoring Suite")
    parser.add_argument("--mode", choices=["dashboard", "cli", "vitals"], default="dashboard", help="Execution mode")
    parser.add_argument("--video", help="Path to video file (required for cli mode)")
    parser.add_argument("--age", type=int, default=6, help="Infant age in months (default: 6)")
    parser.add_argument("--vlm-model", default="google/paligemma-3b-pt-224", help="VLM model ID for diagnostics")
    
    args = parser.parse_args()
    
    if args.mode == "dashboard":
        print("Launching Telemedicine Dashboard...")
        subprocess.run(["streamlit", "run", "src/ui/dashboard.py"])
    
    elif args.mode == "vitals":
        print("Launching Vitals Simulation...")
        subprocess.run(["python", "src/api/vitals_simulator.py"])
        
    elif args.mode == "cli":
        if not args.video:
            print("Error: --video path is required for cli mode.")
            return
        
        print(f"Running CLI Analysis for {args.video}...")
        from src.core.main_controller import RespiratoryController
        controller = RespiratoryController(
            "src/config/test_infer.yaml",
            "data/models/VIRENet_Coarse2fine_Body_best.pth"
        )
        results = controller.process_video(args.video, args.age, vlm_model_id=args.vlm_model)
        print("\n--- Diagnostic Result ---")
        print(f"BPM: {results['bpm']}")
        print(f"VLM Analysis ({args.vlm_model}):")
        print(results['vlm_assessment'])
        print(f"Status: {results['diagnosis']['status']}")
        print(f"Recommendation: {results['recommendation']}")

if __name__ == "__main__":
    main()
