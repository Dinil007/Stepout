"""PHASE 6: Frame escalation - FIXED."""
import json
import time
import subprocess
from pathlib import Path

def run_pipeline(frames, output):
    cmd = f"python scripts/run_match_analysis.py --max-frames {frames} --output {output}"
    start = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    
    # Check if pipeline produced outputs regardless of return code
    outputs_exist = False
    try:
        out_path = Path(output)
        if out_path.exists():
            outputs_exist = True
    except:
        pass
    
    return {
        "success": outputs_exist and result.returncode == 0,
        "returncode": result.returncode,
        "outputs_exist": outputs_exist,
        "runtime_seconds": elapsed,
        "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        "stderr": result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
    }

def main():
    print("\n=== PHASE 6: FRAME ESCALATION ===")
    frame_counts = [100, 500, 1000, 2000, 10000]
    results = {}
    
    for n in frame_counts:
        print(f"\nTesting {n} frames...")
        out = f"outputs/escalation_{n}.json"
        res = run_pipeline(n, out)
        results[str(n)] = res
        
        status = "PASS" if res["success"] else "FAIL"
        print(f"  Status: {status}")
        print(f"  Return code: {res['returncode']}")
        print(f"  Outputs exist: {res['outputs_exist']}")
        print(f"  Runtime: {res['runtime_seconds']:.2f}s")
        
        if not res["success"]:
            print(f"  ERROR: Pipeline failed at {n} frames")
            if res["stderr"]:
                print(f"  STDERR: {res['stderr'][-200:]}")
            break
    
    out = Path("outputs/frame_escalation.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()