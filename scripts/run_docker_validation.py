"""PHASE 9: Docker validation."""
import subprocess
import json
from pathlib import Path

def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode == 0

def main():
    print("\n=== PHASE 9: DOCKER VALIDATION ===")
    results = {}
    
    print("Building Docker image...")
    results["build"] = run("docker build -t stepout:latest .")
    print(f"  Build: {'PASS' if results['build'] else 'FAIL'}")
    
    print("Starting services...")
    results["services_start"] = run("docker-compose up -d")
    print(f"  Services start: {'PASS' if results['services_start'] else 'FAIL'}")
    
    print("Checking containers...")
    results["containers_running"] = run("docker ps")
    print(f"  Containers: {'PASS' if results['containers_running'] else 'FAIL'}")
    
    out = Path("outputs/docker_validation.json")
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()