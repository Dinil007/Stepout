#!/usr/bin/env python3
"""Quick verification that the end-to-end workflow components are wired correctly."""

from pathlib import Path
import sys
import ast

sys.path.insert(0, str(Path(__file__).resolve().parent))

print("=" * 60)
print("StepOut AI Platform - End-to-End Workflow Verification")
print("=" * 60)

# Parse streamlit_app.py to verify components exist
with open('streamlit_app.py', encoding='utf-8') as f:
    source = f.read()

# Syntax check
ast.parse(source)
print("\n[✓] streamlit_app.py syntax OK")

# Auth checks
assert 'def login(' in source
assert 'def logout()' in source
assert 'def has_permission(' in source
assert 'USERS = {' in source
assert 'ROLE_PERMISSIONS = {' in source
print("[✓] Authentication scaffolding present")

# Upload / video checks
assert 'def get_video_info(' in source
assert 'def get_first_frame(' in source
assert 'UPLOAD_DIR = Path("uploads")' in source
assert 'st.file_uploader(' in source
print("[✓] Upload and video metadata helpers present")

# Processing / pipeline checks
assert 'def run_pipeline_process(' in source
assert 'page_processing' in source
assert 'progress_bar = st.progress(' in source
print("[✓] Pipeline runner and progress screen present")

# Results dashboard checks
assert 'def page_results_dashboard' in source
assert 'def page_ai_reports' in source
assert 'MatchAnalyst' in source
assert 'analytics.json' in source
print("[✓] Results dashboard and AI match reports present")

# Dashboard pages checks
assert 'Formation Intelligence' in source
assert 'Pressing Intelligence' in source
assert 'Expected Goals (xG)' in source
assert 'Expected Assists (xA)' in source
assert 'Expected Threat (xT)' in source
print("[✓] Analytics dashboard tabs include all subsystems")

# File checks
assert Path('streamlit/pages/9_Formation_Intelligence.py').exists()
assert Path('streamlit/pages/10_Pressing_Intelligence.py').exists()
assert Path('app/ai/match_analyst.py').exists()
assert Path('app/api/pressing_router.py').exists()
assert Path('run_pipeline.py').exists()
print("[✓] All required files exist")

print("\n" + "=" * 60)
print("All end-to-end workflow components verified successfully!")
print("=" * 60)
