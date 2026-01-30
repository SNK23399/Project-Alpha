#!/usr/bin/env python
"""
Test adaptive convergence MC on the pipeline_copy
"""
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import time

print("\n" + "=" * 80)
print("TESTING ADAPTIVE CONVERGENCE MC")
print("=" * 80)
print(f"Start time: {datetime.now().isoformat()}")
print()

# Modify pipeline_copy to test only N=5
script_path = Path("pipeline copy/5_precompute_mc_ir_stats.py")

print("Running test with:")
print(f"  - Adaptive convergence enabled")
print(f"  - Batch size: 500,000 samples")
print(f"  - Convergence threshold: 0.1%")
print(f"  - N value: 5 satellites only (for speed)")
print()

# Replace N_SATELLITES_TO_PRECOMPUTE temporarily
test_code = '''
import sys
sys.path.insert(0, ".")
from pathlib import Path

# Quick test - only N=5
import importlib.util
spec = importlib.util.spec_from_file_location("mc_module", "pipeline copy/5_precompute_mc_ir_stats.py")
mc_module = importlib.util.module_from_spec(spec)

# Override N values before loading
sys.modules['mc_module'] = mc_module
import pipeline_copy.data as would_import  # This will fail, but we can manually patch

# Instead, just run main through the module directly
spec.loader.exec_module(mc_module)

# Now patch N_SATELLITES_TO_PRECOMPUTE
original_n = mc_module.N_SATELLITES_TO_PRECOMPUTE
mc_module.N_SATELLITES_TO_PRECOMPUTE = [5]  # Only test N=5

# Run main
if hasattr(mc_module, 'main'):
    mc_module.main()
else:
    # Run full pipeline step
    import subprocess
    result = subprocess.run([sys.executable, "pipeline copy/5_precompute_mc_ir_stats.py"])
    sys.exit(result.returncode)
'''

start_time = time.time()
result = subprocess.run(
    [sys.executable, "-c", test_code],
    cwd=str(Path.cwd())
)
end_time = time.time()

print()
print("=" * 80)
if result.returncode == 0:
    elapsed = end_time - start_time
    print(f"SUCCESS! Test completed in {elapsed:.1f} seconds")
    print(f"End time: {datetime.now().isoformat()}")
else:
    print(f"FAILED with return code {result.returncode}")
    print(f"End time: {datetime.now().isoformat()}")

sys.exit(result.returncode)
