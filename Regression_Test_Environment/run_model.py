import subprocess
import sys

scripts = [
    "clean_data.py",
    "logistic_regression/logistic_regression_lib.py",
    "logistic_regression/logistic_regression_own.py"
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, script])

    if result.returncode != 0:
        print(f"ERROR: {script} failed. Stopping.")
        break

print("Pipeline finished")
