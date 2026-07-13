import subprocess
import sys
import time

datasets = [
    "BPIC11",
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
    "BPIC20P",
    "Helpdesk",
    "Sepsis",
]

for idx, dataset in enumerate(datasets, start=1):
    print("=" * 80)
    print(f"[{idx}/{len(datasets)}] Running dataset: {dataset}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "KDTest.py",
        "--dataset", dataset,
        "--window_type", "month",
        "--save_excel", "True",
    ]

    start_time = time.time()

    result = subprocess.run(cmd)

    elapsed = time.time() - start_time
    print(f"\nFinished dataset: {dataset}")
    print(f"Elapsed time: {elapsed / 60:.2f} minutes")

    if result.returncode != 0:
        print(f"\n[ERROR] Dataset {dataset} failed with return code {result.returncode}.")
        print("Stopping the batch run.")
        sys.exit(result.returncode)

print("\nAll datasets finished successfully.")