import subprocess
import sys
import time

'''
datasets = [
    "BPIC11",
    "BPIC15_1",
    "BPIC15_2",
    "BPIC15_3",
    "BPIC15_4",
    "BPIC15_5",
    "BPIC20P",
    "helpdesk",
    "sepsis",
]
'''

datasets = [
    "BPIC15_2"
]

# min_unseen ratios in windows setting
# ratios = [0.01, 0.1, 0.5, 0.9]
ratios = [0.01]


for ratio in ratios:
    for idx, dataset in enumerate(datasets, start=1):

        excel_path = (
            f"./runs/window_metrics_"
            f"{dataset}_"
            f"MinRatio_{ratio}.xlsx"
        )

        print("=" * 80)
        print(f"[ratio={ratio}] [{idx}/{len(datasets)}] Running dataset: {dataset}")
        print(f"excel_path={excel_path}")
        print("=" * 80)

        cmd = [
            sys.executable,
            "KDTest_MinRatio.py",
            "--dataset", dataset,
            "--window_type", "month",
            "--save_excel", "True",
            "--excel_path", excel_path,
            "--min_unseen_ratio", str(ratio),
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
