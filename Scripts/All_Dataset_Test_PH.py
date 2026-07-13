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

# PH lambdas in windows setting
# ph_lambdas = [0.01, 0.1, 0.5, 0.9]
ph_lambdas = [0.05]


for ph_lambda in ph_lambdas:
    for idx, dataset in enumerate(datasets, start=1):

        excel_path = (
            f"./runs/window_metrics_"
            f"{dataset}_"
            f"PH_{ph_lambda}.xlsx"
        )

        print("=" * 80)
        print(f"[ph_lambda={ph_lambda}] [{idx}/{len(datasets)}] Running dataset: {dataset}")
        print(f"excel_path={excel_path}")
        print("=" * 80)

        cmd = [
            sys.executable,
            "KDTest_PH.py",
            "--dataset", dataset,
            "--window_type", "month",
            "--save_excel", "True",
            "--excel_path", excel_path,
            "--ph_lambda", str(ph_lambda),
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
