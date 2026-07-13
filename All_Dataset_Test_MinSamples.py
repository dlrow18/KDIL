import subprocess
import sys
import time
import math

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

total_prefix_samples = {
    "BPIC11": 149148,
    "BPIC15_1": 51018,
    "BPIC15_2": 43522,
    "BPIC15_3": 58272,
    "BPIC15_4": 46240,
    "BPIC15_5": 57927,
    "BPIC20P": 79517,
    "helpdesk": 16768,
    "sepsis": 14141,
}

'''
# ratio for min_samples_per_class setting
ratio = 0.1

for idx, dataset in enumerate(datasets, start=1):
    n_prefix = total_prefix_samples[dataset]
    min_sample_per_class = max(1, math.ceil(ratio * n_prefix))

    excel_path = (
        f"./runs/window_metrics_"
        f"{dataset}_"
        f"minsamples_{min_sample_per_class}.xlsx"
    )

    print("=" * 80)
    print(f"[{idx}/{len(datasets)}] Running dataset: {dataset}")
    print(f"ratio={ratio}, n_prefix={n_prefix}, min_sample_per_class={min_sample_per_class}")
    print("=" * 80)

    cmd = [
        sys.executable,
        "KDTest.py",
        "--dataset", dataset,
        "--window_type", "month",
        "--save_excel", "True",
        "--excel_path", excel_path,
        "--min_sample_per_class", str(min_sample_per_class),
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

'''

# ratio for setting the min samples per class according to total number of prefixes to be predicted in each dataset
ratios = [0.001, 0.01, 0.05, 0.075, 0.1, 0.25, 0.5]

for ratio in ratios:
    for idx, dataset in enumerate(datasets, start=1):
        n_prefix = total_prefix_samples[dataset]
        min_sample_per_class = max(1, math.ceil(ratio * n_prefix))

        excel_path = (
            f"./runs/window_metrics_"
            f"{dataset}_"
            f"MinSamples_{min_sample_per_class}.xlsx"
        )

        print("=" * 80)
        print(f"[ratio={ratio}] [{idx}/{len(datasets)}] Running dataset: {dataset}")
        print(f"n_prefix={n_prefix}")
        print(f"min_sample_per_class={min_sample_per_class}")
        print(f"excel_path={excel_path}")
        print("=" * 80)

        cmd = [
            sys.executable,
            "KDTest.py",
            "--dataset", dataset,
            "--window_type", "month",
            "--save_excel", "True",
            "--excel_path", excel_path,
            "--min_sample_per_class", str(min_sample_per_class),
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
