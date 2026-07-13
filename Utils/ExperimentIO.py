import os
import pandas as pd

"""
  Save one dataset's window-level records into a shared Excel file.

  Behavior:
  - If the Excel file does not exist, create it.
  - If it exists, preserve all other datasets' rows.
  - Replace only the rows of the current dataset_name.
  """

'''
# functions for recording results to Excel
def _str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")
'''
def save_window_metrics_to_excel(
        records,
        dataset_name: str,
        excel_path: str,
        sheet_name: str = "all_windows",
):
    if not records:
        return

    os.makedirs(os.path.dirname(excel_path) or ".", exist_ok=True)

    new_df = pd.DataFrame(records)

    preferred_cols = [
        "dataset",
        "window_index",
        "window_id",
        "n_samples",
        "unseen_count",
        "unseen_ratio",
        "unseen_event_ratio",
        "acc",
        "precision",
        "recall",
        "f1",
    ]
    existing_cols = [c for c in preferred_cols if c in new_df.columns]
    other_cols = [c for c in new_df.columns if c not in existing_cols]

    # Put total_update_count at the very end if it exists
    if "total_update_count" in other_cols:
        other_cols.remove("total_update_count")
        other_cols.append("total_update_count")

    new_df = new_df[existing_cols + other_cols]

    if os.path.exists(excel_path):
        try:
            old_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        except Exception:
            old_df = pd.DataFrame()

        if not old_df.empty and "dataset" in old_df.columns:
            old_df = old_df[old_df["dataset"] != dataset_name]

        all_windows_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        all_windows_df = new_df.copy()

    sort_cols = [c for c in ["dataset", "window_index"] if c in all_windows_df.columns]
    if sort_cols:
        all_windows_df = all_windows_df.sort_values(sort_cols).reset_index(drop=True)

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        all_windows_df.to_excel(writer, sheet_name=sheet_name, index=False)

    #print(f"[Excel] Saved dataset '{dataset_name}' to {excel_path} (sheet: {sheet_name})")


def save_checkpoint_and_vocab(model, loader, out_dir: str, dataset: str):
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(out_dir, f"{dataset}_baseline.pt")
    vocab_path = os.path.join(out_dir, f"{dataset}_vocab.json")

    model.save_model(ckpt_path)
    loader.vocab_mapper.save_vocab(vocab_path)

    #print(f"[Checkpoint] Saved model to {ckpt_path}")
    #print(f"[Checkpoint] Saved vocab to {vocab_path}")


def build_window_record(
    dataset_name,
    window_index,
    window_id,
    batch_df,
    win_acc,
    win_p,
    win_r,
    win_f1,
    info,
    unseen_event_ratio,
    current_unseen_target_n,
    current_unseen_target_ratio,
    learned_novel_target_metrics,
    novel_context_old_target_metrics,
    novel_decomp,
):
    return {
        "dataset": dataset_name,
        "window_index": window_index,
        "window_id": str(window_id),
        "n_samples": int(len(batch_df)),

        "acc": float(win_acc),
        "precision": float(win_p),
        "recall": float(win_r),
        "f1": float(win_f1),

        "unseen_count": int(info["unseen_count_in_window"]),
        "unseen_ratio": float(info["unseen_ratio_in_window"]),
        "unseen_event_ratio": float(unseen_event_ratio),

        "current_unseen_target_n": int(current_unseen_target_n),
        "current_unseen_target_ratio": float(current_unseen_target_ratio),

        "learned_novel_target_n": learned_novel_target_metrics["n"],
        "learned_novel_target_acc": learned_novel_target_metrics["acc"],
        "learned_novel_target_precision": learned_novel_target_metrics["precision"],
        "learned_novel_target_recall": learned_novel_target_metrics["recall"],
        "learned_novel_target_f1": learned_novel_target_metrics["f1"],

        "novel_context_old_target_n": novel_context_old_target_metrics["n"],
        "novel_context_old_target_acc": novel_context_old_target_metrics["acc"],
        "novel_context_old_target_precision": novel_context_old_target_metrics["precision"],
        "novel_context_old_target_recall": novel_context_old_target_metrics["recall"],
        "novel_context_old_target_f1": novel_context_old_target_metrics["f1"],

        "global_unseen_event_count": int(novel_decomp["global_unseen_event_count"]),
        "global_unseen_event_ratio": float(novel_decomp["global_unseen_event_ratio"]),
        "local_unseen_event_count": int(novel_decomp["local_unseen_event_count"]),
        "local_unseen_event_ratio": float(novel_decomp["local_unseen_event_ratio"]),
        "learned_novel_event_count": int(novel_decomp["learned_novel_event_count"]),
        "learned_novel_event_ratio": float(novel_decomp["learned_novel_event_ratio"]),
    }

def build_overall_record(
    dataset_name,
    window_index,
    n_samples,
    total_update_count,
    overall_acc,
    overall_p,
    overall_r,
    overall_f1,
    overall_uhs,
    overall_learned_novel_target_metrics,
    overall_novel_context_old_target_metrics,
):
    return {
        "dataset": dataset_name,
        "window_index": window_index,
        "window_id": "overall",
        "n_samples": int(n_samples),

        "total_update_count": int(total_update_count),

        "unseen_count": None,
        "unseen_ratio": None,
        "unseen_event_ratio": None,

        "acc": float(overall_acc),
        "uhs": None if overall_uhs is None else float(overall_uhs),
        "precision": float(overall_p),
        "recall": float(overall_r),
        "f1": float(overall_f1),

        "learned_novel_target_n": overall_learned_novel_target_metrics["n"],
        "learned_novel_target_acc": overall_learned_novel_target_metrics["acc"],
        "learned_novel_target_precision": overall_learned_novel_target_metrics["precision"],
        "learned_novel_target_recall": overall_learned_novel_target_metrics["recall"],
        "learned_novel_target_f1": overall_learned_novel_target_metrics["f1"],

        "novel_context_old_target_n": overall_novel_context_old_target_metrics["n"],
        "novel_context_old_target_acc": overall_novel_context_old_target_metrics["acc"],
        "novel_context_old_target_precision": overall_novel_context_old_target_metrics["precision"],
        "novel_context_old_target_recall": overall_novel_context_old_target_metrics["recall"],
        "novel_context_old_target_f1": overall_novel_context_old_target_metrics["f1"],
    }

