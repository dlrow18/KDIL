from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader


# UNK_TOKEN = "[UNK]"

# D_unseen: new examples with unseen token/labels
@dataclass
class IncrementalKDBatch:
    novel_df: pd.DataFrame
    stable_df: pd.DataFrame
    new_tokens: List[str]
    new_labels: List[str]

# Extract new tokens and labels from D_unseen that are not in old vocabularies.
def _extract_new_tokens_and_labels(
    novel_df: pd.DataFrame,
    old_token_vocab: Dict[str, int],
    old_label_vocab: Dict[str, int],
) -> Tuple[List[str], List[str]]:
    new_tokens = []
    seen_tokens = set()
    for p in novel_df["prefix"].astype(str).tolist():
        for tok in p.split():
            if tok not in old_token_vocab and tok not in seen_tokens:
                seen_tokens.add(tok)
                new_tokens.append(tok)

    new_labels = []
    seen_labels = set()
    for y in novel_df["next_act"].astype(str).tolist():
        if y not in old_label_vocab and y not in seen_labels:
            seen_labels.add(y)
            new_labels.append(y)

    return new_tokens, new_labels


def prepare_novel_kd_batch(
    unseen_buffer_df: pd.DataFrame,
    trigger_window_df: pd.DataFrame,
    old_token_vocab: Dict[str, int],
    old_label_vocab: Dict[str, int],
) -> IncrementalKDBatch:
    """
    Build data for the two-phase incremental update.

    - novel_df:
        D_novel from the unseen-event buffer.
        Used for CE-based adaptation.

    - stable_df:
        D_stable from the trigger window.
        Contains only old-token prefixes and old labels.
        Used for old-class knowledge distillation.

    - new_tokens / new_labels:
        Newly observed activities in prefixes and target labels.
    """
    novel_df = unseen_buffer_df.copy().reset_index(drop=True)

    new_tokens, new_labels = _extract_new_tokens_and_labels(
        novel_df=novel_df,
        old_token_vocab=old_token_vocab,
        old_label_vocab=old_label_vocab,
    )

    stable_df = build_stable_df(
        trigger_window_df=trigger_window_df,
        old_token_vocab=old_token_vocab,
        old_label_vocab=old_label_vocab,
    )

    return IncrementalKDBatch(
        novel_df=novel_df,
        stable_df=stable_df,
        new_tokens=new_tokens,
        new_labels=new_labels,
    )


# Build D_stable from trigger window: only examples where prefix has all old tokens and next_act is an old label.
# Stable means "seen" in the current model
def build_stable_df(
    trigger_window_df: pd.DataFrame,
    old_token_vocab: Dict[str, int],
    old_label_vocab: Dict[str, int],
) -> pd.DataFrame:

    stable_rows = []

    for _, row in trigger_window_df.iterrows():
        prefix = str(row["prefix"])
        label = str(row["next_act"])

        toks = prefix.split()
        prefix_all_old = all(tok in old_token_vocab for tok in toks)
        label_old = label in old_label_vocab

        if prefix_all_old and label_old:
            stable_rows.append(row)

    if len(stable_rows) == 0:
        return trigger_window_df.iloc[0:0].copy()

    return pd.DataFrame(stable_rows).reset_index(drop=True)

# Encode a dataframe into a DataLoader using the given vocab_mapper.
def encode_df_with_given_vocab(
    df: pd.DataFrame,
    vocab_mapper,
    max_case_length: int,
    batch_size: int = 32,
    shuffle: bool = True,
    expand_tokens: bool = True,
    expand_labels: bool = True,
    allow_unknown_labels: bool = False,
) -> DataLoader:
    token_seqs = [row.split() for row in df["prefix"].astype(str).tolist()]
    labels = df["next_act"].astype(str).tolist()

    if expand_tokens:
        vocab_mapper.expand_token_vocab(token_seqs)
    if expand_labels:
        vocab_mapper.expand_label_vocab(labels)

    batch_ids = []
    lengths = []
    for seq in token_seqs:
        idxs = []
        for tok in seq:
            if tok in vocab_mapper.token_vocab:
                idxs.append(vocab_mapper.token_vocab[tok])
            else:
                idxs.append(vocab_mapper.token_vocab[UNK_TOKEN])

        seq_len = min(len(idxs), max_case_length)
        lengths.append(seq_len)

        if len(idxs) > max_case_length:
            idxs = idxs[-max_case_length:]

        pad_len = max_case_length - len(idxs)
        batch_ids.append([vocab_mapper.pad_idx] * pad_len + idxs)

    label_idxs = []
    for y in labels:
        if y in vocab_mapper.label_vocab:
            label_idxs.append(vocab_mapper.label_vocab[y])
        else:
            if allow_unknown_labels:
                label_idxs.append(-1)
            else:
                raise KeyError(f"Unknown label '{y}' with expand_labels=False")

    label_tensor = torch.tensor(label_idxs, dtype=torch.long)
    input_tensor = torch.tensor(batch_ids, dtype=torch.long)
    length_tensor = torch.tensor(lengths, dtype=torch.long)

    dataset = TensorDataset(input_tensor, label_tensor, length_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# Build teacher and student models by copying the original model. Teacher is frozen and student is expanded with new vocab sizes.
def build_teacher_student_models(model, new_vocab_size: int, new_num_classes: int, device=None):

    teacher = copy.deepcopy(model)
    student = copy.deepcopy(model)

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    student.expand_vocab(new_vocab_size)
    student.expand_num_classes(new_num_classes)

    if device is not None:
        teacher = teacher.to(device)
        student = student.to(device)

    return teacher, student