# Utils/OnlineUpdate.py

import copy
from Utils.KDPrepare import prepare_novel_kd_batch, build_teacher_student_models
from Utils.KDTrainer import incremental_kd_update


def perform_incremental_update(
    model,
    loader,
    batch_df,
    unseen_buffer_df,
    old_token_vocab,
    old_label_vocab,
    learned_novel_events,
    device,
    batch_size=32,
    adaptation_epochs=20,
    kd_epochs=4,
    lambda_kd=0.5,
    temperature=2.0,
    adaptation_lr=2e-3,
    kd_lr=1e-3,
    use_kd=True,
    full_finetune_ce_only=True,
    adaptation_val_ratio=0.1,
    adaptation_patience=3,
    adaptation_min_delta=1e-3,
    verbose=True,
):
    kd_batch = prepare_novel_kd_batch(
        unseen_buffer_df=unseen_buffer_df,
        trigger_window_df=batch_df,
        old_token_vocab=old_token_vocab,
        old_label_vocab=old_label_vocab
    )

    learned_novel_events.update(kd_batch.new_tokens)
    learned_novel_events.update(kd_batch.new_labels)

    loader.vocab_mapper.expand_token_vocab(
        [row.split() for row in kd_batch.novel_df["prefix"].astype(str).tolist()]
    )
    loader.vocab_mapper.expand_label_vocab(
        kd_batch.novel_df["next_act"].astype(str).tolist()
    )

    new_vocab_size = len(loader.vocab_mapper.token_vocab)
    new_num_classes = len(loader.vocab_mapper.label_vocab)

    teacher, student = build_teacher_student_models(
        model=model,
        new_vocab_size=new_vocab_size,
        new_num_classes=new_num_classes,
        device=device,
    )

    student, kd_history = incremental_kd_update(
        kd_batch=kd_batch,
        teacher=teacher,
        student=student,
        loader=loader,
        old_token_vocab=old_token_vocab,
        old_label_vocab=old_label_vocab,
        batch_size=batch_size,
        adaptation_epochs=adaptation_epochs,
        kd_epochs=kd_epochs,
        lambda_kd=lambda_kd,
        temperature=temperature,
        adaptation_lr=adaptation_lr,
        kd_lr=kd_lr,
        use_kd=use_kd,
        full_finetune_ce_only=full_finetune_ce_only,
        adaptation_val_ratio=adaptation_val_ratio,
        adaptation_patience=adaptation_patience,
        adaptation_min_delta=adaptation_min_delta,
        device=device,
        verbose=verbose,
    )

    model = student

    old_token_vocab = copy.deepcopy(loader.vocab_mapper.token_vocab)
    old_label_vocab = copy.deepcopy(loader.vocab_mapper.label_vocab)

    return model, old_token_vocab, old_label_vocab, learned_novel_events, kd_history