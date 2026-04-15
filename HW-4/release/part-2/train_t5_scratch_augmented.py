import os

from torch.utils.data import DataLoader

import train_t5_scratch as base


AUGMENTED_TRAIN_PREFIX = "train_plus_augmented"


class AugmentedScratchT5Dataset(base.ScratchT5Dataset):
    def process_data(self, data_folder, split):
        source_split = AUGMENTED_TRAIN_PREFIX if split == "train" else split
        nl_queries = base.load_lines(os.path.join(data_folder, f"{source_split}.nl"))
        sql_queries = None if split == "test" else base.load_lines(os.path.join(data_folder, f"{source_split}.sql"))

        examples = []
        if split == "test":
            for nl_query in nl_queries:
                encoder_ids = self.tokenizer(
                    base.build_encoder_input(nl_query),
                    add_special_tokens=True,
                    return_attention_mask=False,
                )["input_ids"]
                examples.append(
                    {
                        "encoder_ids": base.torch.tensor(encoder_ids, dtype=base.torch.long),
                        "initial_decoder_input": base.torch.tensor([self.decoder_start_token_id], dtype=base.torch.long),
                    }
                )
            return examples

        for nl_query, sql_query in zip(nl_queries, sql_queries):
            sql_ids = self.tokenizer(
                sql_query,
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
            if split == "train" and len(sql_ids) > base.MAX_SQL_TOKENS:
                continue

            encoder_ids = self.tokenizer(
                base.build_encoder_input(nl_query),
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
            decoder_sequence = [self.decoder_start_token_id] + sql_ids
            examples.append(
                {
                    "encoder_ids": base.torch.tensor(encoder_ids, dtype=base.torch.long),
                    "decoder_sequence": base.torch.tensor(decoder_sequence, dtype=base.torch.long),
                    "initial_decoder_input": base.torch.tensor([self.decoder_start_token_id], dtype=base.torch.long),
                }
            )
        return examples


def get_dataloader(batch_size, split):
    dset = AugmentedScratchT5Dataset("data", split)
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=base.normal_collate_fn if split != "test" else base.test_collate_fn,
    )


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def main():
    args = base.get_args()
    args.experiment_name = base.attach_slurm_job_id(args.experiment_name)
    base.set_random_seeds(args.seed)

    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = base.initialize_model()
    optimizer = base.initialize_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = base.initialize_scheduler(args, optimizer, len(train_loader))

    best_state_dict = base.train(args, model, train_loader, dev_loader, optimizer, scheduler)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    experiment_name = args.experiment_name
    model_type = "scr"
    gt_sql_path = "data/dev.sql"
    gt_record_path = "records/ground_truth_dev.pkl"
    dev_sql_path = os.path.join("results", f"t5_{model_type}_{experiment_name}_dev.sql")
    dev_record_path = os.path.join("records", f"t5_{model_type}_{experiment_name}_dev.pkl")

    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = base.eval_epoch(
        args, model, dev_loader, gt_sql_path, dev_sql_path, gt_record_path, dev_record_path
    )
    print(
        f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, "
        f"Record EM: {dev_record_em}, SQL EM: {dev_sql_em}"
    )
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    test_sql_path = os.path.join("results", f"t5_{model_type}_{experiment_name}_test.sql")
    test_record_path = os.path.join("records", f"t5_{model_type}_{experiment_name}_test.pkl")
    base.test_inference(args, model, test_loader, test_sql_path, test_record_path)

    if not args.skip_submission_copy:
        base.copy_extra_credit_submission_files(args)
        print("Copied extra-credit submission files:")
        print("  results/t5_ft_experiment_ec_test.sql")
        print("  records/t5_ft_experiment_ec_test.pkl")
        print(f"  results/t5_ft_experiment_ec_{experiment_name}_test.sql")
        print(f"  records/t5_ft_experiment_ec_{experiment_name}_test.pkl")

    print("\nUsing augmented training data from:")
    print(f"  data/{AUGMENTED_TRAIN_PREFIX}.nl")
    print(f"  data/{AUGMENTED_TRAIN_PREFIX}.sql")
    print("\nIf you see loss improving while F1 plateaus, try:")
    print("- Changing the learning rate.")
    print("- Verifying BOS/EOS/special tokens are set correctly.")
    print("- Checking whether your decoding / sampling setup is appropriate.")


if __name__ == "__main__":
    main()
