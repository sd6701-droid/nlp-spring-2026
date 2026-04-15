import argparse
import os
import random
import re
import site
import sys

from tqdm import tqdm

user_site = site.getusersitepackages()
if isinstance(user_site, str):
    user_sites = [user_site]
else:
    user_sites = list(user_site)
for user_site_path in user_sites:
    while user_site_path in sys.path:
        sys.path.remove(user_site_path)

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    GenerationConfig,
    T5Config,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from utils import compute_metrics, save_queries_and_records


MODEL_NAME = "google-t5/t5-small"
PAD_IDX = 0
TASK_PREFIX = "translate English to SQL: "
MAX_SQL_TOKENS = 400
SCHEMA_HINT = (
    "tables: flight, airport_service, city, airline, fare, flight_fare, "
    "flight_stop, ground_service, airport, days, date_day, fare_basis, restriction"
)

CITY_MAP = {
    "los angeles": "LOS ANGELES",
    "la": "LOS ANGELES",
    "new york city": "NEW YORK",
    "new york": "NEW YORK",
    "nyc": "NEW YORK",
    "san francisco": "SAN FRANCISCO",
    "sf": "SAN FRANCISCO",
    "washington dc": "WASHINGTON",
    "dc": "WASHINGTON",
    "washington": "WASHINGTON",
    "fort worth": "FORT WORTH",
    "salt lake city": "SALT LAKE CITY",
    "kansas city": "KANSAS CITY",
    "boston": "BOSTON",
    "denver": "DENVER",
    "dallas": "DALLAS",
    "pittsburgh": "PITTSBURGH",
    "philadelphia": "PHILADELPHIA",
    "atlanta": "ATLANTA",
    "chicago": "CHICAGO",
    "houston": "HOUSTON",
    "baltimore": "BALTIMORE",
    "miami": "MIAMI",
    "seattle": "SEATTLE",
    "minneapolis": "MINNEAPOLIS",
    "detroit": "DETROIT",
    "charlotte": "CHARLOTTE",
    "cleveland": "CLEVELAND",
    "indianapolis": "INDIANAPOLIS",
    "orlando": "ORLANDO",
    "milwaukee": "MILWAUKEE",
    "toronto": "TORONTO",
    "newark": "NEWARK",
    "nashville": "NASHVILLE",
    "memphis": "MEMPHIS",
    "st. louis": "ST. LOUIS",
    "saint louis": "ST. LOUIS",
    "columbus": "COLUMBUS",
    "cincinnati": "CINCINNATI",
    "san diego": "SAN DIEGO",
    "phoenix": "PHOENIX",
    "las vegas": "LAS VEGAS",
    "oakland": "OAKLAND",
    "ontario": "ONTARIO",
    "long beach": "LONG BEACH",
    "burbank": "BURBANK",
    "tacoma": "TACOMA",
}
AIRLINE_MAP = {
    # Common spoken names → exact DB codes
    "american airlines": "AA", "american": "AA",
    "united airlines": "UA", "united": "UA",
    "us air": "US", "usair": "US",              # DB value is "USAIR"
    "delta air lines": "DL", "delta airlines": "DL", "delta": "DL",
    "midwest express": "YX", "midwest express airlines": "YX",
    "continental airlines": "CO", "continental": "CO",
    "southwest airlines": "WN", "southwest": "WN",
    "alaska airlines": "AS", "alaska": "AS",
    "northwest airlines": "NW", "northwest": "NW",
    "twa": "TW", "trans world airlines": "TW", "trans world": "TW",
    "lufthansa": "LH", "lufthansa german airlines": "LH",
    "tower air": "FF",
    
    # NEW — were completely missing from your map
    "america west": "HP", "america west airlines": "HP",
    "air canada": "AC",
    "british airways": "BA",
    "atlantic coast airlines": "DH",
    "atlantic southeast": "EV", "atlantic southeast airlines": "EV",
    "canadian airlines": "CP", "canadian": "CP",
    "comair": "OH",
    "business express": "HQ",
    "alpha air": "7V",
    "air wisconsin": "ZW",
    "skywest": "OO", "sky west": "OO", "skywest airlines": "OO",
    "mesaba": "XJ", "mesaba aviation": "XJ",
    "trans states airlines": "9N", "trans states": "9N",
    "mgm grand air": "MG", "mgm grand": "MG",
    "colgan air": "9L", "colgan": "9L",
    "precision airlines": "RP", "precision": "RP",
    "nationair": "NX",
    "american trans air": "TZ",
    "thai airways": "TG",
    "air alliance": "3J",
    "air ontario": "GX",
    "carnival air lines": "KW", "carnival air": "KW",
    "northeast express": "2V",
}
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def attach_slurm_job_id(experiment_name):
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        return experiment_name

    suffix = f"_{job_id}"
    if experiment_name.endswith(suffix):
        return experiment_name
    return f"{experiment_name}{suffix}"


def set_random_seeds(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_lines(path):
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def has_phrase(text, phrase):
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def normalize_nl(nl_query):
    query_lower = nl_query.lower()
    hints = []

    # Sort longest phrase first to prevent partial matches
    # e.g. "american airlines" must match before "american"
    for city_name, canonical_name in sorted(CITY_MAP.items(),
                                             key=lambda x: len(x[0]),
                                             reverse=True):
        if has_phrase(query_lower, city_name):
            hint = f"[city={canonical_name}]"
            if hint not in hints:
                hints.append(hint)

    for airline_name, airline_code in sorted(AIRLINE_MAP.items(),
                                              key=lambda x: len(x[0]),
                                              reverse=True):
        if has_phrase(query_lower, airline_name):
            hint = f"[airline={airline_code}]"
            if hint not in hints:
                hints.append(hint)

    # Aggregation hints
    if any(word in query_lower for word in ["earliest", "first", "minimum", "lowest", "cheapest", "least"]):
        hints.append("[agg=MIN]")
    if any(word in query_lower for word in ["latest", "last", "maximum", "highest", "most", "largest"]):
        hints.append("[agg=MAX]")

    # Select column hints
    if any(word in query_lower for word in ["flight time", "departure time", "arrival time", "what time"]):
        hints.append("[select=departure_time]")
    if any(word in query_lower for word in ["cost", "price", "fare", "how much"]):
        hints.append("[select=fare_id]")
    if any(word in query_lower for word in ["transport", "ground transportation", "train", "bus", "rental"]):
        hints.append("[select=transport_type]")

    # Day/date join hint — tells model to join days + date_day tables
    DAY_WORDS = [
        "monday", "tuesday", "wednesday", "thursday", "friday",
        "saturday", "sunday", "tomorrow", "today", "tonight",
        "morning", "afternoon", "evening", "weekend",
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    if any(word in query_lower for word in DAY_WORDS):
        hints.append("[join=days,date_day]")

    if not hints:
        return nl_query
    return f"{nl_query} {' '.join(hints)}"

def build_encoder_input(nl_query):
    normalized_nl = normalize_nl(nl_query)
    return f"{TASK_PREFIX}{normalized_nl} | {SCHEMA_HINT}"


class ScratchT5Dataset(Dataset):
    def __init__(self, data_folder, split):
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
        self.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        self.examples = self.process_data(data_folder, split)

    def process_data(self, data_folder, split):
        nl_queries = load_lines(os.path.join(data_folder, f"{split}.nl"))
        sql_queries = None if split == "test" else load_lines(os.path.join(data_folder, f"{split}.sql"))

        examples = []
        if split == "test":
            for nl_query in nl_queries:
                encoder_ids = self.tokenizer(
                    build_encoder_input(nl_query),
                    add_special_tokens=True,
                    return_attention_mask=False,
                )["input_ids"]
                examples.append(
                    {
                        "encoder_ids": torch.tensor(encoder_ids, dtype=torch.long),
                        "initial_decoder_input": torch.tensor([self.decoder_start_token_id], dtype=torch.long),
                    }
                )
            return examples

        for nl_query, sql_query in zip(nl_queries, sql_queries):
            sql_ids = self.tokenizer(
                sql_query,
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
            if split == "train" and len(sql_ids) > MAX_SQL_TOKENS:
                continue

            encoder_ids = self.tokenizer(
                build_encoder_input(nl_query),
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
            decoder_sequence = [self.decoder_start_token_id] + sql_ids
            examples.append(
                {
                    "encoder_ids": torch.tensor(encoder_ids, dtype=torch.long),
                    "decoder_sequence": torch.tensor(decoder_sequence, dtype=torch.long),
                    "initial_decoder_input": torch.tensor([self.decoder_start_token_id], dtype=torch.long),
                }
            )
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def normal_collate_fn(batch):
    encoder_ids = pad_sequence(
        [example["encoder_ids"] for example in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()

    decoder_sequences = [example["decoder_sequence"] for example in batch]
    decoder_inputs = pad_sequence(
        [sequence[:-1] for sequence in decoder_sequences],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    decoder_targets = pad_sequence(
        [sequence[1:] for sequence in decoder_sequences],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    initial_decoder_inputs = torch.stack(
        [example["initial_decoder_input"] for example in batch],
        dim=0,
    )

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs


def test_collate_fn(batch):
    encoder_ids = pad_sequence(
        [example["encoder_ids"] for example in batch],
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack(
        [example["initial_decoder_input"] for example in batch],
        dim=0,
    )
    return encoder_ids, encoder_mask, initial_decoder_inputs


def get_dataloader(batch_size, split):
    dset = ScratchT5Dataset("data", split)
    return DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        collate_fn=normal_collate_fn if split != "test" else test_collate_fn,
    )


def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    return train_loader, dev_loader, test_loader


def build_generation_config(args, model):
    return GenerationConfig(
        max_new_tokens=args.max_generation_length,
        num_beams=args.num_beams,
        do_sample=False,
        early_stopping=True,
        decoder_start_token_id=model.config.decoder_start_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
    )


def clean_decoded_query(query):
    query = query.replace("\n", " ").replace("\t", " ").strip()

    # Fix missing spaces around comparison operators before numbers
    query = re.sub(r'([<>])\s*(\d)', r'\1 \2', query)
    
    # Fix missing space between number and AND/OR
    query = re.sub(r'(\d)(AND|OR|WHERE)', r'\1 \2', query, flags=re.IGNORECASE)
    
    # Fix number stuck to closing paren: "800)" → "800 )"
    query = re.sub(r"(\d)\)", r"\1 )", query)

    # Fix string literal stuck to closing paren
    query = re.sub(r"('[\w\s]+')\)", r"\1 )", query)

    # Remove AND 1=1 tautologies
    query = re.sub(r'\bAND\s+1\s*=\s*1\b', '', query, flags=re.IGNORECASE)
    query = re.sub(r'\bWHERE\s+1\s*=\s*1\s+AND\b', 'WHERE', query, flags=re.IGNORECASE)

    # Fix dangling commas before WHERE / closing paren
    query = re.sub(r",\s*(WHERE\b)", r" \1", query)
    query = re.sub(r",\s*\)", r" )", query)

    # Fix double spaces
    query = re.sub(r" {2,}", " ", query)

    # Fix unbalanced parentheses
    open_count = query.count("(")
    close_count = query.count(")")
    if open_count > close_count:
        query = query + " )" * (open_count - close_count)

    return query


def generate_sql_queries(args, model, encoder_input, encoder_mask, initial_decoder_inputs, tokenizer):
    generation_config = build_generation_config(args, model)
    generated = model.generate(
        input_ids=encoder_input,
        attention_mask=encoder_mask,
        decoder_input_ids=initial_decoder_inputs,
        generation_config=generation_config,
    )
    return [clean_decoded_query(query) for query in tokenizer.batch_decode(generated, skip_special_tokens=True)]


def save_checkpoint(model, checkpoint_dir, tag):
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_dir = os.path.join(checkpoint_dir, tag)
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_dir)


def initialize_model():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    decoder_start_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    config = T5Config.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration(config)
    model.config.decoder_start_token_id = decoder_start_token_id
    model.generation_config.decoder_start_token_id = decoder_start_token_id
    return model.to(DEVICE)


def initialize_optimizer(model, learning_rate, weight_decay):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-8, betas=(0.9, 0.999))


def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    if args.scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if args.scheduler_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    raise ValueError(f"Unsupported scheduler_type: {args.scheduler_type}")


def train_epoch(args, model, train_loader, optimizer, scheduler):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    for encoder_input, encoder_mask, decoder_input, decoder_targets, _ in tqdm(train_loader, desc="train", leave=False):
        optimizer.zero_grad(set_to_none=True)
        encoder_input = encoder_input.to(DEVICE)
        encoder_mask = encoder_mask.to(DEVICE)
        decoder_input = decoder_input.to(DEVICE)
        decoder_targets = decoder_targets.to(DEVICE)

        logits = model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
        )["logits"]

        non_pad = decoder_targets != PAD_IDX
        loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_targets.reshape(-1))
        loss.backward()
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            num_tokens = int(torch.sum(non_pad).item())
            total_loss += float(loss.item()) * num_tokens
            total_tokens += num_tokens

    return total_loss / max(total_tokens, 1)


def eval_epoch(args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    generated_sql_queries = []
    tokenizer = dev_loader.dataset.tokenizer

    with torch.no_grad():
        for encoder_input, encoder_mask, decoder_input, decoder_targets, initial_decoder_inputs in tqdm(
            dev_loader, desc="dev", leave=False
        ):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            decoder_input = decoder_input.to(DEVICE)
            decoder_targets = decoder_targets.to(DEVICE)
            initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)

            logits = model(
                input_ids=encoder_input,
                attention_mask=encoder_mask,
                decoder_input_ids=decoder_input,
            )["logits"]

            loss = criterion(logits.reshape(-1, logits.size(-1)), decoder_targets.reshape(-1))
            non_pad = decoder_targets != PAD_IDX
            num_tokens = int(torch.sum(non_pad).item())
            total_loss += float(loss.item()) * num_tokens
            total_tokens += num_tokens

            generated_sql_queries.extend(
                generate_sql_queries(args, model, encoder_input, encoder_mask, initial_decoder_inputs, tokenizer)
            )

    save_queries_and_records(generated_sql_queries, model_sql_path, model_record_path)
    sql_em, record_em, record_f1, error_msgs = compute_metrics(
        gt_sql_path, model_sql_path, gt_record_path, model_record_path
    )
    error_rate = float(np.mean([bool(error_msg) for error_msg in error_msgs])) if error_msgs else 0.0
    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss, record_f1, record_em, sql_em, error_rate


def test_inference(args, model, test_loader, model_sql_path, model_record_path):
    model.eval()
    generated_sql_queries = []
    tokenizer = test_loader.dataset.tokenizer

    with torch.no_grad():
        for encoder_input, encoder_mask, initial_decoder_inputs in tqdm(test_loader, desc="test", leave=False):
            encoder_input = encoder_input.to(DEVICE)
            encoder_mask = encoder_mask.to(DEVICE)
            initial_decoder_inputs = initial_decoder_inputs.to(DEVICE)
            generated_sql_queries.extend(
                generate_sql_queries(args, model, encoder_input, encoder_mask, initial_decoder_inputs, tokenizer)
            )

    save_queries_and_records(generated_sql_queries, model_sql_path, model_record_path)


def train(args, model, train_loader, dev_loader, optimizer, scheduler):
    experiment_name = args.experiment_name
    model_type = "scr"
    checkpoint_dir = os.path.join("checkpoints", f"{model_type}_experiments", experiment_name)
    gt_sql_path = "data/dev.sql"
    gt_record_path = "records/ground_truth_dev.pkl"
    model_sql_path = os.path.join("results", f"t5_{model_type}_{experiment_name}_dev.sql")
    model_record_path = os.path.join("records", f"t5_{model_type}_{experiment_name}_dev.pkl")

    best_f1 = -1.0
    best_state_dict = None
    epochs_since_improvement = 0

    for epoch in range(args.max_n_epochs):
        train_loss = train_epoch(args, model, train_loader, optimizer, scheduler)
        print(f"Epoch {epoch}: Average train loss was {train_loss}")

        eval_loss, record_f1, record_em, sql_em, error_rate = eval_epoch(
            args, model, dev_loader, gt_sql_path, model_sql_path, gt_record_path, model_record_path
        )
        print(
            f"Epoch {epoch}: Dev loss: {eval_loss}, Record F1: {record_f1}, "
            f"Record EM: {record_em}, SQL EM: {sql_em}"
        )
        print(f"Epoch {epoch}: {error_rate*100:.2f}% of the generated outputs led to SQL errors")

        save_checkpoint(model, checkpoint_dir, "last")

        if record_f1 > best_f1:
            best_f1 = record_f1
            epochs_since_improvement = 0
            best_state_dict = {
                name: param.detach().cpu().clone()
                for name, param in model.state_dict().items()
            }
            save_checkpoint(model, checkpoint_dir, "best")
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= args.patience_epochs:
            break

    return best_state_dict


def copy_extra_credit_submission_files(args):
    experiment_name = args.experiment_name
    model_type = "scr"

    source_test_sql = os.path.join("results", f"t5_{model_type}_{experiment_name}_test.sql")
    source_test_records = os.path.join("records", f"t5_{model_type}_{experiment_name}_test.pkl")
    target_test_sql = os.path.join("results", "t5_ft_experiment_ec_test.sql")
    target_test_records = os.path.join("records", "t5_ft_experiment_ec_test.pkl")

    os.makedirs("results", exist_ok=True)
    os.makedirs("records", exist_ok=True)
    with open(source_test_sql, "r") as src, open(target_test_sql, "w") as dst:
        dst.write(src.read())
    with open(source_test_records, "rb") as src, open(target_test_records, "wb") as dst:
        dst.write(src.read())


def get_args():
    parser = argparse.ArgumentParser(description="Train T5-small from scratch for text-to-SQL.")

    parser.add_argument("--experiment_name", type=str, default="experiment_ec")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--scheduler_type", type=str, default="cosine", choices=["none", "cosine", "linear"])
    parser.add_argument("--num_warmup_epochs", type=int, default=3)
    parser.add_argument("--max_n_epochs", type=int, default=40)
    parser.add_argument("--patience_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=16)
    parser.add_argument("--max_generation_length", type=int, default=450)
    parser.add_argument("--num_beams", type=int, default=8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_submission_copy", action="store_true")

    return parser.parse_args()


def main():
    args = get_args()
    args.experiment_name = attach_slurm_job_id(args.experiment_name)
    set_random_seeds(args.seed)

    train_loader, dev_loader, test_loader = load_t5_data(args.batch_size, args.test_batch_size)
    model = initialize_model()
    optimizer = initialize_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = initialize_scheduler(args, optimizer, len(train_loader))

    best_state_dict = train(args, model, train_loader, dev_loader, optimizer, scheduler)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    model.eval()

    experiment_name = args.experiment_name
    model_type = "scr"
    gt_sql_path = "data/dev.sql"
    gt_record_path = "records/ground_truth_dev.pkl"
    dev_sql_path = os.path.join("results", f"t5_{model_type}_{experiment_name}_dev.sql")
    dev_record_path = os.path.join("records", f"t5_{model_type}_{experiment_name}_dev.pkl")

    dev_loss, dev_record_f1, dev_record_em, dev_sql_em, dev_error_rate = eval_epoch(
        args, model, dev_loader, gt_sql_path, dev_sql_path, gt_record_path, dev_record_path
    )
    print(
        f"Dev set results: Loss: {dev_loss}, Record F1: {dev_record_f1}, "
        f"Record EM: {dev_record_em}, SQL EM: {dev_sql_em}"
    )
    print(f"Dev set results: {dev_error_rate*100:.2f}% of the generated outputs led to SQL errors")

    test_sql_path = os.path.join("results", f"t5_{model_type}_{experiment_name}_test.sql")
    test_record_path = os.path.join("records", f"t5_{model_type}_{experiment_name}_test.pkl")
    test_inference(args, model, test_loader, test_sql_path, test_record_path)

    if not args.skip_submission_copy:
        copy_extra_credit_submission_files(args)
        print("Copied extra-credit submission files:")
        print("  results/t5_ft_experiment_ec_test.sql")
        print("  records/t5_ft_experiment_ec_test.pkl")

    print("\nIf you see loss improving while F1 plateaus, try:")
    print("- Changing the learning rate.")
    print("- Verifying BOS/EOS/special tokens are set correctly.")
    print("- Checking whether your decoding / sampling setup is appropriate.")


if __name__ == "__main__":
    main()
