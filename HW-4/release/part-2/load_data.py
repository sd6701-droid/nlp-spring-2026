import os
import re

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import T5TokenizerFast
import torch

PAD_IDX = 0
MODEL_NAME = "google-t5/t5-small"
TASK_PREFIX = "translate English to SQL: "
MAX_SQL_TOKENS = 400
SCHEMA_HINT = (
    "tables: flight, airport_service, city, airline, "
    "fare, flight_fare, flight_stop, ground_service, "
    "airport, days, date_day, fare_basis, restriction"
)

# In load_data.py - expand these fully
CITY_MAP = {
    # Abbreviations
    "los angeles": "LOS ANGELES", "la": "LOS ANGELES",
    "new york": "NEW YORK", "nyc": "NEW YORK", "new york city": "NEW YORK",
    "san francisco": "SAN FRANCISCO", "sf": "SAN FRANCISCO",
    "washington dc": "WASHINGTON", "dc": "WASHINGTON",
    "washington": "WASHINGTON",
    "fort worth": "FORT WORTH",
    "salt lake city": "SALT LAKE CITY",
    "kansas city": "KANSAS CITY",
    # ADD all cities from your flight database
    "boston": "BOSTON", "denver": "DENVER", "dallas": "DALLAS",
    "pittsburgh": "PITTSBURGH", "philadelphia": "PHILADELPHIA",
    "atlanta": "ATLANTA", "chicago": "CHICAGO", "houston": "HOUSTON",
    "baltimore": "BALTIMORE", "miami": "MIAMI", "seattle": "SEATTLE",
    "minneapolis": "MINNEAPOLIS", "detroit": "DETROIT",
    "charlotte": "CHARLOTTE", "cleveland": "CLEVELAND",
    "indianapolis": "INDIANAPOLIS", "orlando": "ORLANDO",
    "milwaukee": "MILWAUKEE", "toronto": "TORONTO",
    "newark": "NEWARK", "nashville": "NASHVILLE",
    "memphis": "MEMPHIS", "st. louis": "ST. LOUIS", "saint louis": "ST. LOUIS",
    "columbus": "COLUMBUS", "cincinnati": "CINCINNATI",
    "san diego": "SAN DIEGO", "phoenix": "PHOENIX",
    "las vegas": "LAS VEGAS", "oakland": "OAKLAND",
    "ontario": "ONTARIO", "long beach": "LONG BEACH",
    "burbank": "BURBANK", "tacoma": "TACOMA",
}

AIRLINE_MAP = {
    "american airlines": "AA", "american": "AA",
    "united airlines": "UA", "united": "UA",
    "us air": "US", "usair": "US",
    "delta": "DL", "delta airlines": "DL",
    "midwest express": "YX",
    "continental": "CO", "continental airlines": "CO",
    "southwest": "WN", "southwest airlines": "WN",
    "alaska": "AS", "alaska airlines": "AS",
    "northwest": "NW", "northwest airlines": "NW",
    "twa": "TW", "trans world": "TW",
    "lufthansa": "LH",
    "tower air": "FF",
}

def has_phrase(text, phrase):
    return re.search(rf"\b{re.escape(phrase)}\b", text) is not None


def normalize_nl(nl_query):
    query_lower = nl_query.lower()
    hints = []

    # Existing city/airline hints...
    for city_name, canonical_name in CITY_MAP.items():
        if has_phrase(query_lower, city_name):
            hint = f"[city={canonical_name}]"
            if hint not in hints:
                hints.append(hint)

    for airline_name, airline_code in AIRLINE_MAP.items():
        if has_phrase(query_lower, airline_name):
            hint = f"[airline={airline_code}]"
            if hint not in hints:
                hints.append(hint)

    # ADD: Aggregation hints
    if any(w in query_lower for w in ["earliest", "first", "minimum", "lowest", "cheapest", "least"]):
        hints.append("[agg=MIN]")
    if any(w in query_lower for w in ["latest", "last", "maximum", "highest", "most", "largest"]):
        hints.append("[agg=MAX]")

    # ADD: SELECT column hints
    if any(w in query_lower for w in ["flight time", "departure time", "arrival time", "what time"]):
        hints.append("[select=departure_time]")
    if any(w in query_lower for w in ["cost", "price", "fare", "how much"]):
        hints.append("[select=fare_id]")
    if any(w in query_lower for w in ["transport", "ground transportation", "train", "bus", "rental"]):
        hints.append("[select=transport_type]")

    if not hints:
        return nl_query
    return f"{nl_query} {' '.join(hints)}"


def build_encoder_input(nl_query):
    normalized_nl = normalize_nl(nl_query)
    return f"{TASK_PREFIX}{normalized_nl} | {SCHEMA_HINT}"

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
        self.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")
        self.examples = self.process_data(data_folder, split, self.tokenizer)

def normalize_sql(sql_query):
    """Normalize SQL targets before training."""
    sql_query = re.sub(r'\bAND\s+1\s*=\s*1\b', '', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'\bWHERE\s+1\s*=\s*1\s+AND\b', 'WHERE', sql_query, flags=re.IGNORECASE)
    sql_query = re.sub(r'\s+', ' ', sql_query).strip()
    return sql_query

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_queries = load_lines(nl_path)

        sql_queries = None
        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_queries = load_lines(sql_path)

        examples = []
        if split == "test":
            for nl_query in nl_queries:
                encoder_ids = tokenizer(
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
            sql_query = normalize_sql(sql_query)  
            sql_ids = tokenizer(
                sql_query,
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
            if split == "train" and len(sql_ids) > MAX_SQL_TOKENS:
                continue

            encoder_ids = tokenizer(
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
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
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
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
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
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    train_x = [build_encoder_input(line) for line in load_lines(os.path.join(data_folder, "train.nl"))]
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = [build_encoder_input(line) for line in load_lines(os.path.join(data_folder, "dev.nl"))]
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = [build_encoder_input(line) for line in load_lines(os.path.join(data_folder, "test.nl"))]
    return train_x, train_y, dev_x, dev_y, test_x
