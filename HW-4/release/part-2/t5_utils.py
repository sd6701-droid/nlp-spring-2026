import os

import torch

import transformers
from transformers import T5ForConditionalGeneration, T5Config, T5TokenizerFast
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

try:
    import wandb
except ImportError:
    wandb = None

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')
MODEL_NAME = "google-t5/t5-small"

def setup_wandb(args):
    # Implement this if you wish to use wandb in your experiments
    if wandb is None:
        raise ImportError("wandb is not installed. Install it or run without --use_wandb.")
    wandb.init(
        project="csci2590-hw4-part2",
        name=args.experiment_name,
        config=vars(args),
    )

def initialize_model(args):
    '''
    Helper function to initialize the model. You should be either finetuning
    the pretrained model associated with the 'google-t5/t5-small' checkpoint
    or training a T5 model initialized with the 'google-t5/t5-small' config
    from scratch.
    '''
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    decoder_start_token_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")

    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    else:
        config = T5Config.from_pretrained(MODEL_NAME)
        model = T5ForConditionalGeneration(config)

    model.config.decoder_start_token_id = decoder_start_token_id
    model.generation_config.decoder_start_token_id = decoder_start_token_id
    return model.to(DEVICE)

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    # Save model checkpoint to be able to load the model later
    mkdir(checkpoint_dir)
    save_dir = os.path.join(checkpoint_dir, 'best' if best else 'last')
    mkdir(save_dir)

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_dir)

def load_model_from_checkpoint(args, best):
    # Load model from a checkpoint
    model_type = 'ft' if args.finetune else 'scr'
    checkpoint_dir = os.path.join('checkpoints', f'{model_type}_experiments', args.experiment_name)
    load_dir = os.path.join(checkpoint_dir, 'best' if best else 'last')

    if not os.path.isdir(load_dir):
        raise FileNotFoundError(f"Checkpoint directory does not exist: {load_dir}")

    model = T5ForConditionalGeneration.from_pretrained(load_dir)
    return model.to(DEVICE)

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    if args.optimizer_type == "AdamW":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8, betas=(0.9, 0.999)
        )
    else:
        raise NotImplementedError(f"Unsupported optimizer type: {args.optimizer_type}")

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result
