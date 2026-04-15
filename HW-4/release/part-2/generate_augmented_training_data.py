import argparse
import random
import re
from pathlib import Path


START_RULES = [
    (r"^okay can you tell me\b", ["can you show me", "show me"]),
    (r"^now can you tell me\b", ["can you show me", "show me"]),
    (r"^okay\b", ["please", "show me"]),
    (r"^now\b", ["please", "show me"]),
    (r"^please show me\b", ["show me"]),
    (r"^show me\b", ["list", "give me"]),
    (r"^list all\b", ["show all", "give me all"]),
    (r"^list\b", ["show", "give me"]),
    (r"^give me\b", ["show me", "list"]),
    (r"^display\b", ["show", "list"]),
    (r"^i would like\b", ["show me", "give me"]),
    (r"^i'd like\b", ["show me", "give me"]),
    (r"^what flights\b", ["show flights", "list flights"]),
    (r"^what are the flights\b", ["show the flights", "list the flights"]),
    (r"^what are\b", ["show", "list"]),
    (r"^what is\b", ["show", "tell me"]),
    (r"^can you tell me\b", ["show me"]),
]

CONTENT_RULES = [
    (r"\bleaving\b", ["departing"]),
    (r"\bcoming back\b", ["returning"]),
    (r"\bground transportation\b", ["ground transport"]),
    (r"\bairfare\b", ["fare"]),
    (r"\bcost\b", ["price", "fare"]),
    (r"\barrive at\b", ["land at"]),
    (r"\barriving in\b", ["landing in"]),
    (r"\barriving\b", ["landing"]),
    (r"\bstopping in\b", ["with a stop in"]),
    (r"\bshow me\b", ["please show me"]),
]


def load_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def apply_rule(text: str, rules: list[tuple[str, list[str]]], rng: random.Random) -> tuple[str, bool]:
    for pattern, replacements in rules:
        if re.search(pattern, text, flags=re.IGNORECASE):
            replacement = rng.choice(replacements)
            updated = re.sub(pattern, replacement, text, count=1, flags=re.IGNORECASE)
            if updated != text:
                return normalize_spaces(updated), True
    return text, False


def fallback_wrap(text: str, rng: random.Random) -> str:
    lower = text.lower()
    if lower.startswith(("what ", "which ", "when ", "how ")):
        return f"can you tell me {text}"
    if lower.startswith(("show ", "list ", "give ", "display ", "find ")):
        return f"please {text}"
    if lower.startswith(("flights ", "flight ", "airfare ", "fare ", "ground ")):
        return f"show me {text}"
    return rng.choice([f"please {text}", f"show me {text}", f"can you show me {text}"])


def augment_nl(text: str, seed: int) -> str:
    rng = random.Random(seed)
    augmented = normalize_spaces(text)

    augmented, changed = apply_rule(augmented, START_RULES, rng)

    if rng.random() < 0.7:
        augmented, content_changed = apply_rule(augmented, CONTENT_RULES, rng)
        changed = changed or content_changed

    if not changed:
        augmented = normalize_spaces(fallback_wrap(augmented, rng))

    if augmented == text:
        augmented = normalize_spaces(f"please {text}")

    return augmented


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rule-based NL augmentations for text-to-SQL training.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--augmented_prefix", type=str, default="train_augmented")
    parser.add_argument("--combined_prefix", type=str, default="train_plus_augmented")
    args = parser.parse_args()

    train_nl_path = args.data_dir / "train.nl"
    train_sql_path = args.data_dir / "train.sql"

    train_nl = load_lines(train_nl_path)
    train_sql = load_lines(train_sql_path)

    if len(train_nl) != len(train_sql):
        raise ValueError("train.nl and train.sql must have the same number of lines")

    augmented_nl = [augment_nl(text, args.seed + idx) for idx, text in enumerate(train_nl)]
    augmented_sql = list(train_sql)

    combined_nl = train_nl + augmented_nl
    combined_sql = train_sql + augmented_sql

    augmented_nl_path = args.data_dir / f"{args.augmented_prefix}.nl"
    augmented_sql_path = args.data_dir / f"{args.augmented_prefix}.sql"
    combined_nl_path = args.data_dir / f"{args.combined_prefix}.nl"
    combined_sql_path = args.data_dir / f"{args.combined_prefix}.sql"

    write_lines(augmented_nl_path, augmented_nl)
    write_lines(augmented_sql_path, augmented_sql)
    write_lines(combined_nl_path, combined_nl)
    write_lines(combined_sql_path, combined_sql)

    num_changed = sum(a != b for a, b in zip(train_nl, augmented_nl))
    print(f"Wrote {len(augmented_nl)} augmented examples to {augmented_nl_path.name} / {augmented_sql_path.name}")
    print(f"Wrote {len(combined_nl)} combined examples to {combined_nl_path.name} / {combined_sql_path.name}")
    print(f"Changed {num_changed}/{len(train_nl)} NL queries during augmentation")


if __name__ == "__main__":
    main()
