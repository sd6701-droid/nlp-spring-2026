import argparse
import pickle
import re

from utils import read_queries


DEFAULT_EXPERIMENT = "t5_ft_experiment_ft_ep31_pat100_lr1e-4_wd0p0_bs32_tbs16_beam8_gen512_schedcosine_warm1_manual"


def get_args():
    parser = argparse.ArgumentParser(description="Analyze text-to-SQL dev errors for a saved run.")
    parser.add_argument(
        "--experiment_name",
        default=DEFAULT_EXPERIMENT,
        help="Run prefix used in results/ and records/ without the trailing _dev suffix.",
    )
    parser.add_argument(
        "--predicted_sql",
        default=None,
        help="Optional explicit path to the predicted dev SQL file.",
    )
    parser.add_argument(
        "--predicted_records",
        default=None,
        help="Optional explicit path to the predicted dev records .pkl file.",
    )
    return parser.parse_args()


def resolve_prediction_paths(args):
    predicted_sql = args.predicted_sql
    predicted_records = args.predicted_records

    if predicted_sql is None:
        predicted_sql = f"results/{args.experiment_name}_dev.sql"
    if predicted_records is None:
        predicted_records = f"records/{args.experiment_name}_dev.pkl"

    return predicted_sql, predicted_records


def per_example_f1(gt_set, pred_set):
    if len(pred_set) == 0:
        precision = 1.0
    else:
        precision = len(pred_set & gt_set) / len(pred_set)

    if len(gt_set) == 0:
        recall = 1.0
    else:
        recall = len(gt_set & pred_set) / len(gt_set)

    return 2 * precision * recall / (precision + recall + 1e-8)


def main():
    args = get_args()
    predicted_sql_path, predicted_records_path = resolve_prediction_paths(args)

    gt_nl = open("data/dev.nl").read().strip().split("\n")
    gt_sql = read_queries("data/dev.sql")
    pred_sql = read_queries(predicted_sql_path)

    with open("records/ground_truth_dev.pkl", "rb") as f:
        gt_records, _ = pickle.load(f)

    with open(predicted_records_path, "rb") as f:
        pred_records, pred_errors = pickle.load(f)

    syntax_errors = []
    wrong_results = []
    correct = []

    for i in range(len(gt_sql)):
        gt_set = set(gt_records[i])
        pred_set = set(pred_records[i])

        if pred_errors[i] != "":
            syntax_errors.append(i)
        elif gt_set == pred_set:
            correct.append(i)
        else:
            wrong_results.append(i)

    print(f"Using predicted SQL: {predicted_sql_path}")
    print(f"Using predicted records: {predicted_records_path}")
    print("=== SUMMARY ===")
    print(f"Total: {len(gt_sql)}")
    print(f"Correct: {len(correct)} ({len(correct)/len(gt_sql)*100:.1f}%)")
    print(f"Syntax errors: {len(syntax_errors)} ({len(syntax_errors)/len(gt_sql)*100:.1f}%)")
    print(f"Wrong results: {len(wrong_results)} ({len(wrong_results)/len(gt_sql)*100:.1f}%)")

    print("\n=== SYNTAX ERRORS (first 10) ===")
    for idx in syntax_errors[:10]:
        print(f"\n--- Example {idx} ---")
        print(f"NL:   {gt_nl[idx]}")
        print(f"GT:   {gt_sql[idx][:150]}")
        print(f"PRED: {pred_sql[idx][:150]}")
        print(f"ERR:  {pred_errors[idx]}")

    print("\n=== WRONG RESULTS (first 15) ===")
    for idx in wrong_results[:15]:
        gt_set = set(gt_records[idx])
        pred_set = set(pred_records[idx])
        f1 = per_example_f1(gt_set, pred_set)

        print(f"\n--- Example {idx} (F1={f1:.3f}) ---")
        print(f"NL:   {gt_nl[idx]}")
        print(f"GT:   {gt_sql[idx][:200]}")
        print(f"PRED: {pred_sql[idx][:200]}")
        print(f"GT records:   {len(gt_set)} rows")
        print(f"PRED records: {len(pred_set)} rows")

    print("\n=== ERROR PATTERN ANALYSIS ===")

    error_types = {}
    for idx in syntax_errors:
        err = pred_errors[idx]
        err_type = err.split(":")[0] if ":" in err else err
        error_types[err_type] = error_types.get(err_type, 0) + 1

    print("\nSyntax error types:")
    for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")

    print("\nWrong result patterns:")
    missing_keyword = 0
    wrong_table = 0
    wrong_value = 0

    for idx in wrong_results:
        gt = gt_sql[idx].upper()
        pred = pred_sql[idx].upper()

        gt_tables = set(w for w in gt.split() if w in ["FLIGHT", "FARE", "AIRPORT", "AIRLINE", "CITY"])
        pred_tables = set(w for w in pred.split() if w in ["FLIGHT", "FARE", "AIRPORT", "AIRLINE", "CITY"])

        if gt_tables != pred_tables:
            wrong_table += 1

        if ("BETWEEN" in gt) != ("BETWEEN" in pred):
            missing_keyword += 1

        gt_values = set(re.findall(r"'([^']*)'", gt))
        pred_values = set(re.findall(r"'([^']*)'", pred))
        if gt_values != pred_values:
            wrong_value += 1

    print(f"  Wrong/missing tables: {wrong_table}/{len(wrong_results)}")
    print(f"  Missing BETWEEN/range: {missing_keyword}/{len(wrong_results)}")
    print(f"  Wrong string values: {wrong_value}/{len(wrong_results)}")

    print("\n=== COMPLETE FAILURES (F1=0, first 10) ===")
    count = 0
    for idx in wrong_results:
        gt_set = set(gt_records[idx])
        pred_set = set(pred_records[idx])
        if len(gt_set & pred_set) == 0 and len(gt_set) > 0:
            print(f"\n--- Example {idx} ---")
            print(f"NL:   {gt_nl[idx]}")
            print(f"GT:   {gt_sql[idx][:250]}")
            print(f"PRED: {pred_sql[idx][:250]}")
            count += 1
            if count >= 10:
                break


if __name__ == "__main__":
    main()
