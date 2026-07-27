"""Audit split and external-data leakage for residue-level experiments."""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parent


def load_dataset(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def count_prediction_rows(dataset):
    return sum(len(item[1]) for item in dataset.values())


def labels_for_frame(dataset, frame):
    offsets = {protein_id: 0 for protein_id in dataset}
    labels = []
    for protein_id in frame["protein_id"].astype(str):
        index = offsets[protein_id]
        labels.append(int(dataset[protein_id][1][index]))
        offsets[protein_id] += 1
    return labels


def exact_sequence_overlaps(left, right):
    left_by_sequence = {}
    for protein_id, item in left.items():
        left_by_sequence.setdefault(str(item[0]), []).append(protein_id)
    overlaps = []
    for protein_id, item in right.items():
        sequence = str(item[0])
        if sequence in left_by_sequence:
            overlaps.append({"right_id": protein_id, "left_ids": left_by_sequence[sequence]})
    return overlaps


def audit_prediction_dir(prediction_dir, train_data, test_data):
    prediction_dir = prediction_dir.resolve()
    result = {"prediction_dir": str(prediction_dir)}
    valid_path = prediction_dir / "validation_predictions.csv"
    if valid_path.exists():
        valid = pd.read_csv(valid_path)
        valid_ids = set(valid["protein_id"].astype(str))
        result["validation_rows"] = int(len(valid))
        result["validation_proteins"] = int(len(valid_ids))
        result["validation_subset_of_train335"] = bool(valid_ids <= set(train_data))
    test_path = prediction_dir / "test_predictions.csv"
    if not test_path.exists():
        test_path = prediction_dir / "predictions.csv"
    if test_path.exists():
        test = pd.read_csv(test_path)
        test_ids = set(test["protein_id"].astype(str))
        expected_labels = labels_for_frame(test_data, test)
        observed_labels = test["label"].astype(int).tolist() if "label" in test.columns else []
        result["test_rows"] = int(len(test))
        result["expected_test_rows"] = int(count_prediction_rows(test_data))
        result["test_proteins"] = int(len(test_ids))
        result["test_ids_match_test60"] = bool(test_ids == set(test_data))
        result["test_row_count_match_test60"] = bool(len(test) == count_prediction_rows(test_data))
        result["test_labels_match_test60"] = bool(observed_labels == expected_labels)
    return result


def audit_external_dir(external_dir, train_data, test_data):
    external_dir = external_dir.resolve()
    dataset_path = external_dir / "External_SKEMPI2_filtered.pkl"
    result = {"external_dir": str(external_dir), "dataset_exists": dataset_path.exists()}
    if not dataset_path.exists():
        return result
    external = load_dataset(dataset_path)
    reference = {**train_data, **test_data}
    overlaps = exact_sequence_overlaps(reference, external)
    result["external_records"] = int(len(external))
    result["external_exact_reference_duplicate_count"] = int(len(overlaps))
    result["external_exact_reference_duplicates"] = overlaps[:20]
    report_path = external_dir / "preparation_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        result["reported_homology_rule"] = report.get("homology_rule")
        result["reported_homology_or_duplicate_rejections"] = report.get("homology_or_duplicate_rejections")
        result["reported_accepted_chain_records"] = report.get("accepted_chain_records")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--external-dirs", type=Path, nargs="*", default=[])
    parser.add_argument("--train-dataset", type=Path, default=SOURCE_DIR / "Dataset/Train_335.pkl")
    parser.add_argument("--test-dataset", type=Path, default=SOURCE_DIR / "Dataset/Test_60.pkl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_data = load_dataset(args.train_dataset.resolve())
    train_data.pop("2j3rA", None)
    test_data = load_dataset(args.test_dataset.resolve())
    report = {
        "train335_proteins": len(train_data),
        "test60_proteins": len(test_data),
        "train_test_id_overlap": sorted(set(train_data) & set(test_data)),
        "train_test_exact_sequence_duplicates": exact_sequence_overlaps(train_data, test_data),
        "prediction_audits": [audit_prediction_dir(path, train_data, test_data) for path in args.prediction_dirs],
        "external_audits": [audit_external_dir(path, train_data, test_data) for path in args.external_dirs],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
