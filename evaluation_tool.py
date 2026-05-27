import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import argparse
import io
import os
from datetime import datetime
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def resolve_base_dir():
    script_dir = Path(__file__).resolve().parent
    candidates = [script_dir, script_dir.parent]
    for candidate in candidates:
        if (candidate / "Data").exists() and (candidate / "Master Data").exists():
            return candidate
    raise FileNotFoundError("Could not find 'Data' and 'Master Data' folders next to evaluation_tool.py or one level above.")


def load_data(base_dir):
    train = pd.read_csv(base_dir / "Data" / "Training.csv")
    test = pd.read_csv(base_dir / "Data" / "Testing.csv")
    train.columns = train.columns.str.replace(r"\.\d+$", "", regex=True)
    test.columns = test.columns.str.replace(r"\.\d+$", "", regex=True)
    train = train.loc[:, ~train.columns.duplicated()]
    test = test.loc[:, ~test.columns.duplicated()]
    return train, test


def prepare_xy(df):
    cols = df.columns[:-1]
    X = df[cols].values
    y = df['prognosis'].values
    return X, y, cols


def check_overlap(train_df, test_df):
    merged = test_df.merge(train_df.drop_duplicates(), how='left', indicator=True)
    found = (merged['_merge'] == 'both').sum()
    print(f"Testing rows: {len(test_df)}")
    print(f"Rows found in training exactly: {found}")
    return found


def check_diversity(train_df):
    total_rows = len(train_df)
    unique_rows = len(train_df.drop_duplicates())
    per_class = train_df.groupby('prognosis').apply(lambda df: len(df.drop_duplicates()))
    print(f"Total rows: {total_rows}")
    print(f"Total unique rows: {unique_rows}")
    sample = per_class.sort_values().head(10)
    print("Per-class unique feature-row counts (sample):")
    print(sample)
    return total_rows, unique_rows, per_class


def build_model(C=1.0):
    return LogisticRegression(max_iter=1000, C=C)


def print_summary_header(acc, prec, rec, f1):
    print("EVALUATION REPORT")
    print("=" * 50)
    print()
    print(f"Accuracy  : {acc*100:.2f}%")
    print(f"Precision : {prec*100:.2f}%")
    print(f"Recall    : {rec*100:.2f}%")
    print(f"F1-score  : {f1*100:.2f}%")
    print()
    print("-" * 50)
    print("DETAILED CLASSIFICATION REPORT")
    print("-" * 50)


def print_detailed_report(report_dict, label_order, support_counts):
    print(f"{'':41s}{'precision':>9s}{'recall':>9s}{'f1-score':>9s}{'support':>9s}")
    print()
    for label in label_order:
        metrics = report_dict.get(label, {})
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        f1 = metrics.get('f1-score', 0.0)
        support = support_counts.get(label, 0)
        label_text = f"{label:40.40s}"
        print(f"{label_text}{precision:9.2f}{recall:9.2f}{f1:9.2f}{support:9d}")


def evaluate(train_df, flip_rate=0.0, C=1.0):
    # dedupe training rows
    dedup = train_df.drop_duplicates()
    X_all, y_all, cols = prepare_xy(dedup)

    le = preprocessing.LabelEncoder()
    y_all_enc = le.fit_transform(y_all)

    # stratified split
    X_train, X_val, y_train_enc, y_val_enc = train_test_split(
        X_all, y_all_enc, test_size=0.25, random_state=42, stratify=y_all_enc
    )

    # perturb validation if requested (assumes binary features 0/1)
    if flip_rate > 0:
        rng = np.random.RandomState(42)
        mask = rng.rand(*X_val.shape) < flip_rate
        X_val = X_val.copy()
        X_val[mask] = 1 - X_val[mask]

    model = build_model(C=C)
    model.fit(X_train, y_train_enc)

    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val_enc, y_pred)
    prec = precision_score(y_val_enc, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_val_enc, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val_enc, y_pred, average='weighted', zero_division=0)

    print_summary_header(acc, prec, rec, f1)

    cls_report = classification_report(y_val_enc, y_pred, zero_division=0, output_dict=True)
    unique_labels = sorted(set(y_val_enc))
    label_names = [le.inverse_transform([lbl])[0] for lbl in unique_labels]

    support_counts = {le.inverse_transform([lbl])[0]: int((y_val_enc == lbl).sum()) for lbl in unique_labels}

    mapped_report = {}
    for k, v in cls_report.items():
        try:
            idx = int(k)
            name = le.inverse_transform([idx])[0]
            mapped_report[name] = v
        except Exception:
            pass

    print_detailed_report(mapped_report, label_names, support_counts)

    # overall averages (as percentages)
    print()
    total = sum(support_counts.values())
    m_prec = cls_report.get('macro avg', {}).get('precision', 0) * 100
    m_rec = cls_report.get('macro avg', {}).get('recall', 0) * 100
    m_f1 = cls_report.get('macro avg', {}).get('f1-score', 0) * 100
    w_prec = cls_report.get('weighted avg', {}).get('precision', 0) * 100
    w_rec = cls_report.get('weighted avg', {}).get('recall', 0) * 100
    w_f1 = cls_report.get('weighted avg', {}).get('f1-score', 0) * 100

    print(f"{'accuracy':>40s}{acc*100:19.2f}%{total:9d}")
    print(f"{'macro avg':>40s}{m_prec:9.2f}%{m_rec:9.2f}%{m_f1:9.2f}%{total:9d}")
    print(f"{'weighted avg':>40s}{w_prec:9.2f}%{w_rec:9.2f}%{w_f1:9.2f}%{total:9d}")


def main():
    parser = argparse.ArgumentParser(description='Evaluation tool: overlap, diversity, evaluation')
    parser.add_argument('--overlap', action='store_true')
    parser.add_argument('--diversity', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--flip-rate', type=float, default=0.0, help='Fraction of validation feature entries to flip (0-1).')
    parser.add_argument('--C', type=float, default=1.0, help='LogisticRegression regularization C')
    parser.add_argument('--save', action='store_true', help='Save the evaluation output to a timestamped TXT file')
    parser.add_argument('--output-dir', type=str, default='reports', help='Directory to write the timestamped report file')
    args = parser.parse_args()

    base = resolve_base_dir()
    train_df, test_df = load_data(base)

    # Optionally capture stdout and write to timestamped file
    if args.save:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        buf = io.StringIO()
        from contextlib import redirect_stdout
        with redirect_stdout(buf):
            if args.overlap:
                check_overlap(train_df, test_df)

            if args.diversity:
                check_diversity(train_df)

            if args.evaluate:
                evaluate(train_df, flip_rate=args.flip_rate, C=args.C)

        content = buf.getvalue()
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = out_dir / f"evaluation_{ts}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Saved evaluation output to: {filename}")
        return

    if args.overlap:
        check_overlap(train_df, test_df)

    if args.diversity:
        check_diversity(train_df)

    if args.evaluate:
        evaluate(train_df, flip_rate=args.flip_rate, C=args.C)


if __name__ == '__main__':
    main()
