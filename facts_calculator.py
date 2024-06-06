import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from textwrap import dedent

import dotenv
import pandas as pd
from tqdm import tqdm
import openai as oai

from style_calculator import load_data_jsonl, combine_predictions
from metrics_utils import compare_predictions_facts_cot_json, remove_repetitions


def store_stats(df: pd.DataFrame, root: Path, stat_fn_stem: str, ds1: str, ds2: str):
    df.to_csv(root / (stat_fn_stem+'.csv'), index=False)

    def calc_metrics(
            tp_col: pd.Series, fp_col: pd.Series, fn_col: pd.Series) -> tuple[float, float, float]:
        tp = tp_col.sum()
        fp = fp_col.sum()
        fn = fn_col.sum()
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        return prec, rec, f1

    a_prec, a_rec, a_f1 = calc_metrics(df.tp_a, df.fp_a, df.fn_a)
    b_prec, b_rec, b_f1 = calc_metrics(df.tp_b, df.fp_b, df.fn_b)

    rep = dedent(f"""
        Predictions A: {ds1}
        Predictions B: {ds2}

        Facts comparison:
        Total facts: {df.total_facts.sum()}
        A: TP={df.tp_a.sum()}, FP={df.fp_a.sum()}, FN={df.fn_a.sum()}
        B: TP={df.tp_b.sum()}, FP={df.fp_b.sum()}, FN={df.fn_b.sum()}
        
        A: Precision={a_prec:.3f}, Recall={a_rec:.3f}, F1={a_f1:.3f}
        B: Precision={b_prec:.3f}, Recall={b_rec:.3f}, F1={b_f1:.3f}
    """)
    root.joinpath(stat_fn_stem + ".txt").write_text(rep, encoding='utf-8')
    print(f"Reports are stored in {root / (stat_fn_stem+'.*')}")


def main(ds1: str, ds2: str):
    df1 = load_data_jsonl(ds1)
    df2 = load_data_jsonl(ds2)
    rows = combine_predictions(df1, df2)
    results = []
    for r in tqdm(rows, desc="Comparing predictions"):
        try:
            pred_1 = remove_repetitions(r.pred_1)
            pred_2 = remove_repetitions(r.pred_2)
            f_results = compare_predictions_facts_cot_json(r.answer, pred_1, pred_2)
            json_row = {
                "answer": r.answer,
                "pred_1": r.pred_1,
                "pred_2": r.pred_2,
                "raw_res": json.dumps(
                    asdict(f_results),
                    indent=2,
                    ensure_ascii=False,
                ),
                **asdict(f_results.ab_metrics),
            }
            results.append(json_row)
        except oai.APIError as ex:
            print(f"APIError {ex}")
            print(f"Row {r}")
            raise
    stat_fn = Path(ds1).stem + "_vs_" + Path(ds2).stem + "_facts"
    root = Path(ds1).parent.absolute()
    df = pd.DataFrame(results)
    store_stats(df, root, stat_fn, ds1, ds2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        prog="Fact-Metrics Calculator",
        description="Calculate fact-metrics as a comparison for two passed datasets"
    )
    ap.add_argument("dataset1", help="Path to the first jsonl predictions dataset", type=str)
    ap.add_argument("dataset2", help="Path to the second jsonl predictions dataset", type=str)
    args = ap.parse_args()

    dotenv.load_dotenv()

    main(args.dataset1, args.dataset2)
