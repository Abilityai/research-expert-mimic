import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import dotenv
import pandas as pd
from tqdm import tqdm
import openai as oai

from metrics_utils import remove_prediction_artifacts, compare_predictions_style
from metrics_utils.cleanup_utils import remove_repetitions


@dataclass
class RowToComparison:
    answer: str | None = None
    pred_1: str | None = None
    pred_2: str | None = None


def combine_predictions(df1: pd.DataFrame, df2: pd.DataFrame) -> list[RowToComparison]:
    assert len(df1) == len(df2), "DataFrames df1 and df2 should have the same length"

    res = []
    for (_, r1), (_, r2) in zip(df1.iterrows(), df2.iterrows()):
        assert r1.answer == r2.answer, ("Answers should be the same in the same order, "
                                        "because it should be the same dataset")
        res.append(RowToComparison(
            answer=r1.answer,
            pred_1=r1.pred,
            pred_2=r2.pred
        ))
    return res


def load_data_jsonl(ds1: str) -> pd.DataFrame:
    df = pd.read_json(ds1, lines=True)
    df.pred = df.pred.apply(remove_prediction_artifacts)
    df = df[['pred', 'answer']]
    return df


def store_stats(df: pd.DataFrame, root: Path, stat_fn_stem: str, ds1: str, ds2: str):
    df.to_csv(root / (stat_fn_stem+'.csv'), index=False)
    qa1 = Counter(pd.concat([df.q1_a1, df.q1_a2]))

    def _count_to_str(counter: Counter) -> str:
        return ", ".join([f"'{k}':{counter.get(k, 0)}" for k in 'A B ='.split()])

    def _winner(counter: Counter) -> str:
        return max(counter, key=counter.get)

    rep = dedent(f"""
        Predictions A: {ds1}
        Predictions B: {ds2}
        
        Style comparison:
          Winner: {_winner(qa1)}
          {_count_to_str(qa1)}
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
            q1_results = compare_predictions_style(r.answer, pred_1, pred_2)
            json_row = {
                "answer": r.answer,
                "pred_1": r.pred_1,
                "pred_2": r.pred_2,
                "q1_a1": q1_results[0],
                "q1_a2": q1_results[1],
            }
            results.append(json_row)
        except oai.APIError as ex:
            print(f"APIError {ex}")
            print(f"Row {r}")
            raise
    stat_fn = Path(ds1).stem + "_vs_" + Path(ds2).stem + "_style"
    root = Path(ds1).parent.absolute()
    df = pd.DataFrame(results)
    store_stats(df, root, stat_fn, ds1, ds2)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        prog="Metrics Calculator",
        description="Calculate metrics as a comparison for two passed datasets"
    )
    ap.add_argument("dataset1", help="Path to the first jsonl predictions dataset", type=str)
    ap.add_argument("dataset2", help="Path to the second jsonl predictions dataset", type=str)
    args = ap.parse_args()

    dotenv.load_dotenv()

    main(args.dataset1, args.dataset2)
