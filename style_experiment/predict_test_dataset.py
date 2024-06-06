from pathlib import Path

from transformers import pipeline
from datasets import Dataset
from unsloth import FastLanguageModel
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from data_parsers.dataset_utils import read_dataset_messages, PartsEnum, PersonsEnum

load_dotenv()

MAX_LEN = 2048

base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
model_names = [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "atepeq/Mistral-7B-Instruct-v0.3_musk_stex_f_0.5",
    "atepeq/Mistral-7B-Instruct-v0.3_musk_stex_f_1.0",
    "atepeq/Mistral-7B-Instruct-v0.3_musk_stex_f_1.5",
    "atepeq/Mistral-7B-Instruct-v0.3_musk_stex_f_2.0",
]

def make_predictions(model_name: str, test_dataset: Dataset) -> list[str]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_LEN,
        dtype=None,
        load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=2048,
        model_kwargs=dict(
            temperature=0.01,  # Adjust temperature
            # top_p=0.9,  # Use nucleus sampling
            do_sample=True,  # Enable sampling
            repetition_penalty=1.5,  # Penalize repetition
            early_stopping=True  # Enable early stopping
        )
    )

    preds = []
    for row in tqdm(test_dataset):
        prompt = row['messages']
        pred = pipe(prompt, return_full_text=False, temperature=0.001)
        preds.append(pred[0]['generated_text'])

    return preds


def main():
    test_dataset = read_dataset_messages(PersonsEnum.ElonMusk, PartsEnum.test)
    # For quick testing
    # test_dataset = Dataset.from_dict(test_dataset[:10])
    print(test_dataset)
    predictions = {}
    reports_root = Path(__file__).parent / "data" / "musk"
    reports_root.mkdir(parents=True, exist_ok=True)

    for model_name in tqdm(model_names, desc="Predicting per model"):
        preds = make_predictions(model_name, test_dataset)
        model_suffix = model_name.split("stex_")[-1] if "stex_" in model_name else "base"
        predictions[f'pred'] = preds
        results_df = pd.DataFrame(predictions)
        results_df['answer'] = test_dataset['answer']
        results_df["messages"] = test_dataset["messages"]
        results_df.to_json(reports_root / f"predictions_stex_{model_suffix}.jsonl",
                           orient='records', lines=True, force_ascii=False)


if __name__ == '__main__':
    main()
