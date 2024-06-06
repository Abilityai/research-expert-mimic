import argparse
import os
from dataclasses import dataclass, field

import dotenv
from huggingface_hub.hf_api import HfFolder
from transformers import TrainingArguments, AutoTokenizer
from trl import SFTTrainer
from unsloth import FastLanguageModel, get_chat_template
import torch
from datasets import concatenate_datasets
from aenum import StrEnum

from data_parsers.dataset_utils import PartsEnum, PersonsEnum, read_dataset_messages

dotenv.load_dotenv()


class Fraction(StrEnum):
    OneHalf = '0.5'
    Full = '1.0'
    AndAHalf = '1.5'
    Double = '2.0'


@dataclass
class TrainPlan:
    experiment_suffix: str = ''
    fraction: Fraction | str = Fraction.Full


train_plans: list[TrainPlan] = [
    TrainPlan('f_0.5', fraction=Fraction.OneHalf),
    TrainPlan('f_1.0', fraction=Fraction.Full),
    TrainPlan('f_1.5', fraction=Fraction.AndAHalf),
    TrainPlan('f_2.0', fraction=Fraction.Double),
]

r = 32
MAX_LEN = 8192
# base model from huggingFace or path to model
base_model = "mistralai/Mistral-7B-Instruct-v0.3"
new_model_stem = base_model.split('/')[-1] + f"_musk_stex_"


def get_model_and_tokenizer()->tuple[FastLanguageModel, AutoTokenizer]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_LEN,
        dtype=None,
        load_in_4bit=True
    )
    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=1974,
    )

    # Set cache to False
    model.config.use_cache = False
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "mistral", # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth, llama-3
        # mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
    )
    return model, tokenizer


def build_dataset(train_plan: TrainPlan):
    datasets = []
    if train_plan.fraction == Fraction.OneHalf or train_plan.fraction == Fraction.AndAHalf:
        ds = read_dataset_messages(
            PersonsEnum.ElonMusk, PartsEnum.train, dialog_length=5, randomize_length=True)
        ds = ds[::2]
        datasets.append(ds)
    if (train_plan.fraction == Fraction.Full or train_plan.fraction == Fraction.AndAHalf or
            train_plan.fraction == Fraction.Double):
        ds = read_dataset_messages(
            PersonsEnum.ElonMusk, PartsEnum.train, dialog_length=5, randomize_length=True)
        datasets.append(ds)
    if train_plan.fraction == Fraction.Double:
        ds = read_dataset_messages(
            PersonsEnum.ElonMusk, PartsEnum.train, dialog_length=5, randomize_length=True)
        datasets.append(ds)

    return concatenate_datasets(datasets)


def construct_trainer(model, tokenizer, dataset)->SFTTrainer:

    # Config Trainer
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        # dataset_text_field = "messages",
        max_seq_length = MAX_LEN,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # max_steps = 60,
            num_train_epochs=1,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    return trainer


def perform_train(train_plan: TrainPlan):
    new_model = new_model_stem + train_plan.experiment_suffix
    print(f'Training model {new_model}')
    model, tokenizer = get_model_and_tokenizer()
    dataset = build_dataset(train_plan)
    print(f'Dataset size: {len(dataset)}')
    trainer = construct_trainer(model, tokenizer, dataset)
    trainer.train()
    # push the model and tokenizer
    print(f'Pushing model and tokenizer {new_model} to HuggingFace')
    trainer.model.save_pretrained(new_model)
    trainer.model.push_to_hub(new_model)
    tokenizer.push_to_hub(new_model, private = False)
    print(f'Model {new_model} is pushed to HuggingFace')
    


def main(experiment_num: int):
    secret_hf = os.environ.get('HUGGINGFACE_TOKEN')
    if secret_hf is None:
        raise ValueError('HuggingFace token is not set!')
    HfFolder.save_token(secret_hf)

    if experiment_num >= len(train_plans):
        raise ValueError(
            f'Experiment number {experiment_num} is out of range [-1..{len(train_plans) - 1}]')
    if experiment_num < -1:
        raise ValueError(
            f'Experiment number {experiment_num} is out of range [-1..{len(train_plans) - 1}]')
    experiments = [train_plans[experiment_num]] if experiment_num >= 0 else train_plans
    for train_plan in experiments:
        perform_train(train_plan)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script is used to fine-tune the model on the Musk dataset (style mimic experiment)')
    parser.add_argument(
        'experiment_num', type=int, nargs='?', default=-1,
        help='experiment number 0..3, default - train all the models')
    args = parser.parse_args()
    main(args.experiment_num)
