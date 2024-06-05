import argparse
import os
from dataclasses import dataclass, field

import dotenv
from huggingface_hub.hf_api import HfFolder

from data_parsers.dataset_utils import PartsEnum, PersonsEnum, read_dataset_messages

dotenv.load_dotenv()


@dataclass
class TrainPlan:
    experiment_suffix: str = ''
    parts: list[PartsEnum] = field(default_factory=list)
    epochs: int = 2


train_plans: list[TrainPlan] = [
    TrainPlan('old', parts=[PartsEnum.old]),
    # TrainPlan('new', parts=[PartsEnum.new]),
    TrainPlan('old_new', parts=[PartsEnum.old, PartsEnum.new]),
    TrainPlan('new_old', parts=[PartsEnum.new, PartsEnum.old]),
]

r = 32
MAX_LEN = 8192
# base model from huggingFace or path to model
base_model = "mistralai/Mistral-7B-Instruct-v0.3"
new_model_stem = base_model.split('/')[-1] + f"_musk_fmex_"


def perform_train(train_plan: TrainPlan):
    new_model = new_model_stem + train_plan.experiment_suffix
    print(f'Training model {new_model}')
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
        description='this script is used to fine-tune the model on the Musk dataset')
    parser.add_argument(
        'experiment_num', type=int,
        help='experiment number 0..3, default - train all the models', default=-1)
    args = parser.parse_args()
    main(args.experiment_num)
