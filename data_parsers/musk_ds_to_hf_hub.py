# This script saves musk's interviews as a huggingface dataset
import os

from datasets import Dataset, NamedSplit
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

from data_parsers.dataset_utils import PartsEnum, PersonsEnum, read_dataset, get_chat_data_root


def main():
    tokenizer = AutoTokenizer.from_pretrained('unsloth/mistral-7b-instruct-v0.2-bnb-4bit')
    tokenizer.pad_token = tokenizer.unk_token

    train = read_dataset(PersonsEnum.ElonMusk, PartsEnum.train, tokenizer, dialog_length=5, randomize_length=True)
    train = train.remove_columns(['text'])
    train = train.add_column('id', list(range(len(train))))
    train = train.add_column('answer', [m[-1]['content'] for m in train['messages']])
    # train.split = 'train'
    print(train)

    test = read_dataset(PersonsEnum.ElonMusk, PartsEnum.test, tokenizer, dialog_length=5, randomize_length=True)
    test = test.remove_columns(['text'])
    test = test.add_column('id', list(range(len(test))))
    # test.split = 'test'
    print(test)

    ds_train = Dataset.from_list(train, split=NamedSplit('train'))
    print(ds_train)
    ds_test = Dataset.from_list(test, split=NamedSplit('test'))
    print(ds_test)

    ds_name = 'musk_interviews'
    ds_train.push_to_hub(ds_name)
    ds_test.push_to_hub(ds_name)


if __name__ == '__main__':
    load_dotenv()
    login(os.getenv('HUGGINGFACE_TOKEN'))
    main()
