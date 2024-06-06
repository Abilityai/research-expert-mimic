import json
from dataclasses import dataclass, field
from aenum import StrEnum
from pathlib import Path
import random

from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from data_parsers.parse_utils import ChatMessage, chat_root


class PartsEnum(StrEnum):
    train = 'train'
    test = 'test'
    old = 'old'
    new = 'new'


class PersonsEnum(StrEnum):
    ElonMusk = 'musk'
    DonaldTrump = 'trump'
    WarrenBuffet = 'buffet'


@dataclass
class PersonDsDescription:
    """Description of a dataset for a person"""
    person: PersonsEnum
    """Person id"""
    person_ids: list[str] = field(default_factory=list)
    """List of possible person ids in the dataset"""


@dataclass
class Dialogue:
    """Prepared dialogue for training/testing"""
    text: str
    """Templated dialogue text"""
    answer: str
    """Real answer"""
    messages: list[dict[str, str]] = field(default_factory=list)
    """Conversational format - https://huggingface.co/docs/trl/en/sft_trainer#dataset-format-support"""


_descriptions: dict[PersonsEnum, PersonDsDescription] = {
    PersonsEnum.ElonMusk: PersonDsDescription(
        PersonsEnum.ElonMusk, person_ids=['elon', 'elon_musk']
    ),
}

_ds_root = chat_root


def get_chat_data_root() -> Path:
    return Path(_ds_root)


def get_files_for_part(ds_path: Path, part: PartsEnum) -> list[str]:
    files = [str(f) for f in ds_path.glob('*.json') if f.is_file()]
    if part == PartsEnum.train:
        files = [f for f in files if not Path(f).name.startswith('_')]
    elif part == PartsEnum.test:
        files = [f for f in files if Path(f).name.startswith('_')]
    elif part == PartsEnum.old:
        files = [f for f in files if '2023' not in Path(f).name]
    elif part == PartsEnum.new:
        files = [f for f in files if '2023' in Path(f).name]
    else:
        raise ValueError(f'Unknown part: {part}')
    return files


def correct_roles(messages: list[ChatMessage], person_ids: list[str]) -> list[ChatMessage]:
    """
    Corrects roles in the messages. If role is not in person_ids, it is assumed as user role. Otherwise, it is assumed
    as assistant role. Then, the messages with the same role are merged (with '\n' between).

    :param messages: list of messages
    :param person_ids: list of possible person ids
    :return: corrected messages list
    """
    person_ids_set = set(person_ids)
    corrected_messages = [
        ChatMessage('assistant' if m.role in person_ids_set else 'user', m.text) for m in messages
    ]
    merged_messages = []
    for m in corrected_messages:
        if merged_messages and merged_messages[-1].role == m.role:
            merged_messages[-1].text += '\n' + m.text
        else:
            merged_messages.append(m)
    assert any(m.role == 'assistant' for m in merged_messages), 'No assistant messages found'
    assert any(m.role == 'user' for m in merged_messages), 'No user messages found'
    return merged_messages


def load_subdialogues(
        f: str, person_ids: list[str], tokenizer: PreTrainedTokenizerFast,
        add_answer_to_text: bool,
        dialog_length: int = 2
) -> list[Dialogue]:
    """
    Load dataset from json file and prepares short dialogues for training/testing

    :param f: filename of the dataset
    :param person_ids: possible person ids in the dataset. These ids assumed to be assistant role. All other ids are
        assumed to be user role.
    :param tokenizer: tokenizer that will be used to templates the chat (we use
        https://huggingface.co/docs/transformers/main/chat_templating for preparing the dataset)
    :param add_answer_to_text: If True, answer is added to the text as an assistant message
    :param dialog_length: Number of previous sub-dialogues to include in the training data (pair assistant-user
        messages assumed as a sub-dialogue)
    :return: list of prepared dialogues
    """
    person_ids_set = set(person_ids)
    js = json.load(open(f, 'rt', encoding='utf-8'))
    records = []
    messages: list[ChatMessage] = [ChatMessage(**m) for m in js['messages']]
    messages = correct_roles(messages, person_ids)
    if messages[
        0].role == 'assistant':  # if the first message is assistant, add a pseudo user message
        messages.insert(0, ChatMessage('user', '...inaudible...'))
    for n in range(len(messages)):
        m = messages[n]
        if m.role == 'user':
            continue
        messages_to_template = messages[max(0,
                                            n - dialog_length * 2 + 1):n + 1 if add_answer_to_text else n]
        chat = [{'role': m.role, 'content': m.text} for m in messages_to_template]
        templated_chat = tokenizer.apply_chat_template(chat, tokenize=False)
        records.append(Dialogue(templated_chat, m.text, messages=chat))
    return records


def read_dataset(
        person: PersonsEnum | str, part: PartsEnum | str, tokenizer: PreTrainedTokenizerFast,
        dialog_length: int = 2, randomize_length: bool = False,
) -> Dataset:
    """
    Combines prepared dialogues from different files to a single dataset

    :param person: A person to get the dataset
    :param part: A part of the dataset (train or test)
    :param tokenizer: Model tokenizer
    :param dialog_length: How many previous sub-dialogues to include in the training data
    :param randomize_length: If True, make dialog length from 1 to dialog_length in random.
    :return: Dataset. Contains 'text' field for training and 'text', 'answer' fields for testing. Text is a templated
        dialogue, answer is a real answer.
    """
    description = _descriptions[PersonsEnum(person)]
    ds_path = _ds_root / person
    files = get_files_for_part(ds_path, part)
    dfs = []
    for f in files:
        try:
            dfs.append(load_subdialogues(f, description.person_ids, tokenizer,
                                         add_answer_to_text=part == PartsEnum.train,
                                         dialog_length=dialog_length))
        except Exception as e:
            print(f'Error in file {f}: {e}')
            raise e

    if randomize_length:
        for df in dfs:
            for d in df:
                full_subdialogs = len(d.messages) // 2
                if full_subdialogs < 2:
                    continue
                skipped_messages = random.randint(0, full_subdialogs - 1) * 2
                d.messages = d.messages[skipped_messages:]
    if part == PartsEnum.train:
        rows = [{'text': d.text, 'messages': d.messages} for df in dfs for d in df]
    else:
        rows = [{'text': d.text, 'answer': d.answer, 'messages': d.messages} for df in dfs for d in
                df]

    res = Dataset.from_list(rows)
    return res


def load_subdialogues_messages(
        f: str, person_ids: list[str], add_answer_to_text: bool,
        dialog_length: int = 2
) -> list[Dialogue]:
    """
    Load dataset from json file and prepares short dialogues for training/testing/old/new

    :param f: filename of the dataset
    :param person_ids: possible person ids in the dataset. These ids assumed to be assistant role.
        All other ids are assumed to be user role.
    :param add_answer_to_text: If True, answer is added to the text as an assistant message
    :param dialog_length: Number of previous sub-dialogues to include in the training data (pair
        assistant-user messages assumed as a sub-dialogue)
    :return: list of prepared dialogues
    """
    js = json.load(open(f, 'rt', encoding='utf-8'))
    records = []
    messages: list[ChatMessage] = [ChatMessage(**m) for m in js['messages']]
    messages = correct_roles(messages, person_ids)
    if messages[
        0].role == 'assistant':  # if the first message is assistant, add a pseudo user message
        messages.insert(0, ChatMessage('user', '...inaudible...'))
    for n in range(len(messages)):
        m = messages[n]
        if m.role == 'user':
            continue
        messages_to_template = messages[max(0,
                                            n - dialog_length * 2 + 1):n + 1 if add_answer_to_text else n]
        chat = [{'role': m.role, 'content': m.text} for m in messages_to_template]
        records.append(Dialogue('', m.text, messages=chat))
    return records


def read_dataset_messages(
        person: PersonsEnum | str, part: PartsEnum | str,
        dialog_length: int = 2, randomize_length: bool = False,
) -> Dataset:
    """Function similar to read_dataset, but returns only messages column, so we don't need in
    tokenizer.

    :param person: A person to get the dataset
    :param part: A part of the dataset (train or test)
    :param dialog_length: How many previous sub-dialogues to include in the training data
    :param randomize_length: If True, make dialog length from 1 to dialog_length in random.

    :return: Dataset. Contains 'messages' field for training/new/old and 'messages', 'answer'
        fields for testing.
    """
    description = _descriptions[PersonsEnum(person)]
    ds_path = _ds_root / person
    files = get_files_for_part(ds_path, part)
    dfs = []
    for f in files:
        try:
            dfs.append(
                load_subdialogues_messages(
                    f, description.person_ids,
                    add_answer_to_text=part != PartsEnum.test,
                    dialog_length=dialog_length)
            )
        except Exception as e:
            print(f'Error in file {f}: {e}')
            raise e

    if randomize_length:
        for df in dfs:
            for d in df:
                full_subdialogs = len(d.messages) // 2
                if full_subdialogs < 2:
                    continue
                skipped_messages = random.randint(0, full_subdialogs - 1) * 2
                d.messages = d.messages[skipped_messages:]
    if part == PartsEnum.test:
        rows = [{'messages': d.messages, 'answer': d.answer} for df in dfs for d in df]
    else:
        rows = [{'messages': d.messages} for df in dfs for d in df]

    res = Dataset.from_list(rows)
    return res
