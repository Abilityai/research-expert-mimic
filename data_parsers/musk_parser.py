import json
from dataclasses import asdict
from pathlib import Path
import re

from bs4 import BeautifulSoup
from tqdm import tqdm

from data_parsers.parse_utils import raw_root, chat_root, ChatMessage


def parse_recore(src_path: Path, dest_path: Path):
    # Read and parse HTML content
    content = src_path.read_text(encoding='utf-8')
    soup = BeautifulSoup(content, 'html.parser')

    # Hardcode speaker IDs and names
    speakers = {
        "kara_swisher": "Kara Swisher",
        "elon_musk": "Elon Musk"
    }

    # Extract messages
    messages = []
    current_text = []

    for element in soup.find_all(['p', 'h3']):
        if element.name == 'h3':
            continue  # Skip headers

        if element.find('strong') and element.text.strip() == element.strong.text.strip():
            # Message from Kara Swisher
            speaker_id = "kara_swisher"
            message = element.get_text(strip=True)
        else:
            # Message from Elon Musk
            speaker_id = "elon_musk"
            message = element.get_text(strip=True)

        # Append the current message to the list
        if message:
            messages.append(ChatMessage(role=speaker_id, text=message))

    # Create JSON data
    data = {
        "src": "Unknown",  # Source URL is not present
        "speakers": [{"id": id, "name": name} for id, name in speakers.items()],
        "messages": [asdict(message) for message in messages]
    }

    # Write to JSON file
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, 'wt', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)


def parse_bbc(src_path: Path, dest_path: Path):
    source_url, data = _retrieve_url_lines(src_path)

    # extract speakers and messages
    speakers = {}
    messages: list[ChatMessage] = []
    speaker_id = "unknown"
    message: ChatMessage | None = None
    for line in data:
        # https://regex101.com/r/6nTOVL/1
        if m := re.match(r"^([\w][\w\d\s]+[\w\d])\s+\(.+\):$", line):
            speaker_name = m.group(1)
            speaker_id = speaker_name.replace(" ", "_").lower()
            if speaker_id not in speakers:
                speakers[speaker_id] = speaker_name
            if message:
                messages.append(message)
            message = None
        else:
            # skip "(dd:dd:dd):" timestamps
            if re.match(r"^\(\d+:\d+:\d+\):", line):
                continue
            if message:
                message.text += '\n' + line
            else:
                message = ChatMessage(role=speaker_id, text=line)
    if message:
        messages.append(message)

    # Extract speaker names and messages

    # Create JSON data
    _store_as_json(dest_path, source_url, speakers, messages)


def parse_emi(src_path: Path, dest_path: Path):
    source_url, data = _retrieve_url_lines(src_path)

    # extract speakers and messages
    speakers = {}
    for line in data:
        if re.match(r'^\w{2,15}(?:\s\w{2,15}(?:\s\w{2,15})?)?:', line):
            speaker_name = line.split(":")[0]
            speaker_id = speaker_name.replace(" ", "_").lower()
            if speaker_id not in speakers:
                speakers[speaker_id] = speaker_name

    speaker2id = {name: id for id, name in speakers.items()}
    messages: list[ChatMessage] = []
    speaker_id = ""
    message: ChatMessage | None = None
    for line in data:
        if line.endswith(':') and (speaker := line[:-1]) in speaker2id:
            if message:
                messages.append(message)
                message = None
            speaker_id = speaker2id[speaker]
            continue
        if not speaker_id:  # skip lines before the first speaker
            continue
        if message:
            message.text += '\n' + line
        else:
            message = ChatMessage(role=speaker_id, text=line)
    if message:
        messages.append(message)

    # Store JSON data
    _store_as_json(dest_path, source_url, speakers, messages)


def parse_fridman(src_path: Path, dest_path: Path):
    source_url, data = _retrieve_url_lines(src_path)

    # extract speakers and messages
    speakers = {'lex_fridman': 'Lex Fridman', 'elon_musk': 'Elon Musk'}
    speaker2id = {name: id for id, name in speakers.items()}
    messages: list[ChatMessage] = []
    speaker_id = "unknown"
    message: ChatMessage | None = None
    for line in data:
        if line in speaker2id:
            if message:
                messages.append(message)
                message = None
            speaker_id = speaker2id[line]
            continue
        if m := re.match(r"^\(\d+:\d+:\d+\)\s+(.*)$", line, re.IGNORECASE | re.UNICODE):
            msg_text = m.group(1)
            if message:
                message.text += '\n' + msg_text
            else:
                message = ChatMessage(role=speaker_id, text=msg_text)
    if message:
        messages.append(message)

    # Store JSON data
    _store_as_json(dest_path, source_url, speakers, messages)


def parse_scribd(src_path: Path, dest_path: Path):
    source_url, data = _retrieve_url_lines(src_path)

    # extract speakers and messages
    speakers = {'interviewer': 'Interviewer', 'elon': 'Elon'}
    speaker2id = {name: id for id, name in speakers.items()}
    messages: list[ChatMessage] = []
    for line in data:
        speaker_id = line.split(":", 1)[0].lower()
        if speaker_id not in speakers:
            raise ValueError(f"Unknown speaker ID: {speaker_id}")
        message = ChatMessage(role=speaker_id, text=line.split(":", 1)[1].strip())
        messages.append(message)

    # Store JSON data
    _store_as_json(dest_path, source_url, speakers, messages)


def _store_as_json(
        dest_path: Path, source_url: str, speakers: dict[str, str], messages: list[ChatMessage]):
    for message in messages:
        message.text = remove_artefact_patterns(message.text)
    data = {
        "src": source_url,
        "speakers": [{"id": speaker_id, "name": speaker_name}
                     for speaker_id, speaker_name in speakers.items()],
        "messages": [asdict(message) for message in messages]
    }
    # Write to JSON file
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, 'wt', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=2, ensure_ascii=False)


def _retrieve_url_lines(src_path: Path) -> tuple[str, list[str]]:
    lines = src_path.read_text(encoding='utf-8').splitlines()
    # Extract source URL
    source_url = 'Unknown'
    if lines[0].startswith("src: "):
        source_url = lines[0].replace("src: ", "").strip()
    # Find separator index
    separator_index = lines.index("---")
    data = [line.strip() for line in lines[separator_index + 1:]
            if line.strip() and not line.startswith("#")]
    return source_url, data


def remove_artefact_patterns(text: str) -> str:
    """Removes unwanted patterns from the text."""
    # List of patterns to remove
    patterns = [
        r"\(\d{2}:\d{2}:\d{2}\)\n",  # Timestamps with newline
        r"\(\d{2}:\d{2}\)\n",  # Timestamps with newline
        r"\[inaudible \d{2}:\d{2}:\d{2}\][\s]?"  # Inaudible annotations
    ]

    # Combine all patterns into one big pattern with the 'or' operator (|)
    combined_pattern = "|".join(patterns)

    # Substitute the matched patterns with an empty string
    cleaned_text = re.sub(combined_pattern, "", text)

    return cleaned_text


def print_stats():
    total = 0
    chars = 0
    for chat in (chat_root / 'musk').glob('*.json'):
        data = json.loads(chat.read_text(encoding='utf-8'))
        ds_chars = sum(len(msg['text']) for msg in data['messages'])
        print(f"{chat.name}: {len(data['messages'])} messages, chars: {ds_chars}")
        total += len(data['messages'])
        chars += ds_chars
    print(f"Total: {total} messages, {chars} characters")


def main():
    print("Parsing Elon Musk interviews...")

    # src = raw_root / 'musk' / 'elon-musk-interview-with-the-bbc-4-11-2023-transcript.txt'
    # dest = chat_root / 'musk' / src.name.replace(".txt", ".json")
    # parse_bbc(src_path=src, dest_path=dest)
    # print(f"Processed {src}")
    #
    # src = raw_root / 'musk' / '_elon-musk-kara-swisher-decode-podcast.txt'
    # dest = chat_root / 'musk' / src.name.replace(".txt", ".json")
    # parse_recore(src_path=src, dest_path=dest)
    # print(f"Processed {src}")
    #
    # src = raw_root / 'musk' / 'dealbook-summit-2023-elon-musk-interview.txt'
    # dest = chat_root / 'musk' / src.name.replace(".txt", ".json")
    # parse_bbc(src_path=src, dest_path=dest)
    # print(f"Processed {src}")
    #
    # src = raw_root / 'musk' / 'elon-musk-interview-with-don-lemon.txt'
    # dest = chat_root / 'musk' / src.name.replace(".txt", ".json")
    # parse_bbc(src_path=src, dest_path=dest)
    # print(f"Processed {src}")
    #
    # src = raw_root / 'musk' / 'elon-musk-lex-fridman.txt'
    # dest = chat_root / 'musk' / src.name.replace(".txt", ".json")
    # parse_fridman(src_path=src, dest_path=dest)
    # print(f"Processed {src}")
    #
    # src = raw_root / 'musk' / '_elon-how-to-build-future.txt'
    # dest = chat_root / 'musk' / src.name.replace(".txt", ".json")
    # parse_scribd(src_path=src, dest_path=dest)
    # print(f"Processed {src}")

    for emi_file in tqdm(list((raw_root / 'musk').glob('emi-*.txt')), desc="Processing EMI files"):
        dest = chat_root / 'musk' / emi_file.name.replace(".txt", ".json")
        parse_emi(src_path=emi_file, dest_path=dest)
        print(f"Processed {emi_file}")

    print_stats()


if __name__ == '__main__':
    main()
