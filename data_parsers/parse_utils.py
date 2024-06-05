from dataclasses import dataclass
from pathlib import Path

data_root = Path(__file__).parent.parent / 'data'
raw_root = data_root / 'raw'
chat_root = data_root / 'chat'


@dataclass
class ChatMessage:
    role: str = ""
    text: str = ""
