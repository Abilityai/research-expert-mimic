import json
from functools import cache

import openai as oa

gpt_model = "gpt-3.5-turbo-0125"


@cache
def get_openai_client():
    return oa.OpenAI()


def predict(message: str, max_new_tokens: int = 128) -> str:
    messages = [
        {'role': 'user', 'content': message},
    ]
    completion = get_openai_client().chat.completions.create(
        model=gpt_model,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.0
    )
    return completion.choices[0].message.content


def predict_json(message: str) -> dict:
    messages = [
        {'role': 'user', 'content': message},
    ]
    completion = get_openai_client().chat.completions.create(
        model=gpt_model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0
    )
    json_str = completion.choices[0].message.content
    return json.loads(json_str)
