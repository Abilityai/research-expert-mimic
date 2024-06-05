import re

assistant_prefix = 'assistant\n\n'


def remove_prediction_artifacts(pred: str) -> str:
    if pred.startswith(assistant_prefix):
        pred = pred[len(assistant_prefix):]
    return pred


def is_entailed_with(many_tokens: list[str], tail: list[str]) -> bool:
    """Check if many_tokens ends with tail."""
    if len(many_tokens) < len(tail):
        return False
    for i in range(len(tail)):
        if many_tokens[-1 - i] != tail[-1 - i]:
            return False
    return True


def remove_repetitions(text):
    tokens = re.findall(r'\S+\s*', text)
    last_token = tokens.pop()  # remove the last token - it can be partially formed from subtokens

    end = len(tokens)
    found_repetition = False

    for seq_length in range(2, len(tokens) // 4 + 1):
        tested_sequence = tokens[-seq_length:]
        rest = tokens[:-seq_length]
        repetitions = 0
        while is_entailed_with(rest, tested_sequence):
            repetitions += 1
            rest = rest[:-seq_length]

        if repetitions >= 4:
            end = len(rest)
            found_repetition = True
            break

    if found_repetition:
        result = ''.join(tokens[:end] + [last_token])
    else:
        result = text
    return result

# pred_2 = (
#     'I mean, I sleep a lot. I mean, I sleep a lot. I mean, I think, on average, I think I get '
#     'about six hours of sleep a night, and that’s a lot of sleep. And I mean, I think that’s a '
#     'lot of sleep. I mean, I think most people would sleep a lot more than I do. I mean, '
#     'I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, '
#     'I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, I mean, '
#     'I mean, I mean, I mean, I mean, I mean, I mean, '
#
# cleaned_pred_2 = remove_repetitions(pred_2)
# print(cleaned_pred_2)
