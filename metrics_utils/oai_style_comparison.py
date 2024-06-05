import dotenv

from metrics_utils.oai_utils import predict

prompt1 = """
I'll give you the real message of some person in the interview and two fragments (A and B). 
Your task is to tell me, which fragment is closer to the original by style?

[real message]
{original}
[/real message]

[message A]
{message_a}
[/message A]

[message B]
{message_b}
[/message B]

Your answer should contain only one letter of the winner or sign '=' if both variants are nearly equal. And nothing else
Examples of the answer:
A

or 

B

or

=
""".strip()

_valid_answers = {'A', 'B', '='}


def _predict_safe(
        prompt: str, max_new_tokens: int | None = 16, answer_in_the_end: bool = False) -> str:
    answer = predict(prompt, max_new_tokens=max_new_tokens)
    if answer in _valid_answers:
        return answer
    # print(prompt)
    if answer:
        if answer_in_the_end:
            answer = answer.splitlines()[-1].strip()
            if answer.endswith('.'):
                answer = answer[:-1]
            if answer.lower().startswith("step") or answer.lower().startswith("therefore"):
                answer = answer[-1]
        a = answer[0].upper()
        if a in _valid_answers:
            # print(f"Answer is corrected from \"{answer}\"")
            return a
    print(f"Answer \"{answer}\" is not recognized, '=' is used instead")
    return '='


_opposites = {'A': 'B', 'B': 'A', '=': '='}


def compare_predictions_style(original: str, pred_a: str, pred_b: str) -> tuple[str, str]:
    p1 = prompt1.format(original=original, message_a=pred_a, message_b=pred_b)
    answer1 = _predict_safe(p1)
    p2 = prompt1.format(original=original, message_a=pred_b, message_b=pred_a)
    answer2 = _opposites[_predict_safe(p2)]
    return answer1, answer2


if __name__ == '__main__':
    dotenv.load_dotenv()
    # question = "Today we have Elon Musk. Elon, thank you for joining us."
    # print(_predict(question))
    orig = "Well, I think you make some estimates of, whatever this thing is that you're trying to create, what would be the utility delta compared to the current state of the art times how many people it would affect. So that's why I think having something that makes a big difference but affects sort of small to moderate number of people is great, as is something that makes even a small difference but affects a vast number of people. Like, the area under the curve."
    pred_a = "I think you should try to do something that you think is good and that you think is going to make the world a better place."
    pred_b = "I think you just try to find something that you’re good at and that you like doing. And then try to do more of that."

    print(compare_predictions_style(orig, pred_a, pred_b))
