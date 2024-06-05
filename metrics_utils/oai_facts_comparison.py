from dataclasses import dataclass, field, asdict, fields
from pprint import pprint
from typing import TypedDict

import dotenv

from metrics_utils.oai_utils import predict_json

prompt2_cot_json = """
I will provide you with the original message from an interview and two altered fragments (A and B). Your task is to review the fragments, find the mentioned facts and make a report about each fragment.

[real fragment]
{original}
[/real fragment]

[fragment A]
{message_a}
[/fragment A]

[fragment B]
{message_b}
[/fragment B]

Follow this structured approach to assess each fragment:

1. List all factual information from the real fragment.
2. Identify which facts from the real fragment are present in fragment A and enumerate them.
3. List any hallucinated facts (facts not present in the original) in fragment A. If there are no hallucinated facts, return an empty array.
4. Identify which facts from the real fragment are present in fragment B and enumerate them.
5. List any hallucinated facts (facts not present in the original) in fragment B. If there are no hallucinated facts, return an empty array.

Expected JSON response:

{{
  "step1_original_facts": [ # Numerate facts starting from 0
    "0. Fact from the original message",
    "1. Another fact from the original message",
    "2. Additional fact from the original message",
    "3. Further fact from the original message"
  ],
  "step2_fragment_a_facts": [0, 1], # Place here the integer numbers of facts present in fragment A only
  "step3_fragment_a_hallucinated": [], # List hallucinated facts from fragment A, or return an empty array if none
  "step4_fragment_b_facts": [0, 2, 3], # Place here the integer numbers of facts present in fragment B only
  "step5_fragment_b_hallucinated": [] # List hallucinated facts from fragment B, or return an empty array if none
}}
It is only examples of the format of the answer. You should provide the correct JSON response with the actual facts and hallucinated facts using passed fragments.
""".strip()

ResultDict = TypedDict('ResultDict', {
    "step1_original_facts": list[str],
    "step2_fragment_a_facts": list[int],
    "step3_fragment_a_hallucinated": list[str],
    "step4_fragment_b_facts": list[int],
    "step5_fragment_b_hallucinated": list[str]
})


def swap_results_fragments(results: ResultDict) -> ResultDict:
    return {
        "step1_original_facts": results["step1_original_facts"],
        "step2_fragment_a_facts": results["step4_fragment_b_facts"],
        "step3_fragment_a_hallucinated": results["step5_fragment_b_hallucinated"],
        "step4_fragment_b_facts": results["step2_fragment_a_facts"],
        "step5_fragment_b_hallucinated": results["step3_fragment_a_hallucinated"]
    }


@dataclass
class ABMetrics:
    total_facts: int = 0
    tp_a: int = 0
    fp_a: int = 0
    fn_a: int = 0
    tp_b: int = 0
    fp_b: int = 0
    fn_b: int = 0

    def fill_metrics(self, answer: ResultDict):
        self.total_facts = len(answer["step1_original_facts"])
        self.tp_a = min(self.total_facts, len(answer["step2_fragment_a_facts"]))
        self.fp_a = len(answer["step3_fragment_a_hallucinated"])
        self.fn_a = self.total_facts - self.tp_a
        self.tp_b = min(self.total_facts, len(answer["step4_fragment_b_facts"]))
        self.fp_b = len(answer["step5_fragment_b_hallucinated"])
        self.fn_b = self.total_facts - self.tp_b

    @classmethod
    def fields_names(cls):
        return [fld.name for fld in fields(cls)]


@dataclass
class SingleComparisonResult:
    answer: ResultDict
    ab_metrics: ABMetrics = field(default_factory=ABMetrics)

    def fill_metrics(self):
        self.ab_metrics.fill_metrics(self.answer)


@dataclass
class ComparisonResult:
    """Combined comparison result for two fragments."""
    answers: list[SingleComparisonResult] = field(default_factory=list)
    """List of two comparison results. The first - with fragment A as the first fragment, the 
    second - with fragment B as the first fragment. The second result is stored with the swapped
    fragments, so the metrics are calculated correctly."""
    ab_metrics: ABMetrics = field(default_factory=ABMetrics)
    """Combined metrics for both fragments."""

    def fill_metrics(self):
        for a in self.answers:
            a.fill_metrics()
        for f_name in ABMetrics.fields_names():
            setattr(
                self.ab_metrics,
                f_name,
                sum(getattr(a.ab_metrics, f_name) for a in self.answers)
            )


def compare_predictions_facts_cot_json(
        original: str, pred_a: str, pred_b: str) -> ComparisonResult:
    p1 = prompt2_cot_json.format(original=original, message_a=pred_a, message_b=pred_b)
    answer1 = predict_json(p1)
    p2 = prompt2_cot_json.format(original=original, message_a=pred_b, message_b=pred_a)
    answer2 = predict_json(p2)
    answer2 = swap_results_fragments(answer2)
    res = ComparisonResult([
        SingleComparisonResult(answer1),
        SingleComparisonResult(answer2)
    ])
    res.fill_metrics()
    return res


if __name__ == '__main__':
    dotenv.load_dotenv()
    # orig = "Well, I think you make some estimates of, whatever this thing is that you're trying to create, what would be the utility delta compared to the current state of the art times how many people it would affect. So that's why I think having something that makes a big difference but affects sort of small to moderate number of people is great, as is something that makes even a small difference but affects a vast number of people. Like, the area under the curve."
    # pred_a = "I think you should try to do something that you think is good and that you think is going to make the world a better place."
    # pred_b = "I think you just try to find something that you’re good at and that you like doing. And then try to do more of that."

    # orig = "Well, oddly enough, I actually think they're pretty good."
    # pred_a = "I think it’s probably about 10%."
    # pred_b = "I think the probability is above 10%. I think it’s probably, I would say, roughly 20% or more."

    # orig = "No. Of course not. Count how many negative articles there are and how many I respond to. One percent, maybe. But the common rebuttal of journalists is, “Oh. My article’s fine. He’s just thin-skinned.” No, your article is false and you don’t want to admit it."
    # pred_a = "I don’t think I’m particularly sensitive. I think I’m just not a fan of falsehoods."
    # pred_b = "I don’t think I’m particularly sensitive. I think I’m pretty thick-skinned."

    orig = """It's like, people shouldn't think, well, "I feel fear about this and therefore I shouldn't do it," it's normal to feel fear. Like, you'd have to have something mentally wrong with you if you don't feel fear."""
    pred_a = """I think what you're getting at is that I'm not perfect, and I don't have a special immunity to fear. But I do have a strong sense of purpose and values that guide my decision-making. When I'm faced with a situation where others are telling me something is crazy, I try to take a step back and assess the situation objectively.

I ask myself questions like: "Is this idea aligned with my values and goals?" "Is there a logical reason why this could work?" "Have I considered all the potential risks and consequences?" "Is there a way to test or validate this idea before committing to it?"

I also try to surround myself with people who are willing to challenge my thinking and provide constructive feedback. I believe that having a diverse group of people who can offer different perspectives is essential in making informed decisions.

But ultimately, I think the key to making decisions in the face of opposition is to be confident in your own judgment and to be willing to take calculated risks. It's not about being fearless, but about being willing to take a chance on something that you believe in.

As for the internal strength to do this, I think it comes from a combination of factors. For me, it's rooted in my upbringing and the values that my parents instilled in me. They encouraged me to be curious, to question authority, and to think for myself. They also taught me to be resilient and to persevere in the face of adversity.

Additionally, I think that having a strong sense of purpose and meaning helps to give me the courage to take risks and challenge the status quo. When I'm working on something that I'm passionate about, I feel a sense of fulfillment and satisfaction that motivates me to push through the challenges and doubts that others may have.

So, to answer your question, I don't think I have a special formula for making decisions in the face of opposition. I just try to stay true to myself, surround myself with good people, and be willing to take calculated risks."""
    pred_b = "Yeah."

    # print(compare_predictions_facts_cot(orig, pred_a, pred_b))
    pprint(
        asdict(compare_predictions_facts_cot_json(orig, pred_a, pred_b)),
        width=320, compact=True
    )
