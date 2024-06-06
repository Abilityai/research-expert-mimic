# Content description
## Creation from the interview
Here we will place database for checking the facts memoization. 
For preparing the dataset, we use the most fresh interviews, GPT-4o,
and the following prompt:

---
I will provide you with an interview text. Your task is to extract facts (no more than 15, but fewer if there isn't enough information for quality facts) from it and create questions to test if the facts were remembered. Preferably, use the facts mentioned by the interviewee. The response should be presented in the following JSON structure:

```json
[
  {
    "fact": "brief description of the fact here",
    "src": "original excerpt where this fact is mentioned",
    "question": "question phrased as if addressing the interviewee, to determine if the respondent remembers the original fact"
  },
  ...
]
```

The facts should not be general knowledge but specific to what the interviewee said. It is important that if the interviewee is a well-known person, many facts about them might also be general knowledge.

The question should not lead to a binary "yes/no" answer, which can be guessed, but should imply an open or numerical answer. Make sure to phrase the question as if addressing the interviewee directly (e.g., "How many hours do you usually sleep?" instead of "How many hours does Elon Musk usually sleep?").

---
If the interview is long, we can ask to get more facts with:

> Give me up to 15 more other facts in the same format and with the same conditions

## Adding answers as a direct speech
To add answers as a direct speech, we can use the following prompt (+ drop the .json into 
the chat):

---
I will give you a JSON. Add another field called "answer" to the fields fact/src/question. It should mimic an answer to the question from the first-person perspective (direct speech), using only the fact mentioned in the fact field, without adding additional facts that may be in the original phrase. Also, do not include mentions of the interviewee's name, which are artifacts of the interview transcription. Strive to ensure that the answer fully addresses the question - in the original interview, there was a previous phrase from the interviewer, so simply inserting the original phrase from the interviewee may be inconsistent. Our goal is to extract individual facts from the interview, create questions for them (this has already been done), and generate an answer that should only address this question, preferably in the style of the interviewee.

Examples (with the added answer field):

```json
{
    "fact": "Elon Musk values the philosophy of curiosity.",
    "src": "Elon Musk (29:14): If I were to describe my philosophy, it is a philosophy of curiosity.",
    "question": "How would you describe your personal philosophy?",
    "answer": "If I were to describe my philosophy, it is a philosophy of curiosity."
},

{
    "fact": "Musk thinks AI will surpass human intelligence in less than three years.",
    "src": "Elon Musk (53:22): I would say that we're less than three years from that point.",
    "question": "How soon do you believe AI will surpass human intelligence?",
    "answer": "I would say that AI will surpass human intelligence in less than three years from that point."
},
```
