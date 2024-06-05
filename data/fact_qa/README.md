# Content description
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

