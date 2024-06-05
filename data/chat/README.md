Here we place the parsed data in a structured form (JSON)

## File structure

```json
{
  "src": "url_of_the_original_source",
  "speakers": [
    {
      "name": "Full Name",
      "id": "interviewer"
    },
    {
      "name": "Full Name",
      "id": "interviewee"
    }
  ],
  "messages": [
    {
      "role": "interviewer",
      "text": "The text of the interviewer"
    },
    {
      "role": "interviewee",
      "text": "The text of the interviewee"
    },
    ...
  ]
}
```
