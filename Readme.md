## Endpoints
- localhost:<port>/api/CosineSimilarity/
  > Endpoint to train a model with BERT (deepset/sentence_bert) and then finding cosine similarity

- localhost:<port>/api/BERTClassification/
  > Text classification using BERT pretrained (cross-encoder/qnli-electra-base) model

## Input request format
- Method : /POST/
- Body :
```json
{
  "sentences" : {
       "<sentence>":"<label>"
  },
  "class_list" : [["<list of phrases in label1>"]],
   "threshold" : "<float>"
}
```

- localhost:<port>/api/evaluation/
  > getting evaluation metrics in format for the steamlit app for revenue and interview questions

## Input request format
- Method : /POST/
- Body :
```json
{
    "sentence":"<>",
    "label":"<>",
    "models":["BCE","BERTCosine","rasa"],
    "question":" 'q1'/'q2' "
}
```

- localhost:<port>/api/q1processing/
  > getting evaluation metrics in format for the steamlit app for revenue and interview questions

## Input request format
- Method : /POST/
- Body :
```json
{
  "sentences" : ["user persona", "action to be performed","goals"]
}
```
