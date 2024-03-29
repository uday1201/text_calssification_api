from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.nn import functional as F
import numpy as np
tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
model = AutoModel.from_pretrained('deepset/sentence_bert')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from transformers import pipeline
classifier = pipeline("text-classification", model = "cross-encoder/qnli-electra-base")
import torch

# it takes input as a json of sentences and a list of labels as [<list of keywords for label 1>, <>, ..]
# it ouputs a list of encoding objects -> {
#             "sentence": sentence,
#             "encodings": {
#                 "label": label,
#                 "sentence_rep":sentence_rep,
#                 "label_reps":label_reps
#             }}

def BERTencoding(sentences,labels):
    sentence_net = []
    for sentence, labelled_class in sentences.items():
        encodings =[]
        for label in labels:
            # run inputs through model and mean-pool over the sequence
            # dimension to get sequence-level representations
            inputs = tokenizer.batch_encode_plus([sentence] + label,
                                                 return_tensors='pt',
                                                 pad_to_max_length=True)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            output = model(input_ids, attention_mask=attention_mask)[0]
            sentence_rep = output[:1].mean(dim=1)
            label_reps = output[1:].mean(dim=1)

            encodings.append({
                "label": label,
                "sentence_rep":sentence_rep,
                "label_reps":label_reps
            })

        #binding to the output
        sentence_net.append({
            "sentence": sentence,
            "encodings": encodings
            })
    return sentence_net

# takes input the json of the output of BERTencoding list of objects and threshold -> cutoff similarity for a class
# Outputs the list of predictions
def cosineSimilarity(sentence_net, threshold):
    predictions =[]
    details = []
    for sentence in sentence_net:
        print("----------Sentence similarity for : ",sentence["sentence"])
        labels_similarity = []
        for encoding in sentence["encodings"]:
            label = encoding["label"]
            print("--- Class labels : ", label)
            # now find the labels with the highest cosine similarities to
            # the sentence
            similarities = F.cosine_similarity(encoding["sentence_rep"], encoding["label_reps"])
            closest = similarities.argsort(descending=True)
            similarity_list = []
            for ind in closest:
                print(f'label: {label[ind]} \t similarity: {similarities[ind]}')
                similarity_list.append(float(similarities[ind]))

            mean_cosine = np.mean(similarity_list)
            max_cosine = max(similarity_list)
            cosine_class_mean = float(F.cosine_similarity(encoding["sentence_rep"], torch.mean(encoding['label_reps'])))
            # getting the mean similarity for each label
            labels_similarity.append(mean_cosine+max_cosine+cosine_class_mean)
            print("\n")

        details.append(labels_similarity)
        # getting the predicted class
        print(labels_similarity)
        predicted_labels = [1 if (y>threshold and y==max(labels_similarity))else 0 for y in labels_similarity]
        if 1 in predicted_labels:
            predictions.append(predicted_labels.index(max(predicted_labels)))
        else:
            predictions.append(-1)


    return details,predictions


def BERTCosinePrediction(sentence, labels, threshold):
    details = []
    sentence_embed = tokenizer.encode(sentence,return_tensors='pt', padding='max_length', max_length=100)
    labels_metric =[]
    for label in labels:
        label_embed = []
        cosine_with_each_label = []
        for l in label:
            l_embed = tokenizer.encode(l,return_tensors='pt', padding='max_length', max_length=100)
            label_embed.append(l_embed)
            cosine_with_each_label.append(F.cosine_similarity(sentence_embed.float(), l_embed.float()))

        max_cosine = max(cosine_with_each_label)
        avg_cosine = sum(cosine_with_each_label)/len(cosine_with_each_label)

        label_avg_embedding = label_embed[0]
        print("--------------------------------------------------")
        print(label_avg_embedding)

        for i in range(1,len(label_embed)):
            label_avg_embedding = label_avg_embedding + label_embed[i]

        label_avg_embedding = label_avg_embedding/len(label_embed)
        cosine_oftheavglabels = F.cosine_similarity(sentence_embed.float(), label_avg_embedding)

        print("--------------------------------------------------")
        print(label_avg_embedding)

        details.append(
            {
                "Max cosine similarity": float(max_cosine[0]),
                "Average cosine similarity": float(avg_cosine[0]),
                "Cosine similarity with the class average": float(cosine_oftheavglabels[0]),
                "Sum of all cosine similarity": float(max_cosine[0]+avg_cosine[0]+cosine_oftheavglabels[0])
            }
        )

        labels_metric.append(max_cosine+avg_cosine+cosine_oftheavglabels)

    if max(labels_metric)< threshold:
        prediction = -1
    else:
        prediction = labels_metric.index(max(labels_metric))

    return prediction, details



###########
#BERT Text classification
###########

def BCEPrediction(sentence, labels, threshold):
    details = []
    labels_score = []
    for label in labels:
        scores= []
        for l in label:
            c = classifier(sentence+" , "+l)
            print(l, c)
            scores.append(c[0]['score'])

        score_mean = np.mean(scores)
        score_median = np.median(scores)
        score_max = max(scores)

        details.append(
            {
                "Mean similarity" : float(score_mean),
                "Median similarity": float(score_median),
                "Max similarity": float(score_max),
                "Sum ": float(score_mean+score_median+score_max)
            }
        )

        labels_score.append(score_mean+score_median+score_max)

    if max(labels_score)< threshold:
        prediction = -1
    else:
        prediction = labels_score.index(max(labels_score))

    return prediction, details


def BERTCrossEncoder(sentences, labels, threshold):
    predictions = []
    details = []
    for sentence in sentences:
        print("Sentence --> " + sentence)
        label_max = []
        for label in labels:
            scores = []
            for l in label:
                c = classifier(sentence+" , "+l)
                print(l, c)
                scores.append(c[0]['score'])

            score_mean = np.mean(scores)
            score_median = np.median(scores)
            score_max = max(scores)
            label_max.append(score_max+score_median+score_mean)
        details.append(label_max)
        print(label_max)
        if max(label_max)<threshold:
            predictions.append(-1)
        else:
            predictions.append(label_max.index(max(label_max)))
    return details,predictions

# takes list of predictions and labels
def evaluation(class_labels, predictions, count_label):
    labels = [x+1 for x in range(count_label)]
    return classification_report(class_labels, predictions, labels=labels, output_dict=True) #classification report from sklearn
