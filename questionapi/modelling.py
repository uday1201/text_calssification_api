from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from torch.nn import functional as F
import numpy as np
tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
model = AutoModel.from_pretrained('deepset/sentence_bert')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from transformers import pipeline
classifier = pipeline("text-classification", model = "cross-encoder/qnli-electra-base")

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

            # getting the mean similarity for each label
            print("Mean similarity :", np.mean(similarity_list))
            labels_similarity.append(np.mean(similarity_list))
            print("\n")

        # getting the predicted class
        print(labels_similarity)
        predicted_labels = [1 if (y>threshold and y==max(labels_similarity))else 0 for y in labels_similarity]
        if 1 in predicted_labels:
            predicted_labels.append(0)
        else:
            predicted_labels.append(1)

        # append the predicted class to the prediction list
        predictions.append(predicted_labels.index(1)+1)
        print("Predicted class : ", predicted_labels.index(1)+1)
        print("------------------------------------------------------")
    return predictions

###########
#BERT Text classification
###########

def BERTCrossEncoder(sentences, labels, stat=None):
    predictions = []
    for sentence in sentences:
        print("Sentence --> " + sentence)
        label_max = []
        for label in labels:
            score_max = []
            for l in label:
                c = classifier(sentence+" , "+l)
                print(l, c)
                score_max.append(c[0]['score'])

            if stat == "MEAN":
                label_max.append(np.mean(score_max))
            elif stat == "MEDIAN":
                label_max.append(np.median(score_max))
            else:
                label_max.append(max(score_max))

        print(label_max)
        if max(label_max)<.75:
            predictions.append(len(labels)+1)
        else:
            predictions.append(label_max.index(max(label_max))+1)
    return predictions

# takes list of predictions and labels
def evaluation(class_labels, predictions, count_label):
    labels = [x+1 for x in range(count_label)]
    return classification_report(class_labels, predictions, labels=labels, output_dict=True) #classification report from sklearn
