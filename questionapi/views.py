from django.shortcuts import render

# Create your views here.
from .userstorymodelling import *
from .modelling import *
from rest_framework.views import APIView
from rest_framework import viewsets
from .models import *
from .serializers import *
import json
from rest_framework.response import Response
# from django.views.decorators.csrf import csrf_exempt
#
# @csrf_exempt

class q1processing(APIView):
    #{
    #    "sentence":["user persona", "action to be performed","goals"],
    #}
    def post(self, request):
        data = request.body.decode('utf-8')
        valid_data = json.loads(data)

        input = valid_data["sentence"]
        print(input)
        results = Q1BERTEncodingCosine(input)

        return Response({
            "Persona match":results[0],
            "Action match":results[1],
            "Goal match":results[2]

            })


class evaluation(APIView):
    # {
    #     "sentence":<>,
    #     "label":<>,
    #     "models":[<>,<>],
    #     "question":"q1"/"q2"
    # }

    def post(self, request):
        data = request.body.decode('utf-8')
        valid_data = json.loads(data)
        sentence = valid_data["sentence"]
        models = valid_data["models"]

        if "label" in valid_data:
            test_label = valid_data["label"]

        if valid_data["question"]=="q1":
            labels = [["Monthly fee", "Fee per user", "Charge", "Cost", "Average revenue per user", "Recurring monthly fee", "Fee charged per user", "Monthly service charge", "User fee", "Monthly subscription fee"], ["users", "Subscribers", "Customers", "clients", "patrons", "payees", "members"], ["Average lifetime", "lifespan", "Duration subscription", "Churn", "Attrition", "churn rate", "Churn analysis", "Churn prediction", "Churn prevention", "Reducing churn", "Increasing retention", "retention rate", "Customer attrition rate", "Attrition analysis", "Attrition prediction", "Attrition prevention"]]
        else:
            labels = [['Make sure questions are easy to understand', 'Make sure questions are easy to direct', 'Make sure questions are easy to straightforward', 'Use short questions', 'Use succinct questions', 'Use clear questions', 'Avoid questions that are too long'], ['Keep questions open ended', 'Do not ask yes/no questions', 'Do not ask closed questions', 'Do not ask leading questions', 'Avoid questions that suggest the answer you want', "Don't ask directly what they want"], ['Have a logical structure to the questions', 'Questions should flow from one to the other', 'Questions should be sequenced in a natural way', 'Warm up before going into the detailed questions', 'Start with some easier questions to help them relax', 'Prepare your questions beforehand']]

        response = []

        if "BCE" in models:
            prediction, details = BCEPrediction(sentence, labels)
            response.append(
                {
                    "model": "BCE",
                    "prediction": prediction,
                    "details": details
                }
            )

        if "BERTCosine" in models:
            prediction, details = BERTCosinePrediction(sentence, labels)
            response.append(
                {
                    "model": "BERTCosine",
                    "prediction": prediction,
                    "details": details
                }
            )

        return Response(response)

class CosineSimilarity(APIView):
    # Input json expected
        # {
        #     sentences : {
        #         <sentence>:<label>,
        #         ..
        #     },
        #     class_list : [
        #     [<list of phrases in label1>],
        #     ..
        #     ],
        #     threshold : <float>,
        # }

    def post(self, request):
        data = request.body.decode('utf-8')
        valid_data = json.loads(data)
        sentences = valid_data["sentences"]
        labels = valid_data["class_list"]
        count = len(labels)
        threshold = valid_data["threshold"]

        # using BERT sentence encoding with cosine similarity
        sentence_net = BERTencoding(sentences,labels)
        predictions = cosineSimilarity(sentence_net, threshold)

        # generating class labels
        class_labels = []
        for key, value in sentences.items():
            class_labels.append(value)

        return Response(evaluation(predictions, class_labels, count+1))

class BERTClassification(APIView):
    # Input json expected
        # {
        #     sentences : {
        #         <sentence>:<label>,
        #         ..
        #     },
        #     class_list : [
        #     [<list of phrases in label1>],
        #     ..
        #     ],
        #     threshold : <float>,
        # }

    def post(self, request):
        data = request.body.decode('utf-8')
        valid_data = json.loads(data)
        sentences = valid_data["sentences"]
        labels = valid_data["class_list"]
        count = len(labels)
        threshold = valid_data["threshold"]

        # using BERT sentence encoding with cosine similarity
        predictions = BERTCrossEncoder(sentences, labels)

        # generating class labels
        class_labels = []
        for key, value in sentences.items():
            class_labels.append(value)

        return Response(evaluation(predictions, class_labels, count+1))


class SnippetViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.
    """
    queryset = Snippet.objects.all()
    serializer_class = SnippetSerializer
