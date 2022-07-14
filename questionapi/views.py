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
import requests
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

            url = "http://ec2-3-110-115-41.ap-south-1.compute.amazonaws.com:5001/model/parse"

        else:
            labels = [['Monthly fee', 'Fee per user ', 'Charge', 'Cost', 'Average revenue per user', 'Monthly payment', 'Recurring monthly charge', 'Monthly service fee', 'Monthly access fee', 'Monthly maintenance fee', 'Membership fee', 'Subscription fee', 'Dues', 'charge per user', 'fee per user'], ['users', 'Subscribers', 'Customers', 'clients', 'patrons', 'payees', 'members', 'number of people using the service', 'number of users of the service', 'amount of people using the service', 'how many people are using the service', 'how popular is the service', 'how many users does the service have', 'is the service used by a lot of people', 'how well-known is the service', 'what is the user base for the service', 'how big is the market for the service'], ['Average customer lifetime', 'Customer lifespan', 'Duration of subscription', 'Churn', 'Attrition', 'employee turnover', 'Attrition rates', 'Average customer lifetime value', 'Average length of customer life cycle', 'Average time customers remain active', 'Customer attrition rate over time', 'How long do customers stay on average?', 'What is the typical lifespan of a customer?', 'staff turnover', 'Average client lifetime', 'Median customer lifetime', 'Ordinary customer lifetime']]

            url = "http://ec2-3-110-115-41.ap-south-1.compute.amazonaws.com:5002/model/parse"

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

        if "rasa" in models:
            payload = '{"text": "%s"}'%sentence
            headers = {
                'content-type': "application/json",
                }

            rasaresponse = requests.request("POST", url, data=payload, headers=headers)
            prediction = rasaresponse.json()['intent']['name']
            details = rasaresponse.json()['intent_ranking']

            response.append(
                {
                    "model": "Rasa Bot",
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
