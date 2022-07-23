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
        details,results = Q1BERTEncodingCosine(input)

        return Response({
            "Persona match":results[0],
            "Persona details": details[0],
            "Action match":results[1],
            "Action details": details[1],
            "Goal match":results[2],
            "Goal details":details[2]

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

        if valid_data["question"]=="q3":
            labels = [['Monthly fee', 'Fee per user ', 'Charge', 'Cost', 'Average revenue per user', 'Monthly payment', 'Recurring monthly charge', 'Monthly service fee', 'Monthly access fee', 'Monthly maintenance fee', 'Membership fee', 'Subscription fee', 'Dues', 'charge per user', 'fee per user'], ['users', 'Subscribers', 'Customers', 'clients', 'patrons', 'payees', 'members', 'number of people using the service', 'number of users of the service', 'amount of people using the service', 'how many people are using the service', 'how popular is the service', 'how many users does the service have', 'is the service used by a lot of people', 'how well-known is the service', 'what is the user base for the service', 'how big is the market for the service'], ['Average customer lifetime', 'Customer lifespan', 'Duration of subscription', 'Churn', 'Attrition', 'employee turnover', 'Attrition rates', 'Average customer lifetime value', 'Average length of customer life cycle', 'Average time customers remain active', 'Customer attrition rate over time', 'How long do customers stay on average?', 'What is the typical lifespan of a customer?', 'staff turnover', 'Average client lifetime', 'Median customer lifetime', 'Ordinary customer lifetime']]

            url = "http://ec2-65-1-1-17.ap-south-1.compute.amazonaws.com:5001/model/parse"

        else:
            labels = [['Make sure questions are easy to understand', 'Make sure questions are easy to direct', 'Make sure questions are easy to straightforward', 'Use short questions', 'Use succinct questions', 'Use clear questions', 'Avoid questions that are too long'], ['Keep questions open ended', 'Do not ask yes/no questions', 'Do not ask closed questions', 'Do not ask leading questions', 'Avoid questions that suggest the answer you want', "Don't ask directly what they want"], ['Have a logical structure to the questions', 'Questions should flow from one to the other', 'Questions should be sequenced in a natural way', 'Warm up before going into the detailed questions', 'Start with some easier questions to help them relax', 'Prepare your questions beforehand']]

            url = "http://ec2-65-1-1-17.ap-south-1.compute.amazonaws.com:5002/model/parse"

        response = []

        if "BCE" in models:
            prediction, details = BCEPrediction(sentence, labels)
            if valid_data["save"]:
                entry = Record.objects.create(
                    sentence = sentence,
                    model = "BCE",
                    question = valid_data["question"],
                    label = valid_data["label"],
                    prediction = prediction,
                    details = details

                )
                entry.save()
            response.append(
                {
                    "model": "BCE",
                    "prediction": prediction,
                    "details": details
                }
            )

        if "BERTCosine" in models:
            prediction, details = BERTCosinePrediction(sentence, labels)
            if valid_data["save"]:
                entry = Record.objects.create(
                    sentence = sentence,
                    model = "BERTCosine",
                    question = valid_data["question"],
                    label = valid_data["label"],
                    prediction = prediction,
                    details = details

                )
                entry.save()
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

            if valid_data["save"]:
                entry = Record.objects.create(
                    sentence = sentence,
                    model = "RASA",
                    question = valid_data["question"],
                    label = valid_data["label"],
                    prediction = prediction,
                    details = details

                )
                entry.save()

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
        if valid_data["question"]=="q3":
            labels = [['Monthly fee', 'Fee per user ', 'Charge', 'Cost', 'Average revenue per user', 'Monthly payment', 'Recurring monthly charge', 'Monthly service fee', 'Monthly access fee', 'Monthly maintenance fee', 'Membership fee', 'Subscription fee', 'Dues', 'charge per user', 'fee per user'], ['users', 'Subscribers', 'Customers', 'clients', 'patrons', 'payees', 'members', 'number of people using the service', 'number of users of the service', 'amount of people using the service', 'how many people are using the service', 'how popular is the service', 'how many users does the service have', 'is the service used by a lot of people', 'how well-known is the service', 'what is the user base for the service', 'how big is the market for the service'], ['Average customer lifetime', 'Customer lifespan', 'Duration of subscription', 'Churn', 'Attrition', 'employee turnover', 'Attrition rates', 'Average customer lifetime value', 'Average length of customer life cycle', 'Average time customers remain active', 'Customer attrition rate over time', 'How long do customers stay on average?', 'What is the typical lifespan of a customer?', 'staff turnover', 'Average client lifetime', 'Median customer lifetime', 'Ordinary customer lifetime']]


        else:
            labels = [['Make sure questions are easy to understand', 'Make sure questions are easy to direct', 'Make sure questions are easy to straightforward', 'Use short questions', 'Use succinct questions', 'Use clear questions', 'Avoid questions that are too long'], ['Keep questions open ended', 'Do not ask yes/no questions', 'Do not ask closed questions', 'Do not ask leading questions', 'Avoid questions that suggest the answer you want', "Don't ask directly what they want"], ['Have a logical structure to the questions', 'Questions should flow from one to the other', 'Questions should be sequenced in a natural way', 'Warm up before going into the detailed questions', 'Start with some easier questions to help them relax', 'Prepare your questions beforehand']]


        threshold = valid_data["threshold"]

        # using BERT sentence encoding with cosine similarity
        sentence_net = BERTencoding(sentences,labels)
        details,predictions = cosineSimilarity(sentence_net, threshold)

        # generating response
        response = []
        i=0
        for sentence,label in sentences.items():
            response.append(
                {
                    "sentence":sentence,
                    "prediction": predictions[i],
                    "label": label,
                    "confidence": details[i]
                }
            )
            i=i+1

        return Response(response)

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
        if valid_data["question"]=="q3":
            labels = [['Monthly fee', 'Fee per user ', 'Charge', 'Cost', 'Average revenue per user', 'Monthly payment', 'Recurring monthly charge', 'Monthly service fee', 'Monthly access fee', 'Monthly maintenance fee', 'Membership fee', 'Subscription fee', 'Dues', 'charge per user', 'fee per user'], ['users', 'Subscribers', 'Customers', 'clients', 'patrons', 'payees', 'members', 'number of people using the service', 'number of users of the service', 'amount of people using the service', 'how many people are using the service', 'how popular is the service', 'how many users does the service have', 'is the service used by a lot of people', 'how well-known is the service', 'what is the user base for the service', 'how big is the market for the service'], ['Average customer lifetime', 'Customer lifespan', 'Duration of subscription', 'Churn', 'Attrition', 'employee turnover', 'Attrition rates', 'Average customer lifetime value', 'Average length of customer life cycle', 'Average time customers remain active', 'Customer attrition rate over time', 'How long do customers stay on average?', 'What is the typical lifespan of a customer?', 'staff turnover', 'Average client lifetime', 'Median customer lifetime', 'Ordinary customer lifetime']]


        else:
            labels = [['Make sure questions are easy to understand', 'Make sure questions are easy to direct', 'Make sure questions are easy to straightforward', 'Use short questions', 'Use succinct questions', 'Use clear questions', 'Avoid questions that are too long'], ['Keep questions open ended', 'Do not ask yes/no questions', 'Do not ask closed questions', 'Do not ask leading questions', 'Avoid questions that suggest the answer you want', "Don't ask directly what they want"], ['Have a logical structure to the questions', 'Questions should flow from one to the other', 'Questions should be sequenced in a natural way', 'Warm up before going into the detailed questions', 'Start with some easier questions to help them relax', 'Prepare your questions beforehand']]


        threshold = valid_data["threshold"]

        # using BERT sentence encoding with cosine similarity
        details,predictions = BERTCrossEncoder(sentences, labels, threshold)

        # generating response
        response = []
        i=0
        for sentence,label in sentences.items():
            response.append(
                {
                    "sentence":sentence,
                    "prediction": predictions[i],
                    "label": label,
                    "confidence": details[i]
                }
            )
            i=i+1

        return Response(response)


class SnippetViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides `list`, `create`, `retrieve`,
    `update` and `destroy` actions.
    """
    queryset = Snippet.objects.all()
    serializer_class = SnippetSerializer
