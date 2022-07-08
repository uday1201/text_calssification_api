from django.shortcuts import render

# Create your views here.
from .modelling import *
from rest_framework.views import APIView
from rest_framework import viewsets
from .models import *
from .serializers import *
import json
from rest_framework.response import Response

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
