from django.db import models

# Create your models here.

class Record(models.Model):
    QUESTIONS = (
    ("q2", "q2"),
    ("q3", "q3"),
    )
    id = models.AutoField(primary_key=True)
    sentence = models.CharField(max_length=100)
    model = models.CharField(max_length=100)
    question = models.CharField(max_length=20, choices = QUESTIONS)
    label = models.CharField(max_length=50)
    prediction = models.CharField(max_length=50)
    details = models.JSONField(default=list, blank=True)
    #details = models.CharField(max_length=500)
    created = models.DateTimeField(auto_now_add =True)

    def __str__(self):
        return self.sentence

class Snippet(models.Model):
    created = models.DateTimeField(auto_now_add=True)
    title = models.CharField(max_length=100, blank=True, default='')
