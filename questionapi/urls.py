from django.urls import include, path
from rest_framework import routers
from . import views
from rest_framework.authtoken.views import obtain_auth_token
from rest_framework.documentation import include_docs_urls
from django.conf import settings
from django.conf.urls.static import static


router = routers.DefaultRouter()
router.register(r'snippet', views.SnippetViewSet)


# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('api/', include(router.urls)),
    path('docs/', include_docs_urls(title='Snippet API')),
    path('api/CosineSimilarity/', views.CosineSimilarity.as_view(), name='CosineSimilarity'),
    path('api/BERTClassification/', views.BERTClassification.as_view(), name='BERTClassification'),
    path('api/evaluation/', views.evaluation.as_view(), name='evaluation'),
]
