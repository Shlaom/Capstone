
from django.contrib import admin
from django.urls import path, include
from .views import *

# urlpatterns = [
#     path('admin/', admin.site.urls),
# ]

urlpatterns = [

    path('', include('masking.urls')),
    path('common/', include('common.urls')),  # common(login) page
]
