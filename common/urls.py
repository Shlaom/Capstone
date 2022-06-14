from django.urls import path
from django.contrib.auth import views as auth_views

from . import views

app_name = 'common'

urlpatterns = [

    path('login/', views.login_main, name='login_main'),  # 로그인
    path('logout/', auth_views.LogoutView.as_view(), name='logout'),  # 로그아웃
    path('signUp/', views.signUp, name='signUp'), #회원가입 페이지
]