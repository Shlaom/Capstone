from django.contrib import auth, messages
from django.shortcuts import render, redirect
from .models import User


""" ───────────────────────── 로그인 ───────────────────────── """

def login_main(request):
    if request.method == 'POST':
        userid = request.POST['userid']
        password = request.POST['password']
        user = auth.authenticate(request, userid=userid, password=password)

        if user is None:
            # print('userid 또는 password가 틀렸습니다.')
            return render(request, 'login/login.html', {'error': 'username 또는 password가 틀렸습니다.'})
        else:
            auth.login(request, user)
            return redirect('/')
    elif request.method == 'GET':
        return render(request, 'login/login.html')

""" ───────────────────────── 회원가입 ───────────────────────── """

def signUp(request):
    if request.method == 'GET':
        return render(request, 'login/signup.html')
    elif request.method == 'POST':
        username = request.POST.get('username', None)  # 이름
        userid = request.POST.get('userid', None)  # 아이디
        password1 = request.POST.get('password1', None)  # 비밀번호
        password2 = request.POST.get('password2', None)  # 비밀번호(확인)


        res_data = {}

        # 빈 칸 확인
        if not (username and userid and password1 and password2):
            res_data['error'] = "입력하지 않은 칸이 있습니다."
        # 아이디 중복 확인
        if User.objects.filter(userid=userid).exists():  # 아이디 중복 체크
            print('이미 존재하는 아이디입니다!')
            messages.warning(request, '이미 존재하는 아이디입니다!')
            return render(request, 'login/signup.html')
        # 비밀번호 일치 여부 확인
        if password1 != password2:
            res_data['error'] = '비밀번호가 일치하지 않습니다.'
        else:
            user = User.objects.create_user(userid=userid, username=username, password=password1)
            user.save()
        return render(request, 'main.html', res_data)