from django.shortcuts import render
from django.views.decorators import gzip
from django.http import StreamingHttpResponse, HttpResponse

import numpy as np

from masking.global_variables import frame_proc

# 메인 페이지
def main(request):
    return render(request, 'main.html')

# 영상 송출 페이지
def home(request):
    return render(request, 'home.html')

@gzip.gzip_page
def video(request):
    try:
        # 응답 본문이 데이터를 계속 추가할 것이라고 브라우저에 알리고 브라우저에 원래 데이터를 데이터의 새 부분으로 교체하도록 요청
        # 즉, 서버에서 얻은 비디오가 jpeg 사진으로 변환되어 브라우저에 전달, 브라우저는 비디오 효과를 위해 이전 이미지를 새 이미지로 지속적 교체
        return StreamingHttpResponse(gen(frame_proc), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def registration(request):
    return render(request, 'registration.html')

def face_capture(request):
    try:
        return StreamingHttpResponse(gen_for_registration(frame_proc), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass

def gen_for_registration(camera):
    camera.imgs = []
    count = 0

    while True:
        count += 1
        frame = camera.get_frame_for_registration()
        if count == 60:
            break
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    if camera.imgs != []:
        camera.data['Registered_Face'] = np.array(camera.imgs)
        camera.train()
    else:
        print('There are no faces for registration..')

def mypage(request):
    return render(request, 'navBar/myPage/myPage.html')

def masking_on(request):
    frame_proc.mode = 1
    s = request.GET.get('param')
    print(s)
    return HttpResponse(None, content_type='application/json')

def masking_off(request):
    frame_proc.mode = 0
    s = request.GET.get('param')
    print(s)
    return HttpResponse(None, content_type='application/json')

def mode_mosaic(request):
    frame_proc.sign = 1
    s = request.GET.get('param')
    print(s)
    return HttpResponse(None, content_type='application/json')

def mode_imaging(request):
    frame_proc.sign = 3
    s = request.GET.get('param')
    print(s)
    return HttpResponse(None, content_type='application/json')

def mode_test(request):
    frame_proc.sign = 2
    s = request.GET.get('param')
    print(s)
    return HttpResponse(None, content_type='application/json')
