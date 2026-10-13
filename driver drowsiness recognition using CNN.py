from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
from .forms import DriverRegistrationForm
from .models import DriverRegistrationModel, FattigueInfoModel
import cv2
import os
import numpy as np
import datetime
from keras.models import load_model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from pygame import mixer
def DriverRegisterActions(request):

    if request.method == 'POST':

        form = DriverRegistrationForm(request.POST)

        if form.is_valid():
            form.save()
            messages.success(request, "Registration Successful")
            form = DriverRegistrationForm()

        else:
            messages.error(request, "Email or Mobile already exists")

    else:
        form = DriverRegistrationForm()

    return render(request, 'AutoistRegister.html', {'form': form})
def AutoistLoginCheck(request):

    if request.method == "POST":

        loginid = request.POST.get('loginid')
        password = request.POST.get('pswd')

        try:

            user = DriverRegistrationModel.objects.get(
                loginid=loginid,
                password=password
            )

            if user.status == "activated":

                request.session['id'] = user.id
                request.session['loggeduser'] = user.name
                request.session['loginid'] = user.loginid
                request.session['email'] = user.email
                request.session['vehiclenumber'] = user.vehiclenumber

                return render(request, 'autoist/AutoistHome.html')

            else:
                messages.error(request, "Account not activated")

        except:
            messages.error(request, "Invalid Login ID or Password")

    return render(request, 'AutoistLogin.html')
def TrainCNNModel():

    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu', input_shape=(24,24,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    model.save(os.path.join(settings.MEDIA_ROOT,'models','drowsiness_model.h5'))

    return model
class FatigueDetections:

    def start_process(self):

        mixer.init()

        alarm = mixer.Sound(
            os.path.join(settings.MEDIA_ROOT,'alarm.wav')
        )

        model = load_model(
            os.path.join(settings.MEDIA_ROOT,'models','drowsiness_model.h5')
        )

        face_detector = cv2.CascadeClassifier(
            os.path.join(settings.MEDIA_ROOT,
            'haar cascade files',
            'haarcascade_frontalface_alt.xml')
        )

        eye_detector = cv2.CascadeClassifier(
            os.path.join(settings.MEDIA_ROOT,
            'haar cascade files',
            'haarcascade_lefteye_2splits.xml')
        )

        camera = cv2.VideoCapture(0)

        score = 0
        fatigue_flag = False

        while True:

            ret, frame = camera.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:

                face = frame[y:y+h, x:x+w]

                eyes = eye_detector.detectMultiScale(gray)

                for (ex,ey,ew,eh) in eyes:

                    eye = face[ey:ey+eh, ex:ex+ew]

                    eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                    eye = cv2.resize(eye,(24,24))

                    eye = eye / 255.0
                    eye = eye.reshape(1,24,24,1)

                    prediction = model.predict(eye)

                    if np.argmax(prediction) == 0:
                        score += 1
                        label = "Closed"
                    else:
                        score -= 1
                        label = "Open"

                    if score < 0:
                        score = 0

                    cv2.putText(frame,label,(10,30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,(255,255,255),2)

                    if score > 15:

                        alarm.play()
                        fatigue_flag = True

                        cv2.rectangle(frame,
                                      (0,0),
                                      (frame.shape[1],frame.shape[0]),
                                      (0,0,255),3)

            cv2.imshow("Driver Monitoring",frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera.release()
        cv2.destroyAllWindows()
        return fatigue_flag
def DetectFatigueDriver(request):

    detector = FatigueDetections()

    fatigue = detector.start_process()

    if fatigue:
        user_name = request.session['loggeduser']
        login_user = request.session['loginid']
        email = request.session['email']
        vehicle = request.session['vehiclenumber']

        date = datetime.datetime.now()
        FattigueInfoModel.objects.create(
            user_name=user_name,
            login_user=login_user,
            email=email,
            vehiclenumber=vehicle,
            fatigue="Detected",
            c_date=date
        )

    return render(request,'autoist/DetectionImage.html')