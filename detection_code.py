#Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
import argparse
import pygame
import time
import dlib
import cv2

pygame.mixer.init()
pygame.mixer.music.load("audio_alert.wav")

EYE_ASPECT_RATIO_THRESHOLD = 0.27
MOUTH_AR_THRESH = 0.70


EYE_ASPECT_RATIO_CONSEC_FRAMES = 60
MOUTH_ASPECT_RATIO_CONSEC_FRAMES = 30

COUNTER = 0
w_before=0

#Load face cascade which will be used to draw a rectangle around detected faces.
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#This function calculates and return eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)

    return ear
def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
	A = distance.euclidean(mouth[0], mouth[8]) # 51, 59
	B = distance.euclidean(mouth[2], mouth[6]) # 53, 57

	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
	C = distance.euclidean(mouth[0], mouth[6]) # 49, 55

	# compute the mouth aspect ratio
	mar = (A + B) / (2.0 * C)

	# return the mouth aspect ratio
	return mar    



print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']
# for_mouth
(mStart, mEnd) = (49, 68)

frame_width = 640
frame_height = 360

#Start webcam video capture
video_capture = cv2.VideoCapture(0)
print("[INFO] starting video stream thread...")

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
#Give some time for camera to initialize(not required)
time.sleep(2)

while(True):
    #Read each frame and flip it, and convert to grayscale
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect facial points through detector function
    faces = detector(gray, 0)
    
    no_of_faces=len(faces)

    #Detect faces through haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Draw rectangle around each face detected
    for (x,y,w,h) in face_rectangle:
        if (w-w_before<-15) and (no_of_faces==1):
            print("moving away")
        elif (w-w_before>15) and (no_of_faces==1):
            print("moving closer")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        w_before=w

    #Detect facial points
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Get array of coordinates of leftEye and rightEye
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        eye= (leftEye+rightEye) /2
        eyeEar= eye_aspect_ratio(eye)
        ear= eyeEar
        #mouth_cooordinates
        mouth = shape[mStart:mEnd]
        Mouth_mar = mouth_aspect_ratio(mouth)
        mar= Mouth_mar
        

        #Calculate aspect ratio of both eyes
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2
        print(eyeAspectRatio)



        #Use hull to remove convex contour discrepencies and draw eye shape around eyes
        # leftEyeHull = cv2.convexHull(leftEye)
        # rightEyeHull = cv2.convexHull(rightEye)
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # compute the convex hull for the mouth, then
		# visualize the mouth
        # mouthHull = cv2.convexHull(mouth)
		
        # cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        # cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        #Detect if eye aspect ratio is less than threshold
        if(ear <= EYE_ASPECT_RATIO_THRESHOLD) and mar>=MOUTH_AR_THRESH or ear<= EYE_ASPECT_RATIO_THRESHOLD or mar>= MOUTH_AR_THRESH:

            COUNTER += 1
            #If no. of frames is greater than threshold frames,
            if COUNTER >= MOUTH_ASPECT_RATIO_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            pygame.mixer.music.stop()
            COUNTER = 0
        

# Write the frame into the file 'output.avi'
    out.write(frame)
    #Show video feed
    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

#Finally when video capture is over, release the video capture and destroyAllWindows
video_capture.release()
cv2.destroyAllWindows()