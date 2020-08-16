import cv2
import os

# face_cascade_path = haarcascade_frontalface_default.xmlのpath
# eye_cascade_path = haarcascade_eye.xmlのpath

face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

cap = cv2.VideoCapture(1)


def save_frame_sec(video_path, sec, result_path):

    if not cap.isOpened():
        return

    os.makedirs(os.path.dirname(result_path), exist_ok=True)

    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.set(cv2.CAP_PROP_POS_FRAMES, round(fps * sec))

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(result_path, frame)


while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2.0, minNeighbors=5)
    for x, y, w, h in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        face = img[y: y + h, x: x + w]
        face_gray = gray[y: y + h, x: x + w]
        eyes = eye_cascade.detectMultiScale(face_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(face, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    cv2.imshow('video image', img)
    key = cv2.waitKey(10)
    
    
    
    if key == 27:  # ESCキーで終了
        save_frame_sec(1, 10, 'data/temp/result_screenshot.jpg')
        break

cap.release()
cv2.destroyAllWindows()