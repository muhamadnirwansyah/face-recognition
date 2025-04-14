
import os
import pickle
import cv2
import face_recognition
from sklearn import neighbors


def train():
    x = []
    y = []

    for name in os.listdir("train_dir"):
        if not os.path.isdir(os.path.join("train_dir",name)):
            continue
        
        subdir = os.listdir(os.path.join("train_dir",name))
        for dir in subdir:
            filename = os.path.join("train_dir",name,dir)
            face = face_recognition.load_image_file(filename)
            face_location = face_recognition.face_locations(face)
            
            if len(face_location) !=1:
                print("photo tidak ada object wajahnya ! : {}",filename)
                continue
            
            x.append(face_recognition.face_encodings(face, known_face_locations=face_location)[0])
            y.append(name)
            
    
    model = neighbors.KNeighborsClassifier(n_neighbors=2)
    model.fit(x,y)
    
    with open("face_recognition_model.clf",'wb') as f:
        pickle.dump(model, f)
    
if __name__ == "__main__":
    #train()
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error : Camera not accessible !")
    else:
        print("Camera opened sucessfully !")
    
    with open("face_recognition_model.clf",'rb') as f:
        model = pickle.load(f)
        
    face_names = []    
    face_locatios = []
    while True:
        ret, frame = video_capture.read()
        
        small_size = cv2.resize(frame,None,fx=0.25, fy=0.25)
        face_locatios = face_recognition.face_locations(small_size)
        if len (face_locatios) == 0:
            continue
        
        face_encodings = face_recognition.face_encodings(small_size, known_face_locations=face_locatios)
        face_names = []
        for name in model.predict(face_encodings):
            face_names.append(name)
        
        for (top, right, bottom,left), name in zip(face_locatios, face_names):
            
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            cv2.rectangle(frame, (left, top), (right,bottom),(0,255,0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left, bottom - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 0), 1)
        
        print(face_names)
        cv2.imshow('Video',frame)
        
        if (cv2.waitKey(1) &0xFF) == ord('q'):
            break