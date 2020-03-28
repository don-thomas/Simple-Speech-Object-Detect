# Author: Don Thomas
# Program Description: Simple Audio Image Detection, if you say the classic 80 COCO Objects it will try to find that object in the image
# If the user says "next": it will go to the next image
# If the user says "end": The program will end

import cv2
import os
import speech_recognition as sr  
import cvlib as cv
from cvlib.object_detection import draw_bbox
import matplotlib.pyplot as plt



audio_speech = sr.Recognizer()                                                                                   



images_folder_path = './ListOfImages/'
image_file_names = os.listdir(images_folder_path)
present_image_number = 0
max_image_number = len(image_file_names)
last_message = "Nothing"


def newLabelsToLook(message,bbox,objects,conf):
    new_bbox = []
    new_objects = []
    new_conf = []
    if message == "all":
        return bbox, objects,conf

    for n,present_label in enumerate(objects):
        if present_label.lower() in message:
            new_bbox.append(bbox[n])
            new_objects.append(objects[n])
            new_conf.append(conf[n])
    
    return new_bbox, new_objects, new_conf

orginal_im = cv2.imread(images_folder_path+image_file_names[present_image_number])
cv2.imshow('image',orginal_im)
cv2.waitKey(500)

while last_message.lower() != "end": 
    with sr.Microphone() as source:                                                                       
        audio = audio_speech.listen(source)   
    try:
        last_message = (audio_speech.recognize_google(audio)).lower()
        message_array = last_message.split(" ")
        #last_message = (audio_speech.recognize_sphinx(audio)).lower()
        print(f"Google API Heard: {last_message}")
    except sr.UnknownValueError:
        #print("Google API Can not understand what was said")
        continue
    except sr.RequestError as e:
        print("Google API not working, Maybe Interent is down")
        continue

    if last_message == "next":
        present_image_number += 1
        if present_image_number>=max_image_number:
            present_image_number = 0
        orginal_im = cv2.imread(images_folder_path+image_file_names[present_image_number])
        cv2.imshow('image',orginal_im)
        cv2.waitKey(500)
        continue


    output_image = cv2.imread(images_folder_path+image_file_names[present_image_number])
    bbox, objects, conf = cv.detect_common_objects(output_image)
    bbox, objects, conf = newLabelsToLook(message_array,bbox,objects,conf)
    output_image = draw_bbox(output_image, bbox, objects, conf)
    cv2.imshow('image',output_image)
    cv2.waitKey(500)


cv2.destroyAllWindows()

