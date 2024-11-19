# %% [markdown]
# ## Objective
# 1. detect face in ID card and assign bounding box
# 2. get detected face and separate it to the new image 
# 3. extract the image feature for next process (i.e image comparison)
# 
# ## Approach
# - resize image to uniform size 
# - feed the image to mtcnn for face detection
# - applied bounding box
# - get detected face image
# - feature extraction for image comparison

import os
import glob
import re
import cv2
from cv2 import threshold
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import io
from PIL import Image, ImageOps
from statistics import mean, median
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

not_smiliar = []
not_smiliar_names=[]
similar = []
similar_names=[]
accuracy = []
confusion=[]

threshold = 1.4
y = [x * 0.1 for x in range(0, 10)]

# path
rootdir = ".\dataset"
for i in y:
    y_true=[]
    y_pred=[]
    for subdir, dirs, files in os.walk(rootdir):
        for dir in dirs:
            if(os.path.join(dir)!="bagus"):
                print(os.path.join(dir))
                example = os.path.join(rootdir, os.path.join(dir)) # feel free to change for specific photo that u want to test
                # print(example)
                photos = glob.glob(example + "\\*")

                pattern = r"(\w+\.\w+)"
                pattern2 = r"_(\w+)\."
                dict_photo = dict.fromkeys(["ktp_path", "selfie_path"])
                for photo in photos:
                    match = re.search(pattern, photo)
                    type_photo = re.search(pattern2, match.group(0))
                    if type_photo.group(1) == "ktp":
                        dict_photo["ktp_path"] = photo

            for j in dirs:
                
                if(os.path.join(j)!="bagus"):
                    example = os.path.join(rootdir, os.path.join(j)) # feel free to change for specific photo that u want to test
                    # print(example)
                    photos = glob.glob(example + "\\*")

                    pattern = r"(\w+\.\w+)"
                    pattern2 = r"_(\w+)\."
                    
                    for photo in photos:
                        match = re.search(pattern, photo)
                        type_photo = re.search(pattern2, match.group(0))
                        
                        if type_photo.group(1) == "selfie":
                            dict_photo["selfie_path"] = photo
                
                
                    # #### Face detector: MTCNN

                    # open images and resize
                    with open(dict_photo["ktp_path"], mode='rb') as file:
                        file_bytes = file.read()
                    img_pil = Image.open(io.BytesIO(file_bytes))
                    img_pil = ImageOps.exif_transpose(img_pil)
                    img_ktp = np.array(img_pil)
                    h, w, c = img_ktp.shape
                    width = 1280
                    height = int(h * (1280/w))
                    # print(width,height,c)

                    img_ktp = cv2.resize(img_ktp, (width, height), interpolation=cv2.INTER_LINEAR)

                    with open(dict_photo["selfie_path"], mode='rb') as file:
                        file_bytes = file.read()
                    img_pil = Image.open(io.BytesIO(file_bytes))
                    img_pil = ImageOps.exif_transpose(img_pil)
                    img_selfie = np.array(img_pil)
                    h, w, c = img_selfie.shape
                    width = 1280
                    height = int(h * (1280/w))
                    # print(width,height,c)

                    img_selfie = cv2.resize(img_selfie, (width, height), interpolation=cv2.INTER_LINEAR)

                    imgs = [img_ktp, img_selfie]

                    # image detection(s)
                    results = dict.fromkeys(["ktp_result", "selfie_result"])
                    detector = MTCNN()
                    results["ktp_result"] = detector.detect_faces(img_ktp)
                    results["selfie_result"] = detector.detect_faces(img_selfie)

                    # plt.figure(figsize=(5, 5))
                    # plt.imshow(img_ktp)
                    # plt.title("example id card")
                    # plt.show()

                    results["ktp_result"]

                    # store faces
                    faces = list()
                    detected_face = np.copy(img_ktp)
                    x1, y1, width, height = results["ktp_result"][0]['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1+width, y1+height

                    # detected_face = cv2.rectangle(detected_face, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    detected_face = detected_face[y1:y2, x1:x2]
                    h, w, c = detected_face.shape
                    width = 1280
                    height = int(h * (1280/w))
                    # print(width,height,c)

                    detected_face = cv2.resize(detected_face, (width, height), interpolation=cv2.INTER_LINEAR)

                    faces.append(detected_face)

                    # for selfie
                    # pick face
                    w = results["selfie_result"][0]['box'][2]
                    h = results["selfie_result"][0]['box'][3]
                    max_area = w * h
                    i_result = 0
                    for i in range(len(results["selfie_result"])):
                        w = results["selfie_result"][i]['box'][2]
                        h = results["selfie_result"][i]['box'][3]
                        area = w * h
                        
                        if area > max_area:
                            max_area = area
                            i_result = i
                            
                    # get detected face
                    detected_face = np.copy(img_selfie)
                    x1, y1, width, height = results["selfie_result"][i_result]['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1+width, y1+height
                    # detected_face = cv2.rectangle(detected_face, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    detected_face = detected_face[y1:y2, x1:x2]
                    h, w, c = detected_face.shape
                    width = 1280
                    height = int(h * (1280/w))
                    # print(width,height,c)

                    detected_face = cv2.resize(detected_face, (width, height), interpolation=cv2.INTER_LINEAR)

                    faces.append(detected_face)
                    # plt.figure(figsize=(5, 5))
                    # plt.imshow(faces[1])
                    # plt.title("example id card")
                    # plt.show()

                    from arcface import ArcFace
                    # sharpening
                    kernel = np.array([[0, -1, 0],
                                    [-1, 5,-1],
                                    [0, -1, 0]])
                    faces[0] = cv2.filter2D(src=faces[0], ddepth=-1, kernel=kernel)
                    faces[1] = cv2.filter2D(src=faces[1], ddepth=-1, kernel=kernel)

                    # ArcFace
                    arc_face = ArcFace.ArcFace('./model/arcface_model')
                    emb1 = arc_face.calc_emb(faces[0])
                    emb2 = arc_face.calc_emb(faces[1])

                    distance = arc_face.get_distance_embeddings(emb1, emb2)
                
                    not_smiliar.append(distance)
                    not_smiliar_names.append([os.path.join(dir),os.path.join(j)])
                    print(f"Not similar. Distance {distance}")

                    if os.path.join(dir)!=os.path.join(j):
                        y_true.append(1) #not similar
                    else:
                        y_true.append(0) #similar

                    
                    if distance > threshold:
                        y_pred.append(1) #predicted not similar
                    elif distance <= threshold:
                        y_pred.append(0) #predicted similar
                    
                    
                    # else:
                        # similar_names.append([os.path.join(dir),os.path.join(j)])
                        # not_smiliar.append(distance)
                        # print(f"Similar. Distance {distance}")

    threshold+=0.01
    
    print(confusion_matrix(y_true,y_pred))
    print(accuracy_score(y_true,y_pred))
    accuracy.append(accuracy_score(y_true,y_pred))
    confusion.append(confusion_matrix(y_true,y_pred))

print(confusion)
print(accuracy)
# print(not_smiliar)
# print("%s%s" % ("length: ", len(not_smiliar)))
# print("%s%s" % ("avg: ", mean(not_smiliar)))
# print("%s%s" % ("min: ", min(not_smiliar)))
# print("%s%s" % ("max: ", max(not_smiliar)))
# print(not_smiliar_names)

# print("similar: ")
# print(similar)
# print("%s%s" % ("length: ", len(similar)))
# print("%s%s" % ("avg: ", mean(similar)))
# print("%s%s" % ("min: ", min(similar)))
# print("%s%s" % ("max: ", max(similar)))
# print(similar_names)



# %%
