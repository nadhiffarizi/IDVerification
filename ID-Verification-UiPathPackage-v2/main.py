import io
import json
import cv2
import numpy as np
from PIL import Image, ImageOps
from mtcnn.mtcnn import MTCNN
from arcface import ArcFace


class Main(object):
    def __init__(self):
        self.face_detector = MTCNN()
        self.arc_face = ArcFace.ArcFace('./model/arcface_model')

    def predict(self, skill_input):
        ''' the input are 2 files and stored in list of bytes
         should consist element of 2, provisioned in uipath studio
         the order: ktp, selfie '''

        # open images
        ktp_img = self.open_img(skill_input[0])
        ktp_img = self.preprocess(ktp_img)

        selfie_img = self.open_img(skill_input[1])
        selfie_img = self.preprocess(selfie_img)
        imgs = [ktp_img, selfie_img]

        # detect images
        results = dict.fromkeys(["ktp_result", "selfie_result"])
        results["ktp_result"] = self.face_detector.detect_faces(imgs[0])
        results["selfie_result"] = self.face_detector.detect_faces(imgs[1])

        # filter results
        if len(results["ktp_result"]) == 0:
            skill_output = {"distance": "-1", "note": "ktp", "detected": len(results["ktp_result"])}
            return json.dumps(skill_output)
        elif len(results["ktp_result"]) > 1:
            skill_output = {"distance": "-1", "note": "ktp", "detected": len(results["ktp_result"])}
            return json.dumps(skill_output)
        elif len(results["selfie_result"]) == 0:
            skill_output = {"distance": "-1", "note": "selfie", "detected": len(results["selfie_result"])}
            return json.dumps(skill_output)
        elif len(results["selfie_result"]) == 1:
            skill_output = {"distance": "-1", "note": "selfie", "detected": len(results["selfie_result"])}
            return json.dumps(skill_output)

        # get results
        faces = self.get_results(imgs, results)

        # get face comparison
        distance = self.face_comparison(faces)
        skill_output = {"distance": str(distance), "note": "complete", "detected": len(faces)}
        return json.dumps(skill_output)

    def open_img(self, input_bn):
        img_pil = Image.open(io.BytesIO(input_bn))
        img_pil = ImageOps.exif_transpose(img_pil)
        img = np.array(img_pil)

        return img

    def preprocess(self, img_arr):
        size = 1280
        img = cv2.resize(img_arr, (size, size), interpolation=cv2.INTER_LINEAR)
        return img

    def get_results(self, imgs, results):
        # process ktp
        faces = list()
        detected_face = np.copy(imgs[0])
        x1, y1, width, height = results["ktp_result"][0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        detected_face = detected_face[y1:y2, x1:x2]
        detected_face = cv2.resize(detected_face, (1280, 1280), interpolation=cv2.INTER_LINEAR)
        faces.append(detected_face)

        # process selfie
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

        detected_face = np.copy(imgs[1])
        x1, y1, width, height = results["selfie_result"][i_result]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        detected_face = detected_face[y1:y2, x1:x2]
        detected_face = cv2.resize(detected_face, (1280, 1280), interpolation=cv2.INTER_LINEAR)
        faces.append(detected_face)

        return faces

    def face_comparison(self, faces):
        # kernel sharpening
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        faces[0] = cv2.filter2D(src=faces[0], ddepth=-1, kernel=kernel)
        faces[1] = cv2.filter2D(src=faces[1], ddepth=-1, kernel=kernel)

        # ArcFace
        emb1 = self.arc_face.calc_emb(faces[0])
        emb2 = self.arc_face.calc_emb(faces[1])
        distance = self.arc_face.get_distance_embeddings(emb1, emb2)

        # threshold = 1.6
        return str(distance)
