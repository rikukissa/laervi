# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring

import dlib

face_recognizer = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')
shape_predictor = dlib.shape_predictor('./models/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

def faces_from_image(image):
  upsampling_factor = 0
  faces = [
      (face.height() * face.width(), shape_predictor(image, face))
      for face in face_detector(image, upsampling_factor)
  ]
  return [face for _, face in sorted(faces, reverse=True)]

