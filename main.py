import os
import sys
import cv2
import dlib
import numpy as np
import time
import threading
import glob
import traceback
from skimage import io

win = dlib.image_window()

face_recognition = dlib.face_recognition_model_v1('./dlib_face_recognition_resnet_model_v1.dat')
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

enrolled_faces = {}

# Semaphor for identifying thread
identifying = False

def face_to_vector(image, face):
  return (
    np
    .array(face_recognition.compute_face_descriptor(image, face))
    .astype(float)
  )

def faces_from_image(image):
  UPSAMPLING_FACTOR = 0
  faces = [
    (face.height() * face.width(), shape_predictor(image, face))
    for face in face_detector(image, UPSAMPLING_FACTOR)
  ]
  return [face for _, face in sorted(faces, reverse=True)]

def image_from_file(path):
  return io.imread(path)

def identify(image):
  # Get all faces
  faces = faces_from_image(image)

  def find_match(face):
    # Calculate face descriptor
    descriptor = face_recognition.compute_face_descriptor(image, face)
    face_vector = np.array(descriptor).astype(float)

    # THIS is probably hazardous as ordering may not be always the same?
    enroll_identifiers = np.array(list(enrolled_faces.keys()))
    enroll_matrix = np.array(list(enrolled_faces.values()))

    # Calculate differences between the face and all enrolled faces
    differences = np.subtract(np.array(enroll_matrix), face_vector)
    distances = np.linalg.norm(differences, axis=1)
    # and pick the closest one
    closest_index = np.argmin(distances)

    return enroll_identifiers[closest_index], distances[closest_index], face

  return map(find_match, faces)



def handle_frame(origFrame, cb):
  global identifying
  try:
    frame = cv2.resize(origFrame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    start = time.time()

    identified_matches = identify(frame)

    valid_matches = list(filter((lambda match: match[1] < 0.6), identified_matches))

    cb(valid_matches)

    sys.stdout.flush()

  except Exception as e:
    exc = e
    print(e)
    cb(None, 0, time.time() - start)
    # print(e)

  identifying = False

def webcam():
  global identifying

  video_capture = cv2.VideoCapture(0)

  while True:
    video_capture.grab()
    if (not identifying):
      ret, frame = video_capture.retrieve()

      if (ret == False):
        print('No frame')
        break
      identifying = True

      thread = threading.Thread(target=handle_frame, args=(frame, (lambda res: logger(res, frame))))
      thread.daemon=True
      thread.start()

  # When everything is done, release the capture
  video_capture.release()

def load_enrolled_faces():
  global enrolled_faces
  enrolled_faces = np.load('faces.npy').item()

def enroll_face(image, name):
  # find face
  faces = faces_from_image(image)
  # Pick largest face
  face = faces[0] if faces else None
  # face to vector
  face_vector = face_to_vector(image, face)
  # save to enrolled faces list
  enrolled_faces[name] = face_vector
  # save npy file

def logger(faces, frame):
  win.set_image(frame)
  if len(faces) > 0:
    win.clear_overlay()

  for i, (_, _, face_vector) in enumerate(faces):
    win.add_overlay(face_vector)

def enrollImages():
  print('Enrolling all images')
  for filename in glob.glob('faces/*.jpg'):
    name = filename.split('/')[1].split('.')[0]
    image = image_from_file(filename)
    enroll_face(image, name)
  np.save('faces.npy', enrolled_faces)
  print('Saved faces.npy')

if (not os.path.isfile('faces.npy')):
    enrollImages()
else:
  print('Loading enrolled faces from faces.npy')
  load_enrolled_faces()

webcam()
