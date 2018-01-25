# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring

import os
import sys
import time
import threading
import cv2
import numpy as np
import dlib
import json
import utils

win = dlib.image_window()

enrolled_faces = {}

# Semaphor for IDENTIFYING thread
IDENTIFYING = False

def identify(image):
  # Get all faces
  faces = utils.faces_from_image(image)

  def find_match(face):
    # Calculate face descriptor
    descriptor = utils.face_recognizer.compute_face_descriptor(image, face)
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
  global IDENTIFYING
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

  IDENTIFYING = False

def webcam():
  global IDENTIFYING

  video_capture = cv2.VideoCapture(0)

  while True:
    video_capture.grab()
    if (not IDENTIFYING):
      ret, frame = video_capture.retrieve()
      if (not ret):
        raise Exception('No frame received!')

      IDENTIFYING = True

      thread = threading.Thread(target=handle_frame, args=(frame, (lambda res: logger(res, frame))))
      thread.daemon=True
      thread.start()

  # When everything is done, release the capture
  video_capture.release()


def logger(matches, frame):
  win.set_image(frame)
  if len(matches) > 0:
    win.clear_overlay()

  for _, (_, _, face_vector) in enumerate(matches):
    win.add_overlay(face_vector)

  def match_to_json(match):
    return { 'id': match[0], 'confidence': match[1] }

  print(json.dumps(list(map(match_to_json, matches))))

def load_enrolled_faces():
  global enrolled_faces
  enrolled_faces = np.load('encodings.npy').item()

if not os.path.isfile('encodings.npy'):
  utils.eprint("No encodings.npy file found! Create it by running create-face-encodings")
else:
  load_enrolled_faces()
  webcam()
