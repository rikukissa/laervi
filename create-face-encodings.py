# -*- coding: utf-8 -*-
# pylint: disable=missing-docstring

import glob
import numpy as np
from skimage import io

import utils

def face_to_vector(image, face):
  return (
      np
      .array(utils.face_recognizer.compute_face_descriptor(image, face))
      .astype(float)
  )


def create_encoding(image):
  # find face
  faces = utils.faces_from_image(image)
  # Pick largest face
  face = faces[0] if faces else None
  # face to vector
  face_vector = face_to_vector(image, face)
  return face_vector

def main():
  files = glob.glob('faces/*.jpg')

  encodings = {}

  for i, filename in enumerate(files):
    identifier = filename.split('/')[1].split('.')[0]
    try:
      image = io.imread(filename)
      encodings[identifier] = create_encoding(image)
      print(i, "/", len(files), "Created encoding for", filename)
    except:
      utils.eprint(i, "/", len(files), "Could not create an encoding for", filename)
      pass

  np.save('encodings.npy', encodings)

if __name__ == "__main__":
  main()
