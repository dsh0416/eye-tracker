import cv2
import dlib
import imutils
import numpy as np

def rect_to_borders(rect):
	# Convert dlib rectangles to borders
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
	return (x, y, w, h)


if __name__ == '__main__':
  video = cv2.VideoCapture(0)
  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor('checkpoints/shape_predictor_68_face_landmarks.dat')

  while (True):
    ret, frame = video.read()

    # Preprocessing
    image = imutils.resize(frame, width=720)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(grayscale, 1)

    for (i, rect) in enumerate(rects):
      # Facial border
      (x, y, w, h) = rect_to_borders(rect)
      cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

      # Facial landmark
      predicted = predictor(grayscale, rect)
      landmarks = np.zeros((68, 2), dtype='int')
      for i in range(0, 68):
          (x, y) = (predicted.part(i).x, predicted.part(i).y)
          landmarks[i] = (x, y)
      
      # mark eyes
      left_eye = landmarks[36:42]
      right_eye = landmarks[42:48]

      for landmark in left_eye:
        cv2.circle(image, tuple(landmark), 1, (0, 0, 255), -1)
      cv2.putText(image, 'L', (left_eye[0][0], left_eye[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
      
      for landmark in right_eye:
        cv2.circle(image, tuple(landmark), 1, (0, 255, 0), -1)
      cv2.putText(image, 'R', (right_eye[0][0], right_eye[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Press \'Q\' to quit', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  video.release()
  cv2.destroyAllWindows()
