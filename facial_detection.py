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
    display = image.copy()
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(grayscale, 1)

    for (i, rect) in enumerate(rects):
      # Facial border
      (x, y, w, h) = rect_to_borders(rect)
      cv2.rectangle(display, (x, y), (x + w, y + h), (255, 255, 0), 2)

      # Facial landmark
      predicted = predictor(grayscale, rect)
      landmarks = np.zeros((68, 2), dtype='int')
      for i in range(0, 68):
          (x, y) = (predicted.part(i).x, predicted.part(i).y)
          landmarks[i] = (x, y)
      
      # Mark eyes
      left_eye = landmarks[36:42]
      left_eye_center = left_eye.mean(axis=0).astype('int')
      right_eye = landmarks[42:48]
      right_eye_center = right_eye.mean(axis=0).astype('int')
      cv2.circle(display, tuple(left_eye_center), 1, (0, 0, 255), -1)
      cv2.circle(display, tuple(right_eye_center), 1, (0, 255, 0), -1)


      # Get 32x32 eye image
      eye_width = 32
      eye_height = 32
      eye_left_x = left_eye_center[0] - eye_width // 2
      eye_left_y = left_eye_center[1] - eye_height // 2
      eye_right_x = right_eye_center[0] - eye_width // 2
      eye_right_y = right_eye_center[1] - eye_height // 2
      if eye_left_x < 0 or eye_left_y < 0 or eye_right_x < 0 or eye_right_y < 0:
        continue
      if eye_left_x + eye_width > image.shape[1] or eye_left_y + eye_height > image.shape[0] or eye_right_x + eye_width > image.shape[1] or eye_right_y + eye_height > image.shape[0]:
        continue

      eye_left = image[eye_left_y:eye_left_y + eye_height, eye_left_x:eye_left_x + eye_width]
      eye_right = image[eye_right_y:eye_right_y + eye_height, eye_right_x:eye_right_x + eye_width]
      cv2.putText(display, 'L', (eye_left_x, eye_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
      cv2.putText(display, 'R', (eye_right_x, eye_right_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

      # Show image
      eye_window = np.zeros((eye_height, eye_width * 2, 3), dtype='uint8')
      eye_window[0:eye_height, 0:eye_width] = eye_left
      eye_window[0:eye_height, eye_width:eye_width * 2] = eye_right
      display[0:eye_height, 0:eye_width * 2] = eye_window

    cv2.imshow('Press \'Q\' to quit', display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  video.release()
  cv2.destroyAllWindows()
