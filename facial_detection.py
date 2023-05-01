import cv2
import dlib
import imutils

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
    image = imutils.resize(frame, width=500)
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(grayscale, 1)

    for (i, rect) in enumerate(rects):
      # Facial border
      (x, y, w, h) = rect_to_borders(rect)
      cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 0), 2)

      # Facial landmark
      landmarks = predictor(grayscale, rect)
      for i in range(0, 68):
          (x, y) = (landmarks.part(i).x, landmarks.part(i).y)
          cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    cv2.imshow('Press \'Q\' to quit', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

  video.release()
  cv2.destroyAllWindows()
