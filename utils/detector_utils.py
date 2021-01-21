import numpy as np
import tensorflow as tf
from threading import Thread
import cv2
from utils import label_map_util
from pygame import mixer

mixer.init()
sound = mixer.Sound('alarm.wav')


detection_graph = tf.Graph()
TRAINED_MODEL_DIR = 'my_model'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = TRAINED_MODEL_DIR + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = TRAINED_MODEL_DIR + '/labelmap.pbtxt'

NUM_CLASSES = 4
# load label map using utils provided by tensorflow object detection api
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

a = b = 0
score=0
# Load a frozen infrerence graph into memory
def load_inference_graph():
    # load frozen tensorflow model into memory

    print("> ====== Loading frozen graph into memory")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.Session(graph=detection_graph)
    print(">  ====== Inference graph loaded.")
    return detection_graph, sess

# Actual detection .. generate scores and bounding boxes given an image
def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
         detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def draw_box_on_image(num_obj_detect, score_thresh, scores, boxes, classes, im_width, im_height, image_np):
    # Determined using a piece of paper of known length, code can be found in distance to camera
    focalLength = 875
    # added an inch since thumb is not included
    avg_width = 4.0
    # To more easily differetiate distances and detected bboxes

    global a, b ,score
    color = None
    color0 = (0,255,0)
    color1 = (255, 0, 0)
    for i in range(num_obj_detect):

        if (scores[i] > score_thresh):

            if classes[i] == 1:
                id = 'open'
            elif classes[i] == 2:
                id = 'close'

            elif classes[i] == 3:
                id = 'yawn'
            elif classes[i] == 4:
                id = 'no yawn'

            if i == 1:
                color = color0
            elif i==2:
                color = color1
            elif i==3:
                color=color1
            else:
                color=color0

            if (classes[i] == 2 and classes[i] == 2):
                score = score + 1
            else:
                score = score - 1

            if (score < 0):
                score = 0
            if (score > 10):
                # person is feeling sleepy so we beep the alarm

                try:
                    sound.play()

                except:
                    pass

            (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                          boxes[i][0] * im_height, boxes[i][2] * im_height)
            p1 = (int(left), int(top))
            p2 = (int(right), int(bottom))

            dist = distance_to_camera(avg_width, focalLength, int(right - left))

            if dist:
                cv2.rectangle(image_np, p1, p2, color, 3, 1)

                cv2.putText(image_np,'' + str(i) + ': ' + id, (int(left), int(top) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color,1, 2)

                cv2.putText(image_np, '' + str("{0:.2f}".format(scores[i])),
                            (int(left), int(top) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.putText(image_np, 'distance from camera: ' + str("{0:.2f}".format(dist) + ' inches'),
                            (int(im_width * 0.65), int(im_height * 0.9 + 30 * i)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)

    return a, b


# Show fps value on image.
def draw_text_on_image(fps, image_np):
    cv2.putText(image_np, fps, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (77, 255, 9), 2)


# compute and return the distance from the hand to the camera using triangle similarity
def distance_to_camera(knownWidth, focalLength, pixelWidth):
    return (knownWidth * focalLength) / pixelWidth




class WebcamVideoStream:
	def __init__(self, src=0, name="WebcamVideoStream"):
		# initialize the video camera stream and read the first frame
		# from the stream
		self.stream = cv2.VideoCapture(src)
		(self.grabbed, self.frame) = self.stream.read()

		# initialize the thread name
		self.name = name

		# initialize the variable used to indicate if the thread should
		# be stopped
		self.stopped = False

	def start(self):
		# start the thread to read frames from the video stream
		t = Thread(target=self.update, name=self.name, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		# keep looping infinitely until the thread is stopped
		while True:
			# if the thread indicator variable is set, stop the thread
			if self.stopped:
				return

			# otherwise, read the next frame from the stream
			(self.grabbed, self.frame) = self.stream.read()

	def read(self):
		# return the frame most recently read
		return self.frame

	def stop(self):
		# indicate that the thread should be stopped
		self.stopped = True


class VideoStream:
	def __init__(self, src=0, usePiCamera=False, resolution=(320, 240),
		framerate=32, **kwargs):
		# check to see if the picamera module should be used
		if usePiCamera:
			print()

		# otherwise, we are using OpenCV so initialize the webcam
		# stream
		else:
			self.stream = WebcamVideoStream(src=src)

	def start(self):
		# start the threaded video stream
		return self.stream.start()

	def update(self):
		# grab the next frame from the stream
		self.stream.update()

	def read(self):
		# return the current frame
		return self.stream.read()

	def stop(self):
		# stop the thread and release any resources
		self.stream.stop()