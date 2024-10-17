import cv2
import numpy as np

# Threshold to detect object and NMS threshold
thres = 0.45
nms_threshold = 0.2

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Load class names from the COCO dataset
classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Paths to model configuration and weights
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

# Load the DNN model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()

    if not success:
        break

    # Perform object detection
    classIds, confs, bbox = net.detect(img, confThreshold=thres)

    if len(classIds) != 0:
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        # Apply Non-Max Suppression
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        if len(indices) > 0:
            for i in indices:
                if isinstance(i, (list, np.ndarray)):
                    i = i[0]

                box = bbox[i]
                x, y, w, h = box[0], box[1], box[2], box[3]
                cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

                # Handling scalar case for classIds[i]
                if isinstance(classIds[i], np.ndarray):
                    classId = classIds[i][0]
                else:
                    classId = classIds[i]

                cv2.putText(img, classNames[classId - 1].upper(),
                            (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Output", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
