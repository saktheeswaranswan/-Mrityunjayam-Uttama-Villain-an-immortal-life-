import cv2
import numpy as np

# Load YOLO model and coco.names
net = cv2.dnn.readNet('face-yolov3-tiny_41000.weights', 'face-yolov3-tiny.cfg')
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()

# Load custom transparent PNG image to stitch
stitch_image = cv2.imread('fff.png', cv2.IMREAD_UNCHANGED)

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam, or provide the video file path

# Set default window size
window_width = 800
window_height = 600

# Create resizable window
cv2.namedWindow('Face Stitching', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Face Stitching', window_width, window_height)

while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to match window size
    frame = cv2.resize(frame, (window_width, window_height))

    # Perform object detection using YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers_names)

    # Process detected objects using Non-Maximum Suppression
    boxes = []
    confidences = []
    class_ids = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == 0:  # 0 represents the 'person' class
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                # Calculate bounding box coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to the detected boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Apply stitching on detected faces
    for i in indices:
        i = i
        x, y, w, h = boxes[i]

        # Ensure the face region has non-empty dimensions
        if w > 0 and h > 0:
            # Resize the stitch image to match the face region
            stitch_resized = cv2.resize(stitch_image, (w, h))

            # Extract the alpha channel from the stitch image
            stitch_alpha = stitch_resized[:, :, 3] / 255.0

            # Create a mask from the alpha channel
            mask = np.stack([stitch_alpha] * 3, axis=2)

            # Invert the mask
            inv_mask = 1 - mask

            # Adjust the stitch image size if it does not match the face region
            if stitch_resized.shape[0] != h or stitch_resized.shape[1] != w:
                stitch_resized = cv2.resize(stitch_resized, (w, h))

            # Resize the frame region to match the stitch image
            face_region_resized = cv2.resize(frame[y:y+h, x:x+w], (w, h))

            # Apply the stitch image on the face region using the mask
            stitched_face = cv2.multiply(face_region_resized.astype(np.float32), inv_mask, dtype=cv2.CV_32F)
            stitched_image = cv2.multiply(stitch_resized[:, :, :3].astype(np.float32), mask, dtype=cv2.CV_32F)
            stitch_applied = cv2.add(stitched_face, stitched_image).astype(np.uint8)

            # Replace the face region in the original frame with the stitched region
            frame[y:y+h, x:x+w] = stitch_applied

    # Display the frame
    cv2.imshow('Face Stitching', frame)

    # Check for key press
    key = cv2.waitKey(1) & 0xFF

    # Resize window if 'r' is pressed
    if key == ord('r'):
        window_width += 100
        window_height += 100
        cv2.resizeWindow('Face Stitching', window_width, window_height)

    # Exit if 'q' is pressed
    if key == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()

