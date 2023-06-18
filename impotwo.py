import cv2
import mediapipe as mp
import numpy as np

# Load the FaceMesh model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Load the face mask image with alpha channel
mask_image = cv2.imread('masskkkyy.png', cv2.IMREAD_UNCHANGED)

# Define a dictionary to store face mask overlays by face ID
face_overlays = {}

# Function to overlay the face mask on a single face
def overlay_mask_on_face(image, mask_image, landmarks):
    image_height, image_width, _ = image.shape
    mask_height, mask_width = mask_image.shape[:2]

    # Convert landmarks to pixel coordinates
    coords_x = [int(lm.x * image_width) for lm in landmarks]
    coords_y = [int(lm.y * image_height) for lm in landmarks]

    # Find the bounding box of the face region
    face_left = min(coords_x)
    face_top = min(coords_y)
    face_right = max(coords_x)
    face_bottom = max(coords_y)

    # Resize the mask image to fit the face region
    face_width = face_right - face_left
    face_height = face_bottom - face_top
    resized_mask = cv2.resize(mask_image, (face_width, face_height))

    # Create a mask from the alpha channel if it exists
    if mask_image.shape[2] == 4:
        mask = resized_mask[:, :, 3] / 255.0
        mask = np.expand_dims(mask, axis=2)
        resized_mask = resized_mask[:, :, :3]
    else:
        mask = np.ones((face_height, face_width, 1))

    # Adjust the region of interest in the original image
    roi = image[face_top:face_bottom, face_left:face_right]

    # Blend the mask and the region of interest
    blended_roi = (mask * resized_mask) + ((1 - mask) * roi)

    # Place the blended region of interest back into the original image
    image[face_top:face_bottom, face_left:face_right] = blended_roi

    return image

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB for processing with MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect the face landmarks using FaceMesh
    results = face_mesh.process(frame_rgb)
    landmarks = results.multi_face_landmarks

    # Clear the face overlays dictionary
    face_overlays.clear()

    if landmarks:
        # Overlay the face mask on each detected face
        for face_id, face_landmarks in enumerate(landmarks):
            # Check if the face ID already exists in the dictionary
            if face_id not in face_overlays:
                # Overlay the face mask on the current face
                face_overlay = overlay_mask_on_face(frame, mask_image, face_landmarks.landmark)
                # Store the face mask overlay information for the current face ID
                face_overlays[face_id] = face_overlay
            else:
                # Retrieve the stored face mask overlay for the current face ID
                face_overlay = face_overlays[face_id]

            # Display the face mask overlay for the current face ID
            cv2.imshow(f'Face ID: {face_id}', face_overlay)

    # Display the resulting frame
    cv2.imshow('Face Mask Overlay', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()

