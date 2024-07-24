from model import SignLangModel, MODEL_SAVE_PATH  # Import the model
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_loaded = SignLangModel()
model_loaded.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model_loaded.to(device)
model_loaded.eval()

print(f"Loaded model:\n{model_loaded}")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

frame_width, frame_height = 640, 480  
box_width, box_height = 300, 300
box_top_left = (frame_width - box_width - 20, frame_height - box_height - 20)
box_bottom_right = (frame_width - 20, frame_height - 20)
box_color = (255, 255, 0)  # Cyan
box_thickness = 2

lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()


prediction_queue = deque(maxlen=5)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    cv2.rectangle(frame, box_top_left, box_bottom_right, box_color, box_thickness)

    
    roi = frame[box_top_left[1]:box_bottom_right[1], box_top_left[0]:box_bottom_right[0]]
    
    if roi.size > 0:
        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        image_tensor = transform(pil_image).unsqueeze(0).to(device)

        # Make prediction
        with torch.no_grad():
            output = model_loaded(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            predicted_label = predicted.item()
            confidence_score = confidence.item()

        # Convert label to alphabet (assuming labels are 0-25 for letters A-Z)
        alphabet = chr(predicted_label + ord('A'))

        # Append to prediction queue
        prediction_queue.append((alphabet, confidence_score))

        # Calculate average prediction and confidence
        avg_alphabet, avg_confidence = max(set(prediction_queue), key=prediction_queue.count)

        # prediction and confidence on the frame
        cv2.putText(frame, f'Predicted: {avg_alphabet}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Accuracy: {avg_confidence:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    
    cv2.imshow('Sign Language Recognition', frame)

    # Break the loop on 'b' key press
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break


cap.release()
cv2.destroyAllWindows()


