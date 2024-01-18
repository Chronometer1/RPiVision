import cv2
import torch
import time
from torchvision import models, transforms

# Load the model
model = torch.hub.load('pytorch/vision:v0.9.0', 'densenet121', weights=True)
model.eval()

# Move the model to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Preprocess the image
    input_tensor = transform(frame).unsqueeze(0)

    # Move the input tensor to the same device as the model
    input_tensor = input_tensor.to(device)

    # Make predictions
    with torch.no_grad():
        output = model(input_tensor)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    result = {}
    for i in range(top5_prob.size(0)):
        result[categories[top5_catid[i]]] = top5_prob[i].item()

    # Process the output as needed
    _, predicted_idx = torch.max(output, 1)
    print("Predicted class:", result)

    time.sleep(.1)

    # Display the original frame
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()