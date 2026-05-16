import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
import json
import os
import random
from sklearn.model_selection import train_test_split

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=3):
        super(SignLanguageLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def get_unseen_test_videos(dataset_folder, json_path):
    with open(json_path, 'r') as f:
        wlasl_data = json.load(f)
        
    video_paths = []
    y_labels = []
    class_mapping = {}
    current_label_idx = 0
    
    for entry in wlasl_data:
        gloss = entry['gloss']
        for instance in entry['instances']:
            video_id = str(instance['video_id'])
            video_path = None
            for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                temp_path = os.path.join(dataset_folder, video_id + ext)
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break
            
            if video_path is not None:
                if gloss not in class_mapping:
                    class_mapping[gloss] = current_label_idx
                    current_label_idx += 1
                video_paths.append(video_path)
                y_labels.append(class_mapping[gloss])
                
    _, test_paths, _, _ = train_test_split(video_paths, y_labels, test_size=0.2, stratify=y_labels, random_state=42)
    return test_paths

def get_video_prediction(video_path, yolo_model, lstm_model, idx_to_class, device):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sequence_data = []
    
    if total_frames <= 0:
        cap.release()
        return "..."
        
    target_frames = 10
    step = max(1, total_frames // target_frames)
    frame_count = 0
    collected_frames = 0
    
    while collected_frames < target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_resized = cv2.resize(frame, (320, 320))
        results = yolo_model(frame_resized, verbose=False)
        
        if len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints.xyn[0].cpu().numpy().flatten()
            sequence_data.append(keypoints)
        else:
            sequence_data.append(np.zeros(34))
            
        collected_frames += 1
        frame_count += step
        
    cap.release()
    
    while len(sequence_data) < target_frames:
        sequence_data.append(np.zeros(34))
        
    input_tensor = torch.tensor(np.array(sequence_data), dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = lstm_model(input_tensor)
        _, predicted = torch.max(outputs.data, 1)
        return idx_to_class[predicted.item()]

def run_demo(video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open('class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
        
    idx_to_class = {v: k for k, v in class_mapping.items()}
    num_classes = len(idx_to_class)
    
    yolo_model = YOLO('yolov8n-pose.pt')
    lstm_model = SignLanguageLSTM(input_size=34, hidden_size=256, num_classes=num_classes).to(device)
    
    lstm_model.load_state_dict(torch.load('sign_language_model.pth', map_location=device))
    lstm_model.eval()
    
    final_prediction = get_video_prediction(video_path, yolo_model, lstm_model, idx_to_class, device)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 25
    delay = int(1000 / fps)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        cv2.putText(frame, f"Prediction: {final_prediction}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        display_frame = cv2.resize(frame, (800, 600))
        cv2.imshow('UNSEEN Test Video Demo', display_frame)
        
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    dataset_path = "dataset"
    json_path = "WLASL_v0.3.json"
    
    print("Finding unseen test videos...")
    test_videos = get_unseen_test_videos(dataset_path, json_path)
    print(f"Found {len(test_videos)} videos that were NOT used in training.")
    
    random_test_video = random.choice(test_videos)
    print(f"Selected random UNSEEN video: {random_test_video}")
    print("Analyzing video...")
    
    run_demo(random_test_video)