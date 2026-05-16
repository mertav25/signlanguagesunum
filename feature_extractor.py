import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os
import json
import time

model_yolo = YOLO('yolov8n-pose.pt')

def extract_keypoints_fast(video_path, target_frames=10):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sequence_data = []

    if total_frames <= 0:
        cap.release()
        return np.zeros((target_frames, 34))

    step = max(1, total_frames // target_frames)
    frame_count = 0
    collected_frames = 0

    while collected_frames < target_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (320, 320))
        results = model_yolo(frame_resized, verbose=False)

        if len(results[0].keypoints) > 0:
            keypoints = results[0].keypoints.xyn[0].cpu().numpy()
            flattened_keypoints = keypoints.flatten()
            sequence_data.append(flattened_keypoints)
        else:
            sequence_data.append(np.zeros(34))

        collected_frames += 1
        frame_count += step

    cap.release()

    while len(sequence_data) < target_frames:
        sequence_data.append(np.zeros(34))

    return np.array(sequence_data[:target_frames])

def process_wlasl_dataset(dataset_folder, json_path):
    X_data = []
    y_data = []
    
    with open(json_path, 'r') as f:
        wlasl_data = json.load(f)
        
    class_mapping = {}
    current_label_idx = 0
    
    total_videos = sum(len(entry['instances']) for entry in wlasl_data)
    processed_count = 0
    start_time = time.time()
        
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
                    
                features = extract_keypoints_fast(video_path, target_frames=10)
                X_data.append(features)
                y_data.append(class_mapping[gloss])
                
            processed_count += 1
            
            if processed_count % 10 == 0 or processed_count == total_videos:
                elapsed_time = time.time() - start_time
                progress = (processed_count / total_videos) * 100
                avg_time_per_video = elapsed_time / processed_count
                remaining_time = avg_time_per_video * (total_videos - processed_count)
                
                print(f"Progress: {progress:.2f}% | Processed: {processed_count}/{total_videos} | Elapsed: {elapsed_time:.0f}s | ETA: {remaining_time:.0f}s")
    
    with open('class_mapping.json', 'w') as f:
        json.dump(class_mapping, f)
        
    return np.array(X_data), np.array(y_data)

if __name__ == "__main__":
    dataset_path = "dataset"
    json_path = "WLASL_v0.3.json"
    
    print("Starting feature extraction...")
    X, y = process_wlasl_dataset(dataset_path, json_path)
    
    if len(X) > 0:
        np.save('X_data_final.npy', X)
        np.save('y_data_final.npy', y)
        print("Completed successfully.")