import numpy as np
import cv2
import torch
import torchvision
import argparse
import time
import utils

"""construct the argument parser"""
""" info : --input : the path to the video clip
            -- clip_len : jumlah frame yang dipertimbangkan untuk tiap forward pass
            menggunakan lebih banyak frame akan meningkatkan prediksi namuan akan menurunkan FPS
            dan begitu sebaliknya"""
                      
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-c', '--clip-len', dest='clip_len', default=16, type=int,help='jumlah frame yang perlu dipertimbangkan tiap prediksi')
args = vars(parser.parse_args())
print(f"jumlah frame yang perlu dipertimbangkan tiap prediksi: {args['clip_len']}")

""" Prepare the Classes Names and The ResNet 3D model"""
class_names = utils.class_names                                          # mendapatkan class names langsung dari utils.py                                  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # mengaktifkan device yang tersedia
model = torchvision.models.video.r3d_18(pretrained=True, progress=True)  # load model ResNet 3D
model = model.eval().to(device)                                          # load the model onto the computation device

"""Preparing the Video Capture Object and Some Preliminary Code"""
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('error saat membaca video, silahkan check kembali lokasi video')
frame_width = int(cap.get(3))       #untuk mendapatkan lebar dari frame
frame_height = int(cap.get(4))      #untuk mendapatkan tinggi dari frame

save_name = f"{args['input'].split('/')[-1].split('.')[0]}" 
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))            #mendefiniskan nama dari video setalh video model meberi keputusan

frame_count = 0     # menghitung total jumlah frame
total_fps = 0       # mendapatkan nilai akhir dari FPS
clips = []          # a clips list to append and store the individual frames

"""Inferencing on the Video Using ResNet 3D Naural Network"""
# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        image = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = utils.transform(image=frame)['image']
        clips.append(frame)
        if len(clips) == args['clip_len']:
            with torch.no_grad(): # we do not want to backprop any gradients
                input_frames = np.array(clips)
                # add an extra dimension        
                input_frames = np.expand_dims(input_frames, axis=0)
                # transpose to get [1, 3, num_clips, height, width]
                input_frames = np.transpose(input_frames, (0, 4, 1, 2, 3))
                # convert the frames to tensor
                input_frames = torch.tensor(input_frames, dtype=torch.float32)
                input_frames = input_frames.to(device)
                # forward pass to get the predictions
                outputs = model(input_frames)
                # get the prediction index
                _, preds = torch.max(outputs.data, 1)
                
                # map predictions to the respective class names
                label = class_names[preds].strip()
            # get the end time
            end_time = time.time()
            # get the fps
            fps = 1 / (end_time - start_time)
            # add fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            wait_time = max(1, int(fps/4))
            cv2.putText(image, label, (15, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, 
                        lineType=cv2.LINE_AA)
            clips.pop(0)
            cv2.imshow('image', image)
            out.write(image)
            # press `q` to exit
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
    else:
        break

""" Release the VideoCapture Object and Calculate the Avarage FPS"""
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")