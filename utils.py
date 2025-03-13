import os
import cv2
import torch
import numpy as np
from PIL import Image
from RealESRGAN import RealESRGAN
import os
import multiprocessing
from PIL import Image
from functools import partial
    
# import time
def setup_model(scale=4, model_path='weights/RealESRGAN_x4.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=scale)
    model.load_weights(model_path, download=True)
    return model

def create_output_dirs(output_frame_dir='frames_output', output_video_dir='videos_output'):
    os.makedirs(output_frame_dir, exist_ok=True)
    os.makedirs(output_video_dir, exist_ok=True)

def extract_frames(video_path, output_dir='frames_input'):
    os.makedirs(output_dir, exist_ok=True)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    success, frame = video.read()
    while success:
        cv2.imwrite(f"{output_dir}/frame_{frame_count:06d}.png", frame)
        frame_count += 1
        success, frame = video.read()
        # if frame_count>200: break
    video.release()
    return fps, frame_count, height, width
def process_single_frame(frame_name, model, input_dir, output_dir, total_frames, frame_idx):
        frame_path = os.path.join(input_dir, frame_name)
        output_path = os.path.join(output_dir, frame_name)
        
        # Kiểm tra nếu frame đã được xử lý
        if os.path.exists(output_path):
            print(f"Đã tồn tại: {frame_idx+1}/{total_frames} frames ({(frame_idx+1)/total_frames*100:.2f}%) - Frame: {frame_name}")
            return
        
        # Xử lý frame
        try:
            image = Image.open(frame_path).convert('RGB')
            sr_image = model.predict(image)
            sr_image.save(output_path)
            print(f"Upscale: {frame_idx+1}/{total_frames} frames ({(frame_idx+1)/total_frames*100:.2f}%) - Frame: {frame_name}")
        except Exception as e:
            print(f"Lỗi khi xử lý {frame_name}: {e}")
            
def upscale_frames(model, input_dir='frames_input', output_dir='frames_output', num_processes=None):
    
    os.makedirs(output_dir, exist_ok=True)
    
    frames = sorted(os.listdir(input_dir))
    total_frames = len(frames)
    
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Sử dụng {num_processes} processes để upscale {total_frames} frames")
    
    frame_args = [(frames[i], model, input_dir, output_dir, total_frames, i) for i in range(total_frames)]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        pool.starmap(process_single_frame, frame_args)

# def upscale_frames(model, input_dir='frames_input', output_dir='frames_output'):
#     os.makedirs(output_dir, exist_ok=True)

#     frames = sorted(os.listdir(input_dir))
#     total_frames = len(frames)
    
#     for i, frame_name in enumerate(frames):
        
#         frame_path = os.path.join(input_dir, frame_name)
#         if not os.path.isfile(frame_path):
#             continue
            
#         image = Image.open(frame_path).convert('RGB')
#         sr_image = model.predict(image)
#         sr_image.save(os.path.join(output_dir, frame_name))
#         if (i>5): break
#         # Hiển thị tiến trình
#         print(f"Upscale: {i+1}/{total_frames} frames ({(i+1)/total_frames*100:.2f}%) - Frame: {frame_name}")

def create_video(output_video_path, frames_dir='frames_output', fps=30, scale=4, original_width=None, original_height=None):
    frames = sorted(os.listdir(frames_dir))
    if not frames:
        print("Không tìm thấy frames để tạo video!")
        return
    
    frame = cv2.imread(os.path.join(frames_dir, frames[0]))
    if frame is None:
        return
    height, width, _ = frame.shape
    
    if original_width and original_height:
        width = original_width * scale
        height = original_height * scale
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for frame_name in frames:
        frame_path = os.path.join(frames_dir, frame_name)
        frame = cv2.imread(frame_path)
        
        # Resize nếu cần
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
            
        video.write(frame)
    
    video.release()
    print(f"Video đã được tạo: {output_video_path}")

def cleanup_frames(frames_dir='frames_input', upscaled_frames_dir='frames_output', keep_frames=False):
 
    if not keep_frames:
        for directory in [frames_dir, upscaled_frames_dir]:
            if os.path.exists(directory):
                for file in os.listdir(directory):
                    file_path = os.path.join(directory, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                print(f"Đã xóa tất cả các frames trong {directory}")