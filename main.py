import os
import argparse
from utils import (setup_model, create_output_dirs, extract_frames, 
                 upscale_frames, create_video, cleanup_frames)
import time
# def main():
#     s=time.time()

#     parser = argparse.ArgumentParser(description='Video Upscaling with RealESRGAN')
#     parser.add_argument('--input', type=str, required=True, help='Đường dẫn đến video đầu vào')
#     parser.add_argument('--output', type=str, default=None, help='Đường dẫn đến video đầu ra')
#     parser.add_argument('--scale', type=int, default=4, help='Hệ số scale (mặc định: 4)')
#     parser.add_argument('--model_path', type=str, default='weights/RealESRGAN_x4.pth', 
#                         help='Đường dẫn đến weights của model')
#     parser.add_argument('--keep_frames', action='store_true', 
#                         help='Giữ lại frames sau khi tạo video')
    
#     args = parser.parse_args()
    
#     if args.output is None:
#         filename = os.path.basename(args.input)
#         name, ext = os.path.splitext(filename)
#         args.output = f"videos_output/{name}_upscaled{ext}"
    
#     create_output_dirs()
    
#     print("Đang tải model RealESRGAN...")
#     model = setup_model(scale=args.scale, model_path=args.model_path)
    
#     print(f"Đang trích xuất frames từ {args.input}...")
#     fps, frame_count, original_height, original_width = extract_frames(args.input)
#     print(f"Đã trích xuất {frame_count} frames với fps = {fps}")
    
#     print("Đang upscale frames...")
#     upscale_frames(model)
    
#     print("Đang tạo video từ frames đã upscale...")
#     create_video(
#         args.output, 
#         fps=fps, 
#         scale=args.scale,
#         original_width=original_width,
#         original_height=original_height
#     )
    
#     cleanup_frames(keep_frames=args.keep_frames)
    
#     print(f"Hoàn thành! Video đã được upscale và lưu tại: {args.output}")
#     print(time.time()-s)
def main():
    s=time.time()
    parser = argparse.ArgumentParser(description='Video Upscaling with RealESRGAN')
    parser.add_argument('--input', type=str, required=True, help='Đường dẫn đến video đầu vào')
    parser.add_argument('--output', type=str, default=None, help='Đường dẫn đến video đầu ra')
    parser.add_argument('--scale', type=int, default=4, help='Hệ số scale (mặc định: 4)')
    parser.add_argument('--model_path', type=str, default='weights/RealESRGAN_x4.pth', 
                        help='Đường dẫn đến weights của model')
    parser.add_argument('--keep_frames', action='store_true', 
                        help='Giữ lại frames sau khi tạo video')
    parser.add_argument('--num_processes', type=int, default=None,
                        help='Số lượng processes sử dụng cho upscale (mặc định: số lõi CPU - 1)')
    
    args = parser.parse_args()
    
    if args.output is None:
        filename = os.path.basename(args.input)
        name, ext = os.path.splitext(filename)
        args.output = f"videos_output/{name}_upscaled{ext}"
    
    create_output_dirs()
    
    print("Đang tải model RealESRGAN...")
    model = setup_model(scale=args.scale, model_path=args.model_path)
    
    print(f"Đang trích xuất frames từ {args.input}...")
    fps, frame_count, original_height, original_width = extract_frames(args.input)
    print(f"Đã trích xuất {frame_count} frames với fps = {fps}")
    
    print("Đang upscale frames...")
    upscale_frames(model, num_processes=args.num_processes)
    
    print("Đang tạo video từ frames đã upscale...")
    create_video(
        args.output, 
        fps=fps, 
        scale=args.scale,
        original_width=original_width,
        original_height=original_height
    )
    
    cleanup_frames(keep_frames=args.keep_frames)
    print(time.time()-s)
    print(f"Hoàn thành! Video đã được upscale và lưu tại: {args.output}")    
if __name__ == '__main__':
    main()
