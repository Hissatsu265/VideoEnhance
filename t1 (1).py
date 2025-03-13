import os
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

from torch.nn import functional as F
import cv2
from huggingface_hub import hf_hub_url,hf_hub_download

from video_enhance.up_scale.RealESRGAN.rrdbnet_arch import RRDBNet
from video_enhance.up_scale.RealESRGAN.utils import pad_reflect, split_image_into_overlapping_patches, stich_together, \
                   unpad_image

 # export PYTHONPATH="${PYTHONPATH}:/home/jupyter-iec_iot13_toanlm/"

class OptimizedRealESRGAN(RealESRGAN):
    def __init__(self, device, scale=4):
        super().__init__(device, scale)
        # Bật tính năng tối ưu CUDA nếu có thể
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        
        # Chuẩn bị sẵn các tham số tối ưu
        self.default_batch_size = 16  # Tăng batch size nếu GPU mạnh
        self.default_patch_size = 128  # Giảm kích thước patch để tăng tốc
        self.default_padding = 10  # Giảm padding để tăng tốc, nhưng có thể giảm chất lượng
    
    @torch.cuda.amp.autocast()
    def predict(self, lr_image, batch_size=None, patches_size=None, padding=None, pad_size=10):
        """
        Phiên bản tối ưu của phương thức predict
        """
        # Sử dụng giá trị mặc định nếu không được chỉ định
        batch_size = batch_size or self.default_batch_size
        patches_size = patches_size or self.default_patch_size
        padding = padding or self.default_padding
        
        # Chuyển đổi ảnh đầu vào
        if isinstance(lr_image, Image.Image):
            lr_image = np.array(lr_image)
        
        # Phân tích kích thước ảnh để tối ưu
        h, w, _ = lr_image.shape
        
        # Tối ưu kích thước patch dựa trên kích thước ảnh
        if h * w > 1024 * 1024:  # Ảnh lớn hơn 1MP
            patches_size = max(64, patches_size // 2)  # Giảm kích thước patch
        elif h * w < 256 * 256:  # Ảnh nhỏ
            # Với ảnh nhỏ, có thể xử lý toàn bộ ảnh mà không cần chia patch
            patches_size = max(h, w)
            padding = 0
        
        # Đệm ảnh bằng phản chiếu
        lr_image = pad_reflect(lr_image, pad_size)
        
        # Chia ảnh thành các patch chồng lấp
        patches, p_shape = split_image_into_overlapping_patches(
            lr_image, patch_size=patches_size, padding_size=padding
        )
        
        # Chuyển patches sang tensor
        total_patches = patches.shape[0]
        img = torch.FloatTensor(patches/255).permute((0,3,1,2)).to(self.device)
        
        # Tối ưu batch_size dựa trên số lượng patch
        batch_size = min(batch_size, total_patches)
        
        # Xử lý inference
        with torch.no_grad():
            # Sử dụng mixed precision để tăng tốc (nếu GPU hỗ trợ)
            if self.device.type == 'cuda' and torch.cuda.get_device_capability(self.device)[0] >= 7:
                # Sử dụng autocast để tăng tốc độ inference
                with torch.cuda.amp.autocast():
                    res = self.model(img[0:batch_size])
                    for i in range(batch_size, total_patches, batch_size):
                        end_idx = min(i + batch_size, total_patches)
                        res = torch.cat((res, self.model(img[i:end_idx])), 0)
            else:
                # Không có autocast cho các GPU cũ hơn hoặc CPU
                res = self.model(img[0:batch_size])
                for i in range(batch_size, total_patches, batch_size):
                    end_idx = min(i + batch_size, total_patches)
                    res = torch.cat((res, self.model(img[i:end_idx])), 0)
        
        # Chuyển đổi kết quả về numpy
        sr_image = res.permute((0,2,3,1)).clamp(0, 1).cpu()
        np_sr_image = sr_image.numpy()
        
        # Tính toán kích thước ảnh kết quả
        scale = self.scale
        padded_size_scaled = tuple(np.multiply(p_shape[0:2], scale)) + (3,)
        scaled_image_shape = tuple(np.multiply(lr_image.shape[0:2], scale)) + (3,)
        
        # Ghép các patch lại
        np_sr_image = stich_together(
            np_sr_image, 
            padded_image_shape=padded_size_scaled,
            target_shape=scaled_image_shape, 
            padding_size=padding * scale
        )
        
        # Chuyển đổi về định dạng ảnh
        sr_img = (np_sr_image*255).astype(np.uint8)
        sr_img = unpad_image(sr_img, pad_size*scale)
        sr_img = Image.fromarray(sr_img)
        
        return sr_img

def main() -> int:
    # Bắt đầu đo thời gian
    import time
    start_time = time.time()
    
    # Thiết lập thiết bị với các tối ưu CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # Khởi tạo GPU trước khi bắt đầu xử lý chính
        torch.cuda.empty_cache()
        # Tối ưu hóa việc phân bổ bộ nhớ CUDA
        torch.cuda.set_per_process_memory_fraction(0.85)  # Sử dụng 85% GPU memory
    else:
        device = torch.device('cpu')
        # Tối ưu cài đặt PyTorch cho CPU
        torch.set_num_threads(max(4, os.cpu_count()))
    
    print(f"Sử dụng thiết bị: {device}")
    
    # Tạo thư mục kết quả nếu chưa tồn tại
    os.makedirs("results", exist_ok=True)
    
    # Tải và khởi tạo model được tối ưu hóa
    model = OptimizedRealESRGAN(device, scale=4)
    model.load_weights('weights/RealESRGAN_x4.pth', download=True)
    
    # Xử lý từng ảnh
    input_files = [f for f in os.listdir("inputs") if os.path.isfile(os.path.join("inputs", f))]
    
    for i, image_file in enumerate(input_files):
        image_path = os.path.join("inputs", image_file)
        if not os.path.isfile(image_path):
            continue
            
        image_start_time = time.time()
        
        # Đọc ảnh
        image = Image.open(image_path).convert('RGB')
        
        # Thực hiện upscale
        sr_image = model.predict(image)
        
        # Lưu kết quả
        sr_image.save(f'results/{i}.png')
        
        image_time = time.time() - image_start_time
        print(f"{i}. Đã xử lí {image_file} trong {image_time:.3f} giây")
    
    total_time = time.time() - start_time
    print(f"Tổng thời gian: {total_time:.3f} giây")
    
    return 0

if __name__ == '__main__':
    main()