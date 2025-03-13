import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os


class ESRGAN_Tiny(nn.Module):
    def __init__(self):
        super(ESRGAN_Tiny, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ) for _ in range(4)]
        )
        # First upsampling: 128x128 -> 256x256
        self.upsample1 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        # Second upsampling: 256x256 -> 512x512
        self.upsample2 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.residual(x) + x
        x = self.relu(self.upsample1(x))
        x = self.upsample2(x)
        return x

# 2. Hàm load mô hình
def load_model(model_path, device):
    model = ESRGAN_Tiny().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def enhance_image(model, image_path, output_path, device):
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert("RGB")
    image = image.resize((128, 128), Image.BICUBIC)  # Resize giống như khi train

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        sr_image = model(image_tensor).squeeze(0).cpu()

    sr_image = transforms.ToPILImage()(torch.clamp(sr_image, 0, 1))
    sr_image.save(output_path)
    print(f"Saved enhanced image to: {output_path}")

if __name__ == "__main__":
    # 4. Thiết bị (GPU nếu có, nếu không thì CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 5. Load mô hình đã huấn luyện
    model_path = "/home/jupyter-iec_iot13_toanlm/video_enhance/up_scale/student_model_200ep.pth"
    model = load_model(model_path, device)
    print("Model loaded successfully.")

    input_folder = "/home/jupyter-iec_iot13_toanlm/video_enhance/up_scale/inputs"
    output_folder = "/home/jupyter-iec_iot13_toanlm/video_enhance/up_scale/results"
    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(input_folder):
        input_path = os.path.join(input_folder, img_file)
        output_path = os.path.join(output_folder, f"sr_{img_file}")
        if not os.path.isfile(input_path):  
            continue
        enhance_image(model, input_path, output_path, device)
