import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm

# Dataset preparation

class My_Dataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image

transform = transforms.Compose([
    transforms.Resize((256, 256)), 
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


img_dir = '/data/raml/group1/Arbeit-chinna/RAML/Dataset'
dataset = My_Dataset(img_dir, transform=transform)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)




def PSNR(x, gt):
    return 10 * torch.log10(torch.max(x) / torch.mean((x - gt) ** 2))


class GaussianFourierFeature(nn.Module):
    def __init__(self, in_features, mapping_size, scale):
        super(GaussianFourierFeature, self).__init__()
        
        self.B = nn.Parameter(scale * torch.randn(mapping_size, in_features), requires_grad=False)
   
    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B.t()
        return torch.concat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierFeatureNetwork(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, mapping_size, scale):
        super(FourierFeatureNetwork, self).__init__()
        self.f_tr = GaussianFourierFeature(in_features, mapping_size, scale)
        self.fc1 = nn.Linear(2 * mapping_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.f_tr(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
# 3rd Complex Gerbor with WIRE   
class ComplexGaborLayer2D(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega0=10.0, sigma0=10.0,
                 trainable=False):
        super().__init__()
        self.omega_0 = omega0
        self.scale_0 = sigma0
        self.is_first = is_first
        
        self.in_features = in_features
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        self.omega_0 = nn.Parameter(self.omega_0*torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0*torch.ones(1), trainable)
        
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        self.scale_orth = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
    
    def forward(self, input):
        lin = self.linear(input)
        scale_x = lin
        scale_y = self.scale_orth(input)
        freq_term = torch.exp(1j*self.omega_0*lin)
        arg = scale_x.abs().square() + scale_y.abs().square()
        gauss_term = torch.exp(-self.scale_0*self.scale_0*arg)
        return freq_term*gauss_term


class WIRE_cmplx(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 first_omega_0=10, hidden_omega_0=10., scale=10.0):
        super().__init__()
        
        self.nonlin = ComplexGaborLayer2D
        hidden_features = int(hidden_features/2)  # Reduce hidden features for complex numbers
        
        self.net = []
        self.net.append(self.nonlin(in_features, hidden_features, 
                                    omega0=first_omega_0, sigma0=scale, 
                                    is_first=True, trainable=False))

        for _ in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features, 
                                        omega0=hidden_omega_0, sigma0=scale))

        final_linear = nn.Linear(hidden_features, out_features, dtype=torch.cfloat)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        output = self.net(coords)
        return output.real
    
# Training loop
class Deblurring_training():
    def __init__(self, y, net = None,  method_used = str , num_samples=2000, alpha=0.02, lr=0.01 ,**kwargs):
        self.y = y
        self.net = net
        self.method_used = method_used
        self.num_samples = num_samples
        self.alpha = alpha
        self.lr = lr
        self.kwargs = kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.height, self.width = y.shape[2], y.shape[3]
        #self.x_torch = torch.stack(torch.meshgrid(torch.linspace(0, 1, self.height), torch.linspace(0, 1, self.width)), dim=-1).view(-1, 2).to(self.device)
        H, W = self.height, self.width
        a = torch.linspace(-1, 1, W).to(self.device)
        b = torch.linspace(-1, 1, H).to(self.device)

        X, Y = torch.meshgrid(a, b, indexing='xy')
        self.x_torch = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))
        
        self.y = self.y.to(self.device)
        if net == None:
            self.model = None
        else:
            self.model = self.net(**self.kwargs).to(self.device)
        
        self.Gaussian_kernel = self.create_gaussian_kernel().to(self.device)
        self.y_blurry_noisy = self.create_blurry_noisy_image()
        
        self.u_1 = self.y_blurry_noisy.clone().to(self.device) 
        
        if self.method_used == "Explicit":
            self.optimizer = torch.optim.Adam([self.u_1], lr=self.lr)
        else:       
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda x: 0.1**min(x/self.num_samples, 1))
        
        
        self.diff_kernel_x = torch.tensor([[-1, 1]], dtype=torch.float).view(1, 1, 1, 2).repeat(3, 1, 1, 1).to(self.device)
        self.diff_kernel_y = torch.tensor([[-1], [1]], dtype=torch.float).view(1, 1, 2, 1).repeat(3, 1, 1, 1).to(self.device)
        
        
    # Gaussian Blur 
    def create_gaussian_kernel(self, kernel_size=21, sigma=2, channels=3):
        a = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(a, a, indexing='ij')
        kernel = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        kernel = kernel / torch.sum(kernel)
        
        gaussian_kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        
        return gaussian_kernel
    
    def create_blurry_noisy_image(self):
        y_blurry = F.conv2d(self.y, self.Gaussian_kernel, groups=3, padding=10)
        y_blurry_noisy = y_blurry + torch.randn_like(y_blurry) * 0.05
        
        return y_blurry_noisy.to(self.device)

    
    def train(self):
        
        Loss = []
        for i in tqdm(range(self.num_samples)):
            self.optimizer.zero_grad()
            
            if self.method_used == "Explicit":
                u = self.u_1
                u.requires_grad = True
            else:
                u = self.model(self.x_torch).view(1, self.height, self.width, 3).permute(0, 3, 1, 2)
            
            convresult = F.conv2d(u, self.Gaussian_kernel, padding=10, groups=3)
            data_term = torch.sum((convresult - self.y_blurry_noisy) ** 2)
            
            regularization_term_x = torch.sum(torch.abs(F.conv2d(u, self.diff_kernel_x, padding=0, groups=3)))
            regularization_term_y = torch.sum(torch.abs(F.conv2d(u, self.diff_kernel_y, padding=0, groups=3)))
            regularization_term = regularization_term_x + regularization_term_y

            loss = data_term + self.alpha * regularization_term
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            # if i % 100 == 0:
            #     print(f"Epoch {i}, Loss: {loss.item()}")
            
            Loss.append(loss.item())
        # self.model.eval()    
        print(f'Minimum loss for the {self.method_used} method: {min(Loss)}')
                
        return u.detach() 
 
class Pipeline():
    def __init__(self, dataloader, alpha = 0.0001, lr = 1e-3, num_samples=2000):
        self.dataloader = dataloader
        self.alpha = alpha
        self.lr = lr
        self.num_samples = num_samples
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                
    def run(self):
               
        for images in self.dataloader:
            
            images = images.to(self.device)
            
            deblur_explicit = Deblurring_training(images, None, method_used="Explicit", 
                                                 num_samples=self.num_samples, alpha=self.alpha, lr=self.lr,)
            u_explicit = deblur_explicit.train()
            psnr_explicit = PSNR(u_explicit, images).item()
            
            
            deblur_fourier = Deblurring_training(images, FourierFeatureNetwork, method_used="Gaussian_Fourier", 
                                                 num_samples=self.num_samples, alpha=self.alpha, lr=self.lr, 
                                                 in_features=2, hidden_features=512, out_features=3, 
                                                 mapping_size=512, scale=10.0)
            u_fourier = deblur_fourier.train()
            psnr_fourier = PSNR(u_fourier, images).item()
                   
            deblur_wire_c = Deblurring_training(images, WIRE_cmplx, method_used="WIRE_complex", 
                                              num_samples=self.num_samples, alpha=self.alpha, lr=self.lr,
                                              in_features=2, hidden_features=300, hidden_layers=2, out_features=3, 
                                              first_omega_0=10.0, hidden_omega_0=10.0, scale=10.0,)
            u_wire_cmplx = deblur_wire_c.train()
            psnr_wire_cmplx = PSNR(u_wire_cmplx, images).item()          
            
            
        
            self.Gaussian_kernel = self.create_gaussian_kernel().to(self.device)
            y_blurry_noisy = self.create_blurry_noisy_image(images)
            y_blurry_noisy_display = y_blurry_noisy.squeeze(0).permute(1, 2, 0)
            y_blurry_noisy_display = y_blurry_noisy_display * 0.5 + 0.5
            
            psnr_blur = PSNR(y_blurry_noisy, images).item()            
            # # Visualize results
            self.visualize_results(images, y_blurry_noisy, u_explicit, u_fourier, u_wire_cmplx, 
                                   psnr_blur, psnr_explicit, psnr_fourier, psnr_wire_cmplx)
        
        
    def visualize_results(self, images, blurred, u_explicit, u_fourier, u_wire_cmplx, 
                          psnr_blur, psnr_explicit, psnr_fourier, psnr_wire_cmplx):
        
       
        fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
        fig2, axs2 = plt.subplots(1, 3, figsize=(15, 5))

        # Original and blurred images
        images = images.squeeze().permute(1, 2, 0).cpu().numpy()
        images = images * 0.5 + 0.5
        axs1[0].imshow(images)
        axs1[0].set_title("Original image")
        axs1[0].axis('off')

        blurred = blurred.squeeze().permute(1, 2, 0).cpu().numpy()
        blurred = blurred * 0.5 + 0.5
        axs1[1].imshow(blurred)
        axs1[1].set_title(f"Blurry Noisy image\nPSNR: {psnr_blur:.2f}")
        axs1[1].axis('off')
        
        u_explicit = u_explicit.squeeze().permute(1, 2, 0).cpu().numpy()      
        u_explicit = u_explicit * 0.5 + 0.5  
        axs2[0].imshow(u_explicit)
        axs2[0].set_title(f"Deblurred by Explicit\nPSNR: {psnr_explicit:.2f}")
        axs2[0].axis('off')

        # Deblurred images
        u_fourier = u_fourier.squeeze().permute(1, 2, 0).cpu().numpy()
        u_fourier = u_fourier * 0.5 + 0.5
        axs2[1].imshow(u_fourier)
        axs2[1].set_title(f"Deblurred by Gaussian Fourier\nPSNR: {psnr_fourier:.2f}")
        axs2[1].axis('off')

        u_wire_cmplx = u_wire_cmplx.squeeze().permute(1, 2, 0).cpu().numpy()
        u_wire_cmplx = u_wire_cmplx * 0.5 + 0.5
        axs2[2].imshow(u_wire_cmplx)
        axs2[2].set_title(f"Deblurred by WIRE\nPSNR: {psnr_wire_cmplx:.2f}")
        axs2[2].axis('off')

        # Adjust layout and save figures
        fig1.tight_layout()
        fig2.tight_layout()
        plt.show()

        # # Create a directory to save the images if it doesn't exist
        # save_dir = 'deblurred_results'
        # os.makedirs(save_dir, exist_ok=True)

        # # Save the figures
        # fig1.savefig(os.path.join(save_dir, 'original_and_blurred.png'))
        # fig2.savefig(os.path.join(save_dir, 'deblurred_results.png'))

        # # Close the figures to free up memory
        # plt.close(fig1)
        # plt.close(fig2)

        # print(f"Images saved in the '{save_dir}' directory.")


              
                
    # Gaussian Blur 
    def create_gaussian_kernel(self, kernel_size=21, sigma=2, channels=3):
        a = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = torch.meshgrid(a, a, indexing='ij')
        kernel = torch.exp(-0.5 * (xx**2 + yy**2) / sigma**2)
        kernel = kernel / torch.sum(kernel)
        
        gaussian_kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        
        return gaussian_kernel
    
    def create_blurry_noisy_image(self, input):
        y_blurry = F.conv2d(input, self.Gaussian_kernel, groups=3, padding=10)
        y_blurry_noisy = y_blurry + torch.randn_like(y_blurry) * 0.05
        
        return y_blurry_noisy.to(self.device)
 
 
pipeline = Pipeline(dataloader, alpha = 0.004, lr = 5e-3, num_samples=5000)
pipeline.run()        