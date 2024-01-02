'''
anime_face/
  data/
  ├── 1.png
  ├── 2.png
  ├── 3.png
  ├── ...
'''
######################################################################################
# TODO: Design the diffusion process for the Anime Face dataset
# Implementation B.1-4
######################################################################################
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm
from torchvision.transforms import Compose, ToTensor, Lambda, Grayscale
from torchvision.datasets import ImageFolder

import os
from PIL import Image

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(999)

# Hyperparameters
image_size = 64
# Set to 3 for color images
channels = 3  
# n_steps 
n_steps = 600
#smaller b
batch_size = 64
# epochs
epochs = 200
# smaller lr
lr = 0.0002
# Initial beta
start_beta = 1e-4
# End beta
end_beta = 0.02

workspace_dir = '.'
model_store_path = f"{workspace_dir}/model/anime_{epochs}.pt"
save_path = ''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#print(torch.cuda.device_count())
#print(torch.cuda.get_device_name(0))

# Dataset transform
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset and DataLoader
workspace_dir = '.'
dataset_path = f"{workspace_dir}/anime_face"
anime_face_dataset = datasets.ImageFolder(root= dataset_path, transform=transform)
dataloader = DataLoader(anime_face_dataset, batch_size=batch_size, shuffle=True,num_workers=2)

class DDPM_Anime(nn.Module):
    def __init__(self, image_shape=(3, 64, 64), n_steps=200, start_beta=1e-4, end_beta=0.02, device=None):
        super(DDPM_Anime, self).__init__()
        self.device = device
        self.image_shape = image_shape
        self.n_steps = n_steps
        # for 64 x 64 x3
        self.noise_predictor = UNet_Anime(in_channels=3, out_channels=3, base_channels=64, n_steps=n_steps).to(device)
        self.betas = torch.linspace(start_beta, end_beta, n_steps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        n, channel, height, width = x0.shape
        alpha_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, channel, height, width).to(self.device)

        noise = alpha_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - alpha_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noise

    def backward(self, x, t):
        return self.noise_predictor(x, t)

def time_embedding(n, d):
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10000 ** (2 * i / d) for i in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:, ::2] = torch.sin(t * wk[:, ::2])
    embedding[:, 1::2] = torch.cos(t * wk[:, 1::2])
    return embedding

class UNet_Anime(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, n_steps=1000, time_embedding_dim=256):
        super(UNet_Anime, self).__init__()

        # Time embedding
        self.time_step_embedding = nn.Embedding(n_steps, time_embedding_dim)
        self.time_step_embedding.weight.data = time_embedding(n_steps, time_embedding_dim)
        self.time_step_embedding.requires_grad_(False)

        # Initial convolution layer [B, 3, 64, 64]->[B, 64, 64, 64])
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Down-sampling layers
        self.down1 = self._down_block(base_channels, base_channels * 2) # [B, 64, 64, 64] -> [B, 128, 32, 32]
        self.down2 = self._down_block(base_channels * 2, base_channels * 4) # [B, 128, 32, 32] -> [B, 256, 16, 16]
        self.down3 = self._down_block(base_channels * 4, base_channels * 8) # [B, 256, 16, 16] -> [B, 512, 8, 8]

        # Bottleneck layer
        self.bot1 = self._block(base_channels * 8, base_channels * 16) # [B, 512, 8, 8] -> [B, 1024, 8, 8]
        self.bot2 = self._block(base_channels * 16, base_channels * 16) # [B, 1024, 8, 8] -> [B, 1024, 8, 8]
        self.bot3 = self._block(base_channels * 16, base_channels * 8) # [B, 1024, 8, 8] -> [B, 512, 8, 8]

        # Up-sampling layers
        self.up1 = self._up_block(base_channels * 16, base_channels * 4)  # [B, 1024, 8, 8] -> [B, 256, 16, 16]
        self.up2 = self._up_block(base_channels * 8, base_channels * 2)  # [B, 512, 16, 16] -> [B, 128, 32, 32]
        self.up3 = self._up_block(base_channels * 4, base_channels)  # [B, 256, 32, 32] -> [B, 64, 64, 64]

        # Final output layer
        self.final_layer = nn.Conv2d(base_channels, out_channels, kernel_size=1) # [B, 64, 64, 64] -> [B, 3, 64, 64]

        # Time embedding linear layers for each block
        self.time_emb_linear_down1 = nn.Linear(time_embedding_dim, base_channels * 2)
        self.time_emb_linear_down2 = nn.Linear(time_embedding_dim, base_channels * 4)
        self.time_emb_linear_down3 = nn.Linear(time_embedding_dim, base_channels * 8)
        
        self.time_emb_linear_up1 = nn.Linear(time_embedding_dim, base_channels * 4)
        self.time_emb_linear_up2 = nn.Linear(time_embedding_dim, base_channels * 2)
        self.time_emb_linear_up3 = nn.Linear(time_embedding_dim, base_channels)

    def _down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 1x1 Residual Connection
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def _up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 1x1 Residual Connection
        )

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),  # 1x1 Residual Connection
        )

    def forward(self, x, t):
        time_emb = self.time_step_embedding(t)

        # Initial convolution
        x = self.init_conv(x)

        # Down-sampling path with skip connection saving
        x1 = self.down1(x)
        x1_skip = x1 + self.time_emb_linear_down1(time_emb).view(x1.shape[0], -1, 1, 1) 
        x2 = self.down2(x1_skip)
        x2_skip = x2 + self.time_emb_linear_down2(time_emb).view(x2.shape[0], -1, 1, 1)
        x3 = self.down3(x2_skip)
        x3_skip = x3 + self.time_emb_linear_down3(time_emb).view(x3.shape[0], -1, 1, 1)

        # Bottleneck
        x = self.bot1(x3_skip)
        x = self.bot2(x)
        x = self.bot3(x)

        # Up-sampling path with skip connections
        x = self.up1(torch.cat([x3_skip, x], dim=1)) # Concatenation for skip connection
        x = x + self.time_emb_linear_up1(time_emb).view(x.shape[0], -1, 1, 1)
        x = self.up2(torch.cat([x2_skip, x], dim=1)) # Concatenation for skip connection
        x = x + self.time_emb_linear_up2(time_emb).view(x.shape[0], -1, 1, 1)
        x = self.up3(torch.cat([x1_skip, x], dim=1)) # Concatenation for skip connection
        x = x + self.time_emb_linear_up3(time_emb).view(x.shape[0], -1, 1, 1)

        # Output layer
        x = self.final_layer(x)

        return x

ddpm_anime = DDPM_Anime(image_shape=(3, 64, 64), n_steps=n_steps, start_beta=start_beta, end_beta=end_beta, device=device)

#print(ddpm_anime)

def show_64images(images, title=""):
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()
    print("The shape of images: ", images.shape)
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=24)
    rows = int(len(images) ** 0.5)
    cols = round(len(images) / rows)
    index = 0
    for row in range(rows):
        for col in range(cols):
            fig.add_subplot(rows, cols, index + 1)
            if index < len(images):
                frame = plt.gca()
                frame.axes.get_yaxis().set_visible(False)
                frame.axes.get_xaxis().set_visible(False)

                # Check if the image has 1 channel (gray) and convert it to 3 channels (RGB)
                if images[index].shape[0] == 1:
                    temp = np.transpose(images[index], (1, 2, 0))
                    temp = np.repeat(temp, 3, axis=2)  # Duplicate the single channel to all 3 channels
                else:
                    temp = np.transpose(images[index], (1, 2, 0))

                temp = (temp + 1) / 2  # Scale from [-1, 1] to [0, 1]
                temp = np.clip(temp, 0, 1)

                plt.imshow(temp)
                index += 1
    #plt.show()
    save_path = 'result' + str(epochs) + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(save_path + title + '.png')

def generate_new_64images(ddpm, n_samples=16, device=None, frames_per_gif=25, gif_name='result'+ str(epochs) + '/sampling.gif'):
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x = torch.randn(n_samples, 3, 64, 64).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, 3, 64, 64).to(device)
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()
                x = x + sigma_t * z

            if idx in frame_idxs or t == 0:
                # Reshaping and scaling images
                normalized = ((x - x.min()) * (1/(x.max() - x.min())) * 255).byte()
                images_np = normalized.cpu().numpy()

                # Reshaping batch to a square frame
                n_side = int(np.sqrt(n_samples))
                frame = np.zeros((n_side * 64, n_side * 64, 3), dtype=np.uint8)
                for i in range(n_side):
                    for j in range(n_side):
                        idx = i * n_side + j
                        if idx < n_samples:
                            frame[i*64:(i+1)*64, j*64:(j+1)*64, :] = np.transpose(images_np[idx], (1, 2, 0))

                frames.append(Image.fromarray(frame))

    # Saving frames as a gif
    frames[0].save(gif_name, save_all=True, append_images=frames[1:], loop=0, duration=100)

    return x

folder_name = 'training_results'
folder_path = os.path.join('result'+ str(epochs), folder_name)
os.makedirs(folder_path, exist_ok=True)
loss_file = os.path.join(folder_path, 'training_loss.txt')

loss_list_anime = []

def train_ddpm_anime(ddpm_anime, dataloader, n_epochs, lr, device, loss_function, model_store_path):
    optimizer = optim.Adam(ddpm_anime.parameters(), lr=lr)
    best_loss = float("inf")
    n_steps = ddpm_anime.n_steps
    torch.autograd.set_detect_anomaly(True)

    def generate_new_images():
        return generate_new_64images(ddpm_anime, device=device)

    for epoch in tqdm(range(n_epochs), desc="訓練進度", colour="green"):
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(dataloader, leave=False, desc=f"第 {epoch + 1}/{n_epochs} 個訓練周期", colour="blue")):
            # Load data
            x0 = batch[0].to(device)
            n = len(x0)

            # Pick random noise for each of the images in the batch
            eta = torch.randn_like(x0).to(device)
            t = torch.randint(0, n_steps, (n,)).to(device)

            # Compute the noisy image based on x0 and the time step
            noises = ddpm_anime(x0, t, eta)

            # Get model estimation of noise based on the images and the time step
            eta_theta = ddpm_anime.backward(noises, t.reshape(n, -1))

            # Optimize the Mean Squared Error (MSE) between the injected noise and the predicted noise
            loss = loss_function(eta_theta, eta)

            # First, initialize the optimizer's gradient and then update the network's weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Aggregate the loss values from each iteration to compute the loss value for an epoch
            epoch_loss += loss.item() * len(x0) / len(dataloader.dataset)

            # Save Losses for plotting later
            loss_list_anime.append(loss.item())
            
            # Save the loss values to a file
            with open(loss_file, 'a') as f: 
                f.write(f"{epoch+1},{loss.item()}\n")

        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Show images generated at the epoch
        show_64images(generate_new_images(), f"Images generated at epoch {epoch + 1} ")
        plt.close('all')
        # If the current loss is better than the previous one, then store the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm_anime.state_dict(), model_store_path)
            log_string += " <Store the best model.>"

        print(log_string)

optimizer = optim.Adam(ddpm_anime.parameters(), lr=lr)
loss_function = torch.nn.MSELoss()

if __name__ == '__main__':

    train_ddpm_anime(ddpm_anime, dataloader, epochs, lr, device, loss_function, model_store_path)
    ######################################################################################
    # TODO: Plot the loss values of DDPM for the Anime Face dataset
    # Implementation B.1-5
    ######################################################################################
    
    with open(loss_file, 'r') as f:
        loss_values = []
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2 and parts[1].replace('.', '', 1).isdigit():
                loss_values.append(float(parts[1]))
    ###
    save_path = 'result'+str(epochs)+'/'

    plt.figure(figsize=(10, 5))
    #plt.plot(loss_list_anime, label='Training Loss')
    plt.plot(loss_values, label='Training Loss')
    plt.title("Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path +'loss_anime.png')
    plt.show()



    ######################################################################################
    # TODO: Store your generate images in 5*5 grid for the Anime Face dataset
    # Implementation B.1-6
    ######################################################################################

    def show_grid_images(images, title="Generated Images", rows=5, cols=5):
        if type(images) is torch.Tensor:
            images = images.detach().cpu().numpy()
        
        fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
        fig.suptitle(title, fontsize=16)

        for i, ax in enumerate(axes.flatten()):
            if i < len(images):
                img = np.transpose(images[i], (1, 2, 0))
                img = (img + 1) / 2  
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path +'grid.png')
        plt.show()

    generated_images = generate_new_64images(ddpm_anime, n_samples=25, device=device)
    show_grid_images(generated_images)