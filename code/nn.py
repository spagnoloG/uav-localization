#  article dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.utils import make_grid
import random
import json
import os
from PIL import Image
import resource

memory_limit_gb = 24
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (memory_limit_gb * 1024 * 1024 * 1024, hard))

# --------
# CONSTANTS
# --------
IMG_H = 160
IMG_W = 320
DATASET_PATH = "../../datasets/sat_data/woodbridge/images/"

#  configuring device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


def print_memory_usage_gpu():
    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated(0) / 1024**3, 1),
        "GB",
    )
    print(
        "GPU memory cached:   ", round(torch.cuda.memory_cached(0) / 1024**3, 1), "GB"
    )


class GEImagePreprocess:
    def __init__(
        self,
        path=DATASET_PATH,
        patch_w=IMG_W,
        patch_h=IMG_H,
    ):
        super().__init__()
        self.path = path
        self.training_set = []
        self.validation_set = []
        self.patch_w = patch_w
        self.patch_h = patch_h

    def load_images(self):
        images = os.listdir(self.path)
        for image in tqdm(range(len(images)), desc="Loading images"):
            img = Image.open(self.path + images[image])
            img = self.preprocess_image(img)

        return self.training_set, self.validation_set

    def preprocess_image(self, image):
        width, height = image.size
        num_patches_w = width // self.patch_w
        num_patches_h = height // self.patch_h

        for i in range(num_patches_w):
            for j in range(num_patches_h):
                patch = image.crop(
                    (
                        i * self.patch_w,
                        j * self.patch_h,
                        (i + 1) * self.patch_w,
                        (j + 1) * self.patch_h,
                    )
                )
                patch = patch.convert("L")
                patch = np.array(patch).astype(np.float32)
                patch = patch / 255
                if (i + j) % 10 == 0:
                    self.validation_set.append(patch)
                else:
                    self.training_set.append(patch)


training_images, validation_images = GEImagePreprocess().load_images()
tr, val = GEImagePreprocess(path='../../datasets/sat_data/fountainhead/images/').load_images()

training_images.extend(tr)
validation_images.extend(val)

#  defining dataset class
class GEDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]

        if self.transforms != None:
            image = self.transforms(image)
        return image


#  creating pytorch datasets
training_data = GEDataset(
    training_images,
    transforms=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    ),
)

validation_data = GEDataset(
    validation_images,
    transforms=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    ),
)

test_data = GEDataset(
    validation_images,
    transforms=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
        ]
    ),
)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=128,
        latent_dim=1000,
        stride=2,
        act_fn=nn.LeakyReLU(),
        debug=False,
    ):
        super().__init__()
        self.debug = debug

        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            act_fn,
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 2,
                kernel_size=2,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels * 2),
            act_fn,
            nn.Conv2d(
                in_channels=out_channels * 2,
                out_channels=out_channels * 4,
                kernel_size=2,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels * 4),
            act_fn,
            nn.Conv2d(
                in_channels=out_channels * 4,
                out_channels=out_channels * 8,
                kernel_size=2,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels * 8),
            act_fn,
            nn.Conv2d(
                in_channels=out_channels * 8,
                out_channels=out_channels * 8,
                kernel_size=2,
                stride=stride,
            ),
            act_fn,
            nn.BatchNorm2d(out_channels * 8),
            nn.Flatten(),
            nn.Linear(51200, latent_dim),
        )

    def forward(self, x):
        x = x.view(-1, 1, 160, 320)
        # Print also the function name
        for layer in self.net:
            x = layer(x)
            if self.debug:
                print(layer.__class__.__name__, "output shape:\t", x.shape)
        encoded_latent_image = x
        return encoded_latent_image


#  defining decoder
class Decoder(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=128,
        latent_dim=1000,
        stride=2,
        kernel_size=2,
        act_fn=nn.LeakyReLU(),
        debug=False,
    ):
        super().__init__()
        self.debug = debug

        self.out_channels = out_channels

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 51200),
        )

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=out_channels * 8,
                out_channels=out_channels * 8,
                kernel_size=kernel_size,
                stride=stride,
            ),
            act_fn,
            nn.BatchNorm2d(out_channels * 8),
            nn.ConvTranspose2d(
                in_channels=out_channels * 8,
                out_channels=out_channels * 4,
                kernel_size=kernel_size,
                stride=stride,
            ),
            act_fn,
            nn.BatchNorm2d(out_channels * 4),
            nn.ConvTranspose2d(
                in_channels=out_channels * 4,
                out_channels=out_channels * 2,
                kernel_size=kernel_size,
                stride=stride,
            ),
            act_fn,
            nn.BatchNorm2d(out_channels * 2),
            nn.ConvTranspose2d(
                in_channels=out_channels * 2,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            act_fn,
            nn.BatchNorm2d(out_channels),
            nn.ConvTranspose2d(
                in_channels=out_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
            ),
            act_fn,
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        output = self.linear(x)
        output = output.view(len(output), self.out_channels * 8, 5, 10)
        for layer in self.conv:
            output = layer(output)
            if self.debug:
                print(layer.__class__.__name__, "output shape:\t", output.shape)
        return output


#  defining autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.to(device)

        self.decoder = decoder
        self.decoder.to(device)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class ConvolutionalAutoencoder:
    def __init__(self, autoencoder):
        self.network = autoencoder
        self.optimizer = torch.optim.RMSprop(
            self.network.parameters(), lr=0.01, alpha=0.99, eps=1e-08,
            weight_decay=0, momentum=0, centered=False
        )

    def train(
        self, loss_function, epochs, batch_size, training_set, validation_set, test_set
    ):
        #  creating log
        log_dict = {
            "training_loss_per_batch": [],
            "validation_loss_per_batch": [],
            "visualizations": [],
        }

        #  defining weight initialization function
        def init_weights(module):
            if isinstance(module, nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)
            elif isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                module.bias.data.fill_(0.01)

        #  initializing network weights
        self.network.apply(init_weights)

        #  creating dataloaders
        train_loader = DataLoader(training_set, batch_size)
        val_loader = DataLoader(validation_set, batch_size)
        test_loader = DataLoader(test_set, 10)

        #  setting convnet to training mode
        self.network.train()
        self.network.to(device)

        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            train_losses = []

            # ------------
            #  TRAINING
            # ------------
            print("training...")
            for images in tqdm(train_loader):
                #  sending images to device
                images = images.to(device)
                #  reconstructing images
                output = self.network(images)
                #  computing loss
                loss = loss_function(output, images.view(-1, 1, 160, 320))
                #  zeroing gradients
                self.optimizer.zero_grad()
                #  calculating gradients
                loss.backward()
                #  optimizing weights
                self.optimizer.step()

                # --------------
                # LOGGING
                # --------------
                log_dict["training_loss_per_batch"].append(loss.item())

            # --------------
            # VALIDATION
            # --------------
            print("validating...")
            for val_images in tqdm(val_loader):
                with torch.no_grad():
                    #  sending validation images to device
                    val_images = val_images.to(device)
                    #  reconstructing images
                    output = self.network(val_images)
                    #  computing validation loss
                    val_loss = loss_function(output, val_images.view(-1, 1, 160, 320))

                # --------------
                # LOGGING
                # --------------
                log_dict["validation_loss_per_batch"].append(val_loss.item())

            # --------------
            # VISUALISATION
            # --------------
            print(
                f"training_loss: {round(loss.item(), 4)} validation_loss: {round(val_loss.item(), 4)}"
            )
            if epoch % 5 == 0:
                for test_images in test_loader:
                    #  sending test images to device
                    test_images = test_images.to(device)
                    with torch.no_grad():
                        #  reconstructing test images
                        reconstructed_imgs = self.network(test_images)
                    #  sending reconstructed and images to cpu to allow for visualization
                    reconstructed_imgs = reconstructed_imgs.cpu()
                    test_images = test_images.cpu()

                    #  visualisation
                    imgs = torch.stack(
                        [test_images.view(-1, 1, 160, 320), reconstructed_imgs], dim=1
                    ).flatten(0, 1)
                    grid = make_grid(imgs, nrow=10, normalize=True, padding=1)
                    grid = grid.permute(1, 2, 0)
                    plt.figure(dpi=170)
                    plt.title(
                        f"Original/Reconstructed, training loss: {round(loss.item(), 4)} validation loss: {round(val_loss.item(), 4)}"
                    )
                    plt.imshow(grid)
                    log_dict["visualizations"].append(grid)
                    plt.axis("off")
                    plt.savefig(f"epoch_{epoch+1}.png")
                    break

        return log_dict

    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        encoder = self.network.encoder
        return encoder(x)

    def decode(self, x):
        decoder = self.network.decoder
        return decoder(x)


#  training model
model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder()))

log_dict = model.train(
    nn.MSELoss(),
    epochs=30,
    batch_size=14,
    training_set=training_data,
    validation_set=validation_data,
    test_set=test_data,
)
