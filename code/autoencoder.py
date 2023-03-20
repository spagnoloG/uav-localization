""" Autoencoder for satellite images """

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
import os
from PIL import Image
import resource
import argparse
import pickle

# -------------
# MEMORY SAFETY
# -------------
memory_limit_gb = 24
soft, hard = resource.getrlimit(resource.RLIMIT_AS)
resource.setrlimit(resource.RLIMIT_AS, (memory_limit_gb * 1024**3, hard))

# --------
# CONSTANTS
# --------
IMG_H = 160  # On better gpu use 256 and adam optimizer
IMG_W = IMG_H * 2
DATASET_PATHS = [
    "../../diplomska/datasets/oj/montreal_trial1/ge_images/images/",
    #"../../diplomska/datasets/oj/montreal_trial1/teach/images/",
    #"../../diplomska/datasets/oj/suffield_trial1/teach/images/",
    #"../../diplomska/datasets/oj/utiascircle_trial1/teach/images/",
    #"../../diplomska/datasets/oj/utiasday_trial1/teach/images/",
]

#  configuring device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def print_memory_usage_gpu():
    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated(0) / 1024**3, 1),
        "GB",
    )
    print("GPU memory cached:", round(torch.cuda.memory_cached(0) / 1024**3, 1), "GB")


class GEImagePreprocess:
    def __init__(
        self,
        path=DATASET_PATHS[0],
        patch_w=IMG_W,
        patch_h=IMG_H,
    ):
        super().__init__()
        self.path = path
        self.training_set = []
        self.validation_set = []
        self.test_set = []
        self.patch_w = patch_w
        self.patch_h = patch_h

    def load_images(self):
        images = os.listdir(self.path)
        for image in tqdm(range(len(images)), desc="Loading images"):
            if not (images[image].endswith(".jpg") or images[image].endswith(".png")):
                continue
            img = Image.open(self.path + images[image])
            if image % 10 == 0:
                self.validation_set.append(self.preprocess_image(img))
            if image % 10 == 1:
                self.test_set.append(self.preprocess_image(img))
            else:
                self.training_set.append(self.preprocess_image(img))

        return self.training_set, self.validation_set, self.test_set

    def preprocess_image(self, image):
        # ---------
        # DEPRECATED
        # ---------
        # width, height = image.size
        # num_patches_w = width // self.patch_w
        # num_patches_h = height // self.patch_h

        # for i in range(num_patches_w):
        #    for j in range(num_patches_h):
        #        patch = image.crop(
        #            (
        #                i * self.patch_w,
        #                j * self.patch_h,
        #                (i + 1) * self.patch_w,
        #                (j + 1) * self.patch_h,
        #            )
        #        )
        #        patch = patch.convert("L")
        #        patch = np.array(patch).astype(np.float32)
        #        patch = patch / 255
        #        if (i + j) % 30 == 0:
        #            self.validation_set.append(patch)
        #        if (i + j) % 30 == 1:
        #            self.test_set.append(patch)
        #        else:
        #            self.training_set.append(patch)
        image = image.resize((IMG_W, IMG_H))
        image = image.convert("L")
        image = np.array(image).astype(np.float32)
        image = image / 255
        return image


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

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(IMG_H * IMG_W, latent_dim),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=2,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.4),
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
            nn.Dropout(0.3),
            act_fn,
            nn.Conv2d(
                in_channels=out_channels * 4,
                out_channels=out_channels * 8,
                kernel_size=2,
                stride=stride,
            ),
            nn.BatchNorm2d(out_channels * 8),
            nn.Dropout(0.1),
            act_fn,
            nn.Conv2d(
                in_channels=out_channels * 8,
                out_channels=out_channels * 8,
                kernel_size=2,
                stride=stride,
            ),
            act_fn,
            nn.BatchNorm2d(out_channels * 8),
        )

    def forward(self, x):
        x = x.view(-1, 1, IMG_H, IMG_W)
        # Print also the function name
        # for layer in self.net:
        #    x = layer(x)
        #    if self.debug:
        #        print(layer.__class__.__name__, "output shape:\t", x.shape)
        encoded_latent_image = self.conv(x)
        encoded_latent_image = self.linear(encoded_latent_image)
        return encoded_latent_image


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

        self.v, self.u = self.factor()

        self.linear = nn.Sequential(
            nn.Linear(latent_dim, IMG_H * IMG_W),
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
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
        )

    def factor(self):
        dim = IMG_H * IMG_W
        f = dim / (self.out_channels * 8)
        v = np.sqrt(f // 2).astype(int)
        u = (f // v).astype(int)
        return v, u

    def forward(self, x):
        output = self.linear(x)
        output = output.view(len(output), self.out_channels * 8, self.v, self.u)
        # for layer in self.conv:
        #    output = layer(output)
        #    if self.debug:
        #        print(layer.__class__.__name__, "output shape:\t", output.shape)
        output = self.conv(output)
        return output


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
            self.network.parameters(),
            lr=0.01,
            alpha=0.99,
            eps=1e-08,
            weight_decay=0,
            momentum=0,
            centered=False,
        )

    def train(
        self, loss_function, epochs, batch_size, training_set, validation_set, test_set
    ):
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
                loss = loss_function(output, images.view(-1, 1, IMG_H, IMG_W))
                #  zeroing gradients
                self.optimizer.zero_grad()
                #  calculating gradients
                loss.backward()
                #  optimizing weights
                self.optimizer.step()

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
                    val_loss = loss_function(output.flatten(), val_images.flatten())

            # --------------
            # VISUALISATION
            # --------------
            print(
                f"training_loss: {round(loss.item(), 4)} \
                validation_loss: {round(val_loss.item(), 4)}"
            )
            plt_ix = 0
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
                        [test_images.view(-1, 1, IMG_H, IMG_W), reconstructed_imgs],
                        dim=1,
                    ).flatten(0, 1)
                    grid = make_grid(imgs, nrow=10, normalize=True, padding=1)
                    grid = grid.permute(1, 2, 0)
                    plt.figure(dpi=170)
                    plt.title(
                        f"Original/Reconstructed, training loss: {round(loss.item(), 4)} validation loss: {round(val_loss.item(), 4)}"
                    )
                    plt.imshow(grid)
                    plt.axis("off")

                    # Check if directory exists, if not create it
                    if not os.path.exists("visualizations"):
                        os.makedirs("visualizations")
                    if not os.path.exists(f"visualizations/epoch_{epoch+1}"):
                        os.makedirs(f"visualizations/epoch_{epoch+1}")

                    plt.savefig(f"visualizations/epoch_{epoch+1}/img_{plt_ix}.png")
                    plt.clf()
                    plt.close()
                    plt_ix += 1

    def test(self, loss_function, test_set):
        if os.path.exists("./model/encoder.pt") and os.path.exists(
            "./model/decoder.pt"
        ):
            print("Models found, loading...")
        else:
            raise Exception("Models not found, please train the network first")

        self.network.encoder = torch.load("./model/encoder.pt")
        self.network.decoder = torch.load("./model/decoder.pt")
        self.network.eval()

        test_loader = DataLoader(test_set, 10)

        for test_images in test_loader:
            test_images = test_images.to(device)
            with torch.no_grad():
                reconstructed_imgs = self.network(test_images)
                reconstructed_imgs = reconstructed_imgs.cpu()
                test_images = test_images.cpu()
                imgs = torch.stack(
                    [test_images.view(-1, 1, IMG_H, IMG_W), reconstructed_imgs],
                    dim=1,
                ).flatten(0, 1)
                grid = make_grid(imgs, nrow=10, normalize=True, padding=1)
                grid = grid.permute(1, 2, 0)
                plt.figure(dpi=170)
                plt.imshow(grid)
                plt.axis("off")
                plt.show()
                plt.clf()
                plt.close()

    def encode_images(self, test_set):
        if os.path.exists("./model/encoder.pt"):
            print("Models found, loading...")
        else:
            raise Exception("Models not found, please train the network first")

        self.network.encoder = torch.load("./model/encoder.pt")
        self.network.eval()
        test_loader = DataLoader(test_set, 10)
        encoded_images_into_latent_space = []
        for test_images in test_loader:
            test_images = test_images.to(device)
            with torch.no_grad():
                latent_image = self.network.encoder(test_images)
                latent_image = latent_image.cpu()
                encoded_images_into_latent_space.append(latent_image)

        with open("./model/encoded_images.pkl", "wb") as f:
            pickle.dump(encoded_images_into_latent_space, f)

    def autoencode(self, x):
        return self.network(x)

    def encode(self, x):
        encoder = self.network.encoder
        return encoder(x)

    def decode(self, x):
        decoder = self.network.decoder
        return decoder(x)

    def store_model(self):
        if not os.path.exists("model"):
            os.makedirs("model")

        torch.save(self.network.encoder, "./model/encoder.pt")
        torch.save(self.network.encoder.state_dict(), "./model/encoder_state_dict.pt")
        torch.save(self.network.decoder, "./model/decoder.pt")
        torch.save(self.network.decoder.state_dict(), "./model/decoder_state_dict.pt")


def preprocess_data():
    """Load images and preprocess them into torch tensors"""
    training_images, validation_images, test_images = [], [], []
    for path in DATASET_PATHS:
        tr, val, test = GEImagePreprocess(path=path).load_images()
        training_images.extend(tr)
        validation_images.extend(val)
        test_images.extend(test)

    print(
        f"Training on {len(training_images)} images, validating on {len(validation_images)} images, testing on {len(test_images)} images"
    )
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

    return training_data, validation_data, test_data


def main():
    global device
    parser = argparse.ArgumentParser(
        description="Convolutional Autoencoder for GE images"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--train", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--encode", action="store_true", default=False)
    args = parser.parse_args()

    if not (args.train or args.test or args.encode):
        raise ValueError("Please specify whether to train or test")

    if args.no_cuda:
        device = torch.device("cpu")

    if device == torch.device("cuda"):
        print("Using GPU")
    else:
        print("Using CPU")

    if args.train:
        training_data, validation_data, test_data = preprocess_data()
        model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder()))
        model.train(
            nn.MSELoss(reduction="sum"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            training_set=training_data,
            validation_set=validation_data,
            test_set=test_data,
        )
        model.store_model()

    elif args.test:
        t, v, td = preprocess_data()
        model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder()))
        model.test(nn.MSELoss(reduction="sum"), td)

    elif args.encode:
        t, v, td = preprocess_data()
        model = ConvolutionalAutoencoder(Autoencoder(Encoder(), Decoder()))
        model.encode_images(td)


if __name__ == "__main__":
    main()
