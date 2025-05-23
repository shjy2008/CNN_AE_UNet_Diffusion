import torch
import os

import torch.utils
from load_oxford_flowers102 import load_oxford_flowers102
import tqdm
import torchvision
from task2a import AutoEncoderTrainer
from PIL import Image
import random
import matplotlib.pyplot as plt

############# Please change this ################
LOAD_FROM_FILE = True
#################################################

# TODO: make the model simpler and try to make 1 image denoise good

# Color shift problem means it doesn't generalise well on general noise

# Try: simplify model, leaky relu, train on more data

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")

class AugmentedDataset(torch.utils.data.Dataset):

    def __init__(self, img_and_labels):
        self.img_and_labels = img_and_labels
    
    def __getitem__(self, idx):
        return self.img_and_labels[idx]

    def __len__(self):
        return len(self.img_and_labels)
    
# class UNetDenoiser(torch.nn.Module):
#     def __init__(self):
#         super(UNetDenoiser, self).__init__()

#         # -----Downsampling----- #
#         # Input: 12*12*64
#         self.down1_1 = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = 1, padding = 1)
#         # Output: 12*12*128

#         self.output = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = 1, padding = 1)
#         # Output: 12*12*64
    
    
#     def forward(self, x):
#         # is_batch = x.ndim == 4
#         # if not is_batch:
#         #     x = x.unsqueeze(0) # (batch_size(1), channels, w, h)

#         # -----Downsampling----- #
#         x = self.down1_1(x)
#         x = torch.nn.functional.leaky_relu(x, 0.2)

#         x = self.output(x)

#         # if not is_batch:
#         #     x = x.squeeze(0)

#         return x

# class UNetDenoiser(nn.Module):
#     def __init__(self):
#         super(UNetDenoiser, self).__init__()

#         # -----Downsampling----- #
#         self.down1_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
#         self.dropout1 = torch.nn.Dropout(0.2)

#         self.down1_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
#         self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

#         self.down2_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
#         self.dropout2 = torch.nn.Dropout(0.2)

#         self.down2_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
#         self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

#         # -----Bottleneck----- #
#         self.bottleneck1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
#         self.bottleneck2 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)

#         # -----Upsampling----- #
#         self.up2_1 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)

#         self.up2_2 = torch.nn.Conv2d(in_channels=(256 + 256), out_channels=128, kernel_size=(3, 3), stride=1, padding=1)

#         self.up1_1 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)

#         self.up1_2 = torch.nn.Conv2d(in_channels=(128 + 128), out_channels=64, kernel_size=(3, 3), stride=1, padding=1)

#         self.output = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)

#         self.apply(self.init_weights)

#     def init_weights(self, layer):
#         if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
#             torch.nn.init.xavier_uniform_(layer.weight)
#             if layer.bias is not None:
#                 torch.nn.init.constant_(layer.bias, 0)

#     def forward(self, x):
#         is_batch = x.ndim == 4
#         if not is_batch:
#             x = x.unsqueeze(0)

#         # -----Downsampling----- #
#         x = torch.relu(self.down1_1(x))
#         x = self.dropout1(x)

#         x = torch.relu(self.down1_2(x))
#         x1_2 = x
#         x = self.pool1(x)

#         x = torch.relu(self.down2_1(x))
#         x = self.dropout2(x)

#         x = torch.relu(self.down2_2(x))
#         x2_2 = x
#         x = self.pool2(x)

#         # -----Bottleneck----- #
#         x = torch.relu(self.bottleneck1(x))
#         x = torch.relu(self.bottleneck2(x))

#         # -----Upsampling----- #
#         x = torch.relu(self.up2_1(x))
#         x = torch.relu(self.up2_2(torch.cat((x2_2, x), dim=1)))

#         x = torch.relu(self.up1_1(x))
#         x = torch.relu(self.up1_2(torch.cat((x1_2, x), dim=1)))

#         x = self.output(x)

#         if not is_batch:
#             x = x.squeeze(0)

#         return x



class UNetDenoiser(torch.nn.Module):
    def __init__(self):
        super(UNetDenoiser, self).__init__()

        # -----Downsampling----- #
        self.down1_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.down1_1_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        # self.dropout1 = torch.nn.Dropout(0.2)

        self.down1_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.down2_1_1 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        # self.dropout2 = torch.nn.Dropout(0.2)

        self.down2_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        # -----Bottleneck----- #
        self.bottleneck1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.bottleneck2 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1)
        self.bottleneck3 = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.bottleneck4 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)

        # -----Upsampling----- #
        self.up2_1 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.up2_1_1 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)

        self.up2_2 = torch.nn.Conv2d(in_channels=(256 + 256), out_channels=128, kernel_size=(3, 3), stride=1, padding=1)

        self.up1_1 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        self.up1_1_1 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)

        self.up1_2 = torch.nn.Conv2d(in_channels=(128 + 128), out_channels=64, kernel_size=(3, 3), stride=1, padding=1)

        self.output = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)

        self.apply(self.init_weights)

    def init_weights(self, layer):
        if isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.ConvTranspose2d):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        is_batch = x.ndim == 4
        if not is_batch:
            x = x.unsqueeze(0)

        # -----Downsampling----- #
        x = torch.relu(self.down1_1(x))
        x = torch.relu(self.down1_1_1(x))
        # x = self.dropout1(x)

        x = torch.relu(self.down1_2(x))
        x1_2 = x
        x = self.pool1(x)

        x = torch.relu(self.down2_1(x))
        x = torch.relu(self.down2_1_1(x))
        # x = self.dropout2(x)

        x = torch.relu(self.down2_2(x))
        x2_2 = x
        x = self.pool2(x)

        # -----Bottleneck----- #
        x = torch.relu(self.bottleneck1(x))
        x = torch.relu(self.bottleneck2(x))
        x = torch.relu(self.bottleneck3(x))
        x = torch.relu(self.bottleneck4(x))


        # -----Upsampling----- #
        x = torch.relu(self.up2_1(x))
        x = torch.relu(self.up2_1_1(x))
        x = torch.relu(self.up2_2(torch.cat((x2_2, x), dim=1)))

        x = torch.relu(self.up1_1(x))
        x = torch.relu(self.up1_1_1(x))
        x = torch.relu(self.up1_2(torch.cat((x1_2, x), dim=1)))

        x = self.output(x)

        if not is_batch:
            x = x.squeeze(0)

        return x




# For loading dataset, training model, testing model...
class UNetDenoiserTrainer(object):

    def __init__(self, imsize = 96, denoise_steps = 10):
        self.imsize = imsize
        self.denoise_steps = denoise_steps

        # Model: UNetDenoiser
        self.denoiser_model = None

        # Model: Autoencoder
        ae_trainer = AutoEncoderTrainer(imsize = imsize)
        ae_trainer.load_local_model()
        self.ae_encoder = ae_trainer.ae_encoder
        self.ae_decoder = ae_trainer.ae_decoder

        # Dataset
        self.training_set = None
        self.validation_set = None
        self.test_set = None
        self.class_names = None

        self.training_data = None
        self.validation_data = None
        self.test_data = None

        # Save file configs
        path_to_this_scripts_folder = os.path.dirname(os.path.realpath("./DL/"))
        path_to_save_folder = os.path.join(path_to_this_scripts_folder, "saved")
        if not os.path.isdir(path_to_save_folder):
            os.mkdir(path_to_save_folder)
        # save_base_name = os.path.join(path_to_save_folder, "oxford_flowers")
        # self.saved_weights = save_base_name + "_torch_denoiser.weights.h5"
        self.saved_weights = os.path.join(path_to_save_folder, "task2_simple_test") + "/task2b_simple_test.weights.h5"
        self.saved_images_path = os.path.join(path_to_save_folder, "task2_simple_test")
        if not os.path.isdir(self.saved_images_path):
            os.mkdir(self.saved_images_path)

        # Device, gpu, mps, or cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"device: {self.device}")

    def load_dataset(self, fine_grained = False, batch_size = 16, data_augmentation = True):
        self.training_set, self.validation_set, self.test_set, self.class_names = load_oxford_flowers102(imsize = self.imsize, fine = fine_grained)

        if data_augmentation:
            print("Data augmentation. Start preparing data... (Wait about 1 minute...)")
            transform_func = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(), # Horizontal flip
                # torchvision.transforms.RandomRotation(degrees = 25),
                # torchvision.transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
                
                # torchvision.transforms.RandomAffine(degrees = 15, translate = (0.1, 0.1)), # Affine
                torchvision.transforms.CenterCrop(size = (imsize, imsize)),
                torchvision.transforms.ToTensor()
            ])

            label_to_count = {}
            label_to_imgs = {}
            for x, y in self.training_set:
                label = y
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1

                if label in label_to_imgs:
                    label_to_imgs[label].append(x)
                else:
                    label_to_imgs[label] = [x]
            
            # Training data:
            # coarse: {8: 291, 2: 164, 3: 798, 7: 103, 4: 368, 1: 49, 9: 97, 5: 201, 6: 136, 0: 115}
            # fine-grained: {76: 231, 45: 176, 22: 71, 85: 38, 74: 100, 37: 36, 49: 72, 9: 25, 4: 45, 91: 46,
            #  28: 58, 79: 85, 51: 65, 88: 164, 67: 34, 84: 43, 36: 88, 55: 89, 94: 108, 42: 110, 80: 146, 57: 94,
            #  77: 117, 89: 62, 87: 134, 46: 47, 73: 151, 93: 142, 13: 28, 97: 62, 50: 238, 71: 76, 92: 26, 40: 107,
            #  52: 73, 64: 82, 53: 41, 27: 46, 72: 174, 18: 29, 100: 38, 7: 65, 35: 55, 2: 20, 43: 73, 31: 25, 59: 89,
            #  60: 30, 63: 32, 54: 51, 29: 65, 83: 66, 11: 67, 10: 67, 96: 46, 17: 62, 61: 35, 82: 111, 81: 92, 14: 29,
            #  75: 87, 21: 39, 86: 43, 26: 20, 47: 51, 66: 22, 90: 56, 58: 47, 16: 65, 15: 21, 62: 34, 98: 43, 19: 36,
            #  32: 26, 78: 21, 68: 34, 69: 42, 1: 40, 39: 47, 5: 25, 8: 26, 41: 39, 20: 20, 65: 41, 99: 29, 70: 58, 95: 71,
            #  48: 29, 30: 32, 56: 47, 25: 21, 23: 22, 12: 29, 24: 21, 33: 20, 3: 36, 44: 20, 34: 23, 0: 20, 38: 21, 101: 28, 6: 20}

            img_and_labels = []
            max_label_count = max(label_to_count.values()) # The maximum number of images of one label (coarse: 798, fine: 231)
            for label, imgs in label_to_imgs.items():
                augment_img_count = max_label_count - len(imgs)
                index = 0
                for i in range(augment_img_count):
                    pil_img = torchvision.transforms.ToPILImage()(imgs[index])
                    new_img = transform_func(pil_img)
                    img_and_labels.append((new_img, label))

                    index += 1
                    if index >= len(imgs):
                        index = 0
            
            augmented_dataset = AugmentedDataset(img_and_labels = img_and_labels)
            self.training_set = torch.utils.data.ConcatDataset([self.training_set, augmented_dataset])

        
        # Use DataLoader to load data into batches
        self.training_data = torch.utils.data.DataLoader(self.training_set, batch_size = batch_size, shuffle = True)
        self.validation_data = torch.utils.data.DataLoader(self.validation_set, batch_size = batch_size, shuffle = False)
        self.test_data = torch.utils.data.DataLoader(self.test_set, batch_size = batch_size, shuffle = False)
    
    # Add noise to image
    def add_gaussian_noise_to_image(self, image, std = 0.1, mean = 0):
        std = std
        mean = mean # 0: Pure noise, no brightness change
        noise = torch.randn(image.size(), device = self.device) * std + mean
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0, 1)
        return noisy_image

    def train(self, load_from_file = False, epochs = 50, learning_rate = 0.001):
        self.denoiser_model = UNetDenoiser()
        self.denoiser_model.to(self.device)

        print_number_of_trainable_model_parameters(self.denoiser_model)

        if load_from_file and os.path.isfile(self.saved_weights):
            # Load previous model
            print(f"Loading weights from {self.saved_weights}")
            self.denoiser_model.load_state_dict(torch.load(self.saved_weights, weights_only = True, map_location = self.device))
        else:
        # if True:
            # Optimizer
            optimizer = torch.optim.Adam(self.denoiser_model.parameters(), lr = learning_rate, weight_decay=0)

            loss = torch.nn.MSELoss()


            
            sample_count = 100
            indices = random.sample(range(len(self.training_set)), sample_count)
            image_samples = [self.training_set[i][0].to(self.device) for i in indices]
            # image_samples = [ # TODO: set to more/all images and check, because 6 can work if noise is 0.05*(i+1)
            #                   self.training_set[0][0].to(self.device),
            #                   self.training_set[100][0].to(self.device),
            #                   self.training_set[200][0].to(self.device),
            #                   self.training_set[300][0].to(self.device),
            #                   self.training_set[400][0].to(self.device),
            #                   self.training_set[600][0].to(self.device)
            #                   ]
            torchvision.utils.save_image(image_samples, os.path.join(self.saved_images_path, f"image_samples.jpg"), nrow = 10)

            # Check data pixel color imbalance
            pixel_sum = torch.zeros(3, device=self.device)
            for image in image_samples:
                pixel_sum += image.sum(dim=[1, 2])  # image is [3, H, W], so we sum over H and W
            color_distribution = pixel_sum / pixel_sum.sum()
            print("Pixel color distribution (R, G, B):", color_distribution)
        
            for epoch in range(1, epochs + 1):
                # Switch to training mode, activate BatchNorm and Dropout
                self.denoiser_model.train()

                total_loss_training = 0

                print(f"Epoch {epoch}/{epochs}")

                '''
                # # Test training the first image only
                inputs = []
                labels = []
                level_to_inputs = [([], []) for _ in range(self.denoise_steps)] # [[inputs_lv1, outputs_lv1], [inputs_lv2, outputs_lv2], ...]
                for image_sample in image_samples:
                    # Train only 1 image, 10 time steps (from very noisy to less noisy)
                    noisy_latent_images = []
                    noisy_images = self.create_noisy_images_tensor(image_sample)
                    for noisy_image in noisy_images:
                    # for i in range(self.denoise_steps + 1):
                    #     saved_path = os.path.join(self.saved_images_path, f"train_image_{i}.jpg")
                    #     noisy_image = Image.open(saved_path)
                    #     noisy_image = torchvision.transforms.ToTensor()(noisy_image).to(self.device)

                        noisy_latent_image = self.ae_encoder(noisy_image)
                        noisy_latent_images.append(noisy_latent_image)

                    for j in range(len(noisy_latent_images) - 1):
                        input = noisy_latent_images[j + 1].detach()
                        label = noisy_latent_images[j].detach()

                        level_to_inputs[j][0].append(input)
                        level_to_inputs[j][1].append(label)

                        inputs.append(input)
                        labels.append(label)
                


                # Train only 1 image, fixed 1 input and 1 output
                # batch_loss = 0
                # input_img_path = os.path.join(self.saved_images_path, f"train_image_1.jpg")
                # input = torchvision.transforms.ToTensor()(Image.open(input_img_path).convert('RGB'))
                # input = input.to(self.device)
                # input = self.ae_encoder(input)
                # inputs.append(input)

                # label_img_path = os.path.join(self.saved_images_path, f"train_image_0.jpg")
                # label = torchvision.transforms.ToTensor()(Image.open(label_img_path).convert('RGB'))
                # label = label.to(self.device)
                # label = self.ae_encoder(label)
                # labels.append(label)

                for (inputs, labels) in level_to_inputs:
                    batch_loss = 0
                    
                    inputs = torch.stack(inputs)
                    labels = torch.stack(labels)
    
                    outputs = self.denoiser_model(inputs)
                    loss_value = loss(outputs, labels)
    
                    # output = self.denoiser_model(input)
                    # loss_value = loss(output, label)
    
                    batch_loss += loss_value
    
                    optimizer.zero_grad()
                    batch_loss.backward() # Calculate gradient
                    optimizer.step() # # Update weights and biases
    
                    total_loss_training += batch_loss.item()
                '''

                
                batches_loop = tqdm.tqdm(enumerate(self.training_data), total = len(self.training_data), bar_format = '{n_fmt}/{total_fmt} [{bar}] - ETA: {remaining}{postfix}', ascii = ' >=')

                for i, (x_batch, y_batch) in batches_loop:
                    x_batch = x_batch.to(self.device)

                    inputs = []
                    labels = []
                    batch_loss = 0

                    for image in x_batch:
                        # For each image, generate 10 noisy images, from a little bit noisy to almost pure random noise
                        noisy_latent_images = []
                        noisy_images = self.create_noisy_images_tensor(image)
                        for noisy_image in noisy_images:
                            noisy_latent_image = self.ae_encoder(noisy_image) # Encode to latent image (96*96*3 -> 6*6*128)
                            noisy_latent_images.append(noisy_latent_image)
                        
                        # Train the model to denoise each noisy latent image to the less noise one
                        for j in range(len(noisy_latent_images) - 1):
                            input = noisy_latent_images[j + 1]
                            label = noisy_latent_images[j]
                            inputs.append(input)
                            labels.append(label)

                        # random_step = random.randint(0, self.denoise_steps - 1)
                        # input = self.ae_encoder(noisy_images[random_step + 1])
                        # label = self.ae_encoder(noisy_images[random_step])
                        # inputs.append(input)
                        # labels.append(label)
                        

                    inputs = torch.stack(inputs)
                    labels = torch.stack(labels)

                    outputs = self.denoiser_model(inputs)

                    loss_value = loss(outputs, labels)
                    batch_loss += loss_value

                    optimizer.zero_grad()
                    batch_loss.backward() # Calculate gradient
                    optimizer.step() # # Update weights and biases

                    total_loss_training += batch_loss.item()

                    batches_loop.set_postfix_str(f"loss: {total_loss_training/(i + 1):.4f}")
                


                    
                num_batches_training = len(self.training_data) # The number of mini-batches in training data

                average_loss_in_epoch_training = total_loss_training / num_batches_training
                
                print(f"loss: {average_loss_in_epoch_training:.4f}")
            
            # Save model to file
            print(f"Saving model to {self.saved_weights}...")
            torch.save(self.denoiser_model.state_dict(), self.saved_weights)

    # Generate images from random noise and test the denoised images
    def generate_images(self):
        self.denoiser_model.eval()

        # Generate random noise
        all_images = []
        for i in range(10):
            # input_img_path = os.path.join(self.saved_images_path, f"task2b_test_input.jpg")
            # input = torchvision.transforms.ToTensor()(Image.open(input_img_path).convert('RGB'))
            # random_noise = input.to(self.device)
            # random_noise = torch.randn(3, self.imsize, self.imsize, device = self.device) * 0.3 # * 2.8
            # random_noise += 0.5
            # random_noise = torch.clamp(random_noise, 0, 1)
            # random_noise = (random_noise - random_noise.min()) / (random_noise.max() - random_noise.min())
            
            # random_noise = self.training_set[0][0].to(self.device)
            # # # random_noise = self.add_gaussian_noise_to_image(random_noise, std = 0.5)
            # random_noise = self.create_noisy_images_tensor(random_noise)[-1]
            
            base_image = torch.ones(3, self.imsize, self.imsize, device=self.device) * 0.5
            random_noise = self.create_noisy_images_tensor(base_image)[-1]
            
            # saved_path = os.path.join(self.saved_images_path, f"train_image_10.jpg")
            # noisy_image = Image.open(saved_path)
            # random_noise = torchvision.transforms.ToTensor()(noisy_image).to(self.device)

            # Encode the random noise
            random_latent_noise = self.ae_encoder(random_noise)

            # Denoise in the latent space for multiple times
            denoised_images = [random_noise]
            current_denoised_latent_image = random_latent_noise
            for step_index in range(self.denoise_steps):
                current_denoised_latent_image = self.denoiser_model(current_denoised_latent_image)
                current_denoised_image = self.ae_decoder(current_denoised_latent_image) # Decode from latent space back to an image
                denoised_images.append(current_denoised_image)
            
            all_images.extend(denoised_images)
            
        # Save and check the resulting images
        saved_path = os.path.join(self.saved_images_path, f"generate_images.jpg")
        image_count_in_a_row = self.denoise_steps + 1 #int(len(all_images) / (self.denoise_steps + 1))
        #os.remove(saved_path)
        torchvision.utils.save_image(all_images, saved_path, nrow = image_count_in_a_row)

        # Show the image
        img = Image.open(saved_path)
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.title("Images in each row are generated from a different random noise")
        plt.axis('off')  # Hide axes for better visualization
        plt.show()
        
        

    def generate_images_test(self):
        random_noise = self.training_set[0][0].to(self.device)
        # random_noise = self.add_gaussian_noise_to_image(random_noise, std = 0.1)
        random_noise = self.create_noisy_images_tensor(random_noise)[-3]
        
        # random_noise = torch.randn(3, self.imsize, self.imsize, device = self.device)# * 2.8
        # random_noise = torch.clamp(random_noise, 0, 1)
        # saved_path = os.path.join(self.saved_images_path, f"train_image_random_noise.jpg")
        # torchvision.utils.save_image(random_noise, saved_path, nrow = 1)
        # input = torchvision.transforms.ToTensor()(Image.open(saved_path).convert('RGB'))
        
        input = random_noise
        print(self.check_image_range(input))
        # input = torch.clamp(input, 0, 1)  # Ensure values in [0,1]

        # comment these 3 lines, the generated image become purple
        # saved_path = os.path.join(self.saved_images_path, f"train_image_10000.jpg")
        # torchvision.utils.save_image(input, saved_path, nrow = 1)
        # input = torchvision.transforms.ToTensor()(Image.open(saved_path).convert('RGB'))
        
        # input_img_path = os.path.join(self.saved_images_path, f"train_image_10.jpg")
        # input = torchvision.transforms.ToTensor()(Image.open(input_img_path).convert('RGB'))
        input = input.to(self.device)
        input = self.ae_encoder(input)
        
        output = self.denoiser_model(input)

        output_image = self.ae_decoder(output)

        saved_path = os.path.join(self.saved_images_path, f"generate_images_test.jpg")
        torchvision.utils.save_image(output_image, saved_path, nrow = 1)

        # Show the image
        img = Image.open(saved_path)
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.title("Images in each row are generated from a different random noise")
        plt.axis('off')  # Hide axes for better visualization
        plt.show()
        
    def check_image_range(self, tensor):
        below_zero = (tensor < 0).any().item()
        above_one = (tensor > 1).any().item()
        
        if below_zero or above_one:
            print(f"⚠️ Out-of-range values detected!")
            if below_zero:
                print(f" - Has values < 0")
            if above_one:
                print(f" - Has values > 1")
        else:
            print("✅ All values are within [0, 1]")
            
    # Create (self.denoise_steps + 1) images, the first element is the original image, the last one is almost 'pure' random noise
    def create_noisy_images_tensor(self, original_image):
        noisy_images = [original_image]
        noisy_image = original_image
        for i in range(self.denoise_steps):
            std = 0.1 * (i + 1) #0.5 * (i + 1)
            # if i == self.denoise_steps - 1: # The last one, add super large noise
            #     std = 100
            noisy_image = self.add_gaussian_noise_to_image(noisy_image, std = std)
            noisy_images.append(noisy_image)
        noisy_images = torch.stack(noisy_images)
        return noisy_images

    def save_test(self):
        image_sample = self.training_set[0][0].to(self.device)
        image_samples = self.create_noisy_images_tensor(image_sample)
        
        saved_path = os.path.join(self.saved_images_path, f"task2b_train_images.jpg")
        torchvision.utils.save_image(image_samples, saved_path, nrow = len(image_samples))

        for i in range(len(image_samples)):
            saved_path = os.path.join(self.saved_images_path, f"train_image_{i}.jpg")
            torchvision.utils.save_image(image_samples[i], saved_path, nrow = 1)
        

if __name__ == "__main__":

    load_from_file = LOAD_FROM_FILE

    imsize = 96
    batch_size = 16
    data_augmentation = False
    epochs = 50
    learning_rate = 0.0001
    denoise_steps = 10 # How many steps to denoise from random noise to image

    def print_hyper_params():
        print("------------------------------------")
        print(f"imsize:{imsize}")
        print(f"batch_size:{batch_size}")
        print(f"epochs:{epochs}")
        print(f"learning_rate:{learning_rate}")
        print(f"denoise_steps:{denoise_steps}")
        print("------------------------------------")

    print_hyper_params()

    trainer = UNetDenoiserTrainer(imsize = imsize, denoise_steps = denoise_steps)
    trainer.load_dataset(fine_grained = False, batch_size = batch_size, 
                         data_augmentation = data_augmentation and load_from_file == False)
    trainer.save_test()
    trainer.train(load_from_file = load_from_file, epochs = epochs, learning_rate = learning_rate)
    # # trainer.test()
    trainer.generate_images_test()
    trainer.generate_images()

    print_hyper_params()
