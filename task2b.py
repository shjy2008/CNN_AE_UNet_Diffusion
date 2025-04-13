import torch
import os
import torch.utils
from load_oxford_flowers102 import load_oxford_flowers102
import tqdm
import torchvision
from task2a import AutoEncoderTrainer
from PIL import Image
import matplotlib.pyplot as plt

############# Please change this ################
LOAD_FROM_FILE = True
#################################################

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")

class UNetDenoiser(torch.nn.Module):
    def __init__(self):
        super(UNetDenoiser, self).__init__()

        # -----Downsampling----- #
        # Input: 12*12*64
        self.down1_1 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.down1_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.down1_3 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        # Output: 12*12*128
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 6*6*128

        self.down2_1 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.down2_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.down2_3 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        # Output: 6*6*256
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # Output: 3*3*256

        # -----Bottleneck----- #
        self.bottleneck1 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.bottleneck2 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1)
        self.bottleneck3 = torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(3, 3), stride=1, padding=1)
        self.bottleneck4 = torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        # Output: 3*3*256

        # -----Upsampling----- #
        self.up2_1 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        # Output: 6*6*256
        self.up2_2 = torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=1, padding=1)
        self.up2_3 = torch.nn.Conv2d(in_channels=(256 + 256), out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        # Output: 6*6*128

        self.up1_1 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=2, padding=1, output_padding=1)
        # Output: 12*12*128
        self.up1_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.up1_3 = torch.nn.Conv2d(in_channels=(128 + 128), out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        # Output: 12*12*64

        self.output = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        # Output: 12*12*64

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
        x = torch.relu(self.down1_2(x))
        x = torch.relu(self.down1_3(x))
        x1 = x
        x = self.pool1(x)

        x = torch.relu(self.down2_1(x))
        x = torch.relu(self.down2_2(x))
        x = torch.relu(self.down2_3(x))
        x2 = x
        x = self.pool2(x)

        # -----Bottleneck----- #
        x = torch.relu(self.bottleneck1(x))
        x = torch.relu(self.bottleneck2(x))
        x = torch.relu(self.bottleneck3(x))
        x = torch.relu(self.bottleneck4(x))


        # -----Upsampling----- #
        x = torch.relu(self.up2_1(x))
        x = torch.relu(self.up2_2(x))
        x = torch.relu(self.up2_3(torch.cat((x2, x), dim=1)))

        x = torch.relu(self.up1_1(x))
        x = torch.relu(self.up1_2(x))
        x = torch.relu(self.up1_3(torch.cat((x1, x), dim=1)))

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
        path_to_this_scripts_folder = os.path.dirname(os.path.realpath(__file__))
        path_to_save_folder = os.path.join(path_to_this_scripts_folder, "saved")
        if not os.path.isdir(path_to_save_folder):
            os.mkdir(path_to_save_folder)
        self.saved_weights = os.path.join(path_to_save_folder, "Denoiser.weights.h5")
        self.saved_images_path = os.path.join(path_to_save_folder, "task2b_images")
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

    def load_dataset(self, fine_grained = False, batch_size = 16):
        self.training_set, self.validation_set, self.test_set, self.class_names = load_oxford_flowers102(imsize = self.imsize, fine = fine_grained)
        
        # Use DataLoader to load data into batches
        self.training_data = torch.utils.data.DataLoader(self.training_set, batch_size = batch_size, shuffle = True)
        self.validation_data = torch.utils.data.DataLoader(self.validation_set, batch_size = batch_size, shuffle = False)
        self.test_data = torch.utils.data.DataLoader(self.test_set, batch_size = batch_size, shuffle = False)
    
    # Add noise to image
    def add_gaussian_noise_to_image(self, image, std = 0.1, mean = 0):
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
            # Optimizer
            optimizer = torch.optim.Adam(self.denoiser_model.parameters(), lr = learning_rate, weight_decay=0)

            loss = torch.nn.MSELoss()

            for epoch in range(1, epochs + 1):
                # Switch to training mode, activate BatchNorm and Dropout
                self.denoiser_model.train()

                total_loss_training = 0

                print(f"Epoch {epoch}/{epochs}")
                
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
        for i in range(self.denoise_steps):
            # Create a base image whose pixels are all 0.5. Then add random noise to it for <self.denoise_steps> times
            base_image = torch.ones(3, self.imsize, self.imsize, device=self.device) * 0.5
            random_noise = self.create_noisy_images_tensor(base_image)[-1]
            
            # Encode the random noise
            random_latent_noise = self.ae_encoder(random_noise)

            # Denoise in the latent space for <self.denoise_steps> times
            denoised_images = [random_noise]
            current_denoised_latent_image = random_latent_noise
            for step_index in range(self.denoise_steps):
                current_denoised_latent_image = self.denoiser_model(current_denoised_latent_image)
                current_denoised_image = self.ae_decoder(current_denoised_latent_image) # Decode from latent space back to an image
                denoised_images.append(current_denoised_image)
            
            all_images.extend(denoised_images)
            
        # Save and check the resulting images
        saved_path = os.path.join(self.saved_images_path, f"generate_images.jpg")
        image_count_in_a_row = self.denoise_steps + 1
        torchvision.utils.save_image(all_images, saved_path, nrow = image_count_in_a_row)

        # Show the image
        img = Image.open(saved_path)
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.title("Images in each row are generated from a different random noise")
        plt.axis('off')  # Hide axes for better visualization
        plt.show()

    # Create (self.denoise_steps + 1) images, the first element is the original image, the last one is almost 'pure' random noise
    def create_noisy_images_tensor(self, original_image):
        noisy_images = [original_image]
        noisy_image = original_image
        for i in range(self.denoise_steps):
            std = 0.05 * (i + 1) # Each time add more noise. The first few images fewer noise, the last few images more noise
            noisy_image = self.add_gaussian_noise_to_image(noisy_image, std = std) # Add noise to the previous image
            noisy_images.append(noisy_image)
        noisy_images = torch.stack(noisy_images)
        return noisy_images

if __name__ == "__main__":

    load_from_file = LOAD_FROM_FILE

    imsize = 96
    batch_size = 16
    epochs = 1000
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
    trainer.load_dataset(fine_grained = False, batch_size = batch_size)
    trainer.train(load_from_file = load_from_file, epochs = epochs, learning_rate = learning_rate)
    trainer.generate_images()

    print_hyper_params()
