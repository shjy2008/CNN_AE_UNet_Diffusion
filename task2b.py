import torch
import os
from load_oxford_flowers102 import load_oxford_flowers102
import tqdm
import torchvision
from task2a import AutoEncoderTrainer

class UNetDenoiser(torch.nn.Module):
    def __init__(self, in_channels = 128, out_channels = 128):
        super(UNetDenoiser, self).__init__()

        # -----Downsampling----- #
        # Input: 6*6*128
        self.down1_1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 6*6*256

        self.down1_2 = torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 6*6*256

        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # Output: 3*3*256


        # -----Bottleneck----- #
        self.bottleneck1 = torch.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 3*3*512
        
        self.bottleneck2 = torch.nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 3*3*256


        # -----Upsampling----- #
        self.up1_1 = torch.nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = 2, padding = 1, output_padding = 1)
        # Output: 6*6*256

        # Input: the output of down1_2 concatenate the output of up1_1
        self.up1_2 = torch.nn.Conv2d(in_channels = (256 + 256), out_channels = 128, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 6*6*128

        self.output = torch.nn.Conv2d(in_channels = 128, out_channels = out_channels, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 6*6*128
    
    
    def forward(self, x):
        # -----Downsampling----- #
        # Input: 6*6*128
        x = self.down1_1(x)
        x = torch.relu(x)

        x1_2 = self.down1_2(x) # output shape: (256, 6, 6)
        x = torch.relu(x1_2)

        x = self.pool1(x)

        # -----Bottleneck----- #
        x = self.bottleneck1(x)
        x = torch.relu(x)

        x = self.bottleneck2(x)
        x = torch.relu(x)

        # -----Upsampling----- #
        x = self.up1_1(x) # output shape: (256, 6, 6)
        x = torch.relu(x)

        concat_dim = 0
        if x.ndim == 3: # Only process one latent image, x.shape like (256, 6, 6)
            concat_dim = 0
        elif x.ndim == 4: # Batch processing, x.shape like (batch_size, 256, 6, 6)
            concat_dim = 1
        x = self.up1_2(torch.concat((x1_2, x), dim = concat_dim)) # Skip connection: pass the output of x1_2 to up1_2
        x = torch.relu(x)

        x = self.output(x)
        # Output: 6*6*128

        return x

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
        save_base_name = os.path.join(path_to_save_folder, "oxford_flowers")
        self.saved_weights = save_base_name + "_torch_denoiser.weights.h5"
        self.saved_images_path = os.path.join(path_to_save_folder, "task2_images")

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
        std = std
        mean = mean # 0: Pure noise, no brightness change
        noise = torch.randn(image.size(), device = self.device) * std + mean
        return image + noise

    def train(self, load_from_file = False, epochs = 50, learning_rate = 0.001):
        self.denoiser_model = UNetDenoiser(in_channels = 128, out_channels = 128)
        self.denoiser_model.to(self.device)

        if load_from_file and os.path.isfile(self.saved_weights):
            # Load previous model
            print(f"Loading weights from {self.saved_weights}")
            self.denoiser_model.load_state_dict(torch.load(self.saved_weights, weights_only = True, map_location = self.device))
        else:
            # Optimizer
            optimizer = torch.optim.Adam(self.denoiser_model.parameters(), lr = learning_rate)

            loss = torch.nn.MSELoss()

            for epoch in range(1, epochs + 1):
                # Switch to training mode, activate BatchNorm and Dropout
                self.denoiser_model.train()

                total_loss_training = 0

                print(f"Epoch {epoch}/{epochs}")
                batches_loop = tqdm.tqdm(enumerate(self.training_data), total = len(self.training_data), bar_format = '{n_fmt}/{total_fmt} [{bar}] - ETA: {remaining}{postfix}', ascii = ' >=')

                for i, (x_batch, y_batch) in batches_loop:
                    x_batch = x_batch.to(self.device)

                    batch_loss = 0

                    for image in x_batch:
                        # For each image, generate 10 noisy images, from a little bit noisy to almost pure random noise
                        noisy_latent_images = []
                        for step_index in range(self.denoise_steps):
                            noisy_image = self.add_gaussian_noise_to_image(image, std = (step_index + 1) * 0.2)
                            noisy_latent_image = self.ae_encoder(noisy_image) # Encode to latent image (96*96*3 -> 6*6*128)
                            noisy_latent_images.append(noisy_latent_image)
                        
                        # Train the model to denoise each noisy latent image to the less noise one
                        for j in range(len(noisy_latent_images)):
                            denoised_latent_image = self.denoiser_model(noisy_latent_images[j])

                            if j == 0: # If the first noisy latent image is the input, the label is the original image
                                less_noisy_latent_image = self.ae_encoder(image)
                            else: # Otherwise, the label is the previous one
                                less_noisy_latent_image = noisy_latent_images[j - 1]

                            loss_value = loss(denoised_latent_image, less_noisy_latent_image)
                            batch_loss += loss_value

                    optimizer.zero_grad()
                    batch_loss.backward() # Calculate gradient
                    optimizer.step() # # Update weights and biases

                    total_loss_training += batch_loss.item()

                    batches_loop.set_postfix_str(f"loss: {total_loss_training/(i + 1):.4f}")
                    
                # Test every 10 epochs
                if epoch % 10 == 0:
                    self.test()

                num_batches_training = len(self.training_data) # The number of mini-batches in training data

                average_loss_in_epoch_training = total_loss_training / num_batches_training
                
                # Validation after each epoch
                self.denoiser_model.eval()

                total_loss_eval = 0
                with torch.no_grad(): # In evaluation mode, avoid calculating gradient and save computational cost
                    for x_batch, y_batch in self.validation_data:
                        x_batch = x_batch.to(self.device)

                        for image in x_batch:
                            # For each image, generate 10 noisy images, from a little bit noisy to almost pure random noise
                            noisy_latent_images = []
                            for i in range(self.denoise_steps):
                                noisy_image = self.add_gaussian_noise_to_image(image, std = (i + 1) * 0.2)
                                noisy_latent_image = self.ae_encoder(noisy_image) # Encode to latent image (96*96*3 -> 6*6*128)
                                noisy_latent_images.append(noisy_latent_image)
                        
                            for j in range(len(noisy_latent_images)):
                                denoised_latent_image = self.denoiser_model(noisy_latent_images[j])

                                if j == 0: # If the first noisy latent image is the input, the label is the original image in latent space
                                    less_noisy_latent_image = self.ae_encoder(image) # Encode to latent image
                                else: # Otherwise, the label is the previous one
                                    less_noisy_latent_image = noisy_latent_images[j - 1]

                                loss_value = loss(denoised_latent_image, less_noisy_latent_image)

                                total_loss_eval += loss_value.item()
                
                num_batches_validation = len(self.validation_data)

                average_loss_in_epoch_validation = total_loss_eval / num_batches_validation

                print(f"loss: {average_loss_in_epoch_training:.4f}")
                print(f"validation loss: {average_loss_in_epoch_validation:.4f}")
            
            # Save model to file
            print(f"Saving model to {self.saved_weights}...")
            torch.save(self.denoiser_model.state_dict(), self.saved_weights)

    # Test on test dataset
    def test(self):
        self.denoiser_model.eval()
        
        # for 
    
    # Generate images from random noise and test the denoised images
    def generate_images(self):
        self.denoiser_model.eval()

        # Generate random noise
        all_images = []
        for i in range(10):
            random_noise = torch.randn(3, self.imsize, self.imsize, device = self.device)

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
        image_count_in_a_row = self.denoise_steps + 1 #int(len(all_images) / (self.denoise_steps + 1))
        torchvision.utils.save_image(all_images, os.path.join(self.saved_images_path, f"task2b_denoised_images.jpg"), nrow = image_count_in_a_row)

    # Now this function is only for test
    def save_image(self):
        image_sample = self.training_set[0][0] # The first [0] is the first (image, label) pair
        noisy_images = []
        # current_noisy_image = image_sample
        # for i in range(10):
        #     current_noisy_image = self.add_gaussian_noise_to_image(current_noisy_image)
        #     noisy_images.append(current_noisy_image)

        for i in range(10):
            noisy_image = self.add_gaussian_noise_to_image(image_sample, std = (i + 1) * 0.2)
            noisy_images.append(noisy_image)

        noisy_images = torch.stack(noisy_images)

        torchvision.utils.save_image(image_sample, os.path.join(self.saved_images_path, f"task2b_original.jpg"), nrow = 1)
        torchvision.utils.save_image(noisy_images, os.path.join(self.saved_images_path, f"task2b_noisy.jpg"), nrow = len(noisy_images))


if __name__ == "__main__":

    load_from_file = True
    imsize = 96
    batch_size = 16
    epochs = 50
    learning_rate = 0.001
    denoise_steps = 20 # How many steps to denoise from random noise to image

    def print_hyper_params():
        print("------------------------------------")
        print(f"imsize:{imsize}")
        print(f"batch_size:{batch_size}")
        print(f"epochs:{epochs}")
        print(f"learning_rate:{learning_rate}")
        print("------------------------------------")

    print_hyper_params()

    trainer = UNetDenoiserTrainer(imsize = imsize, denoise_steps = denoise_steps)
    trainer.load_dataset(fine_grained = False, batch_size = batch_size)
    # trainer.save_image()
    trainer.train(load_from_file = load_from_file, 
                  epochs = epochs, 
                  learning_rate = learning_rate)
    # trainer.test()
    trainer.generate_images()

    print_hyper_params()
