import torch
import os
from load_oxford_flowers102 import load_oxford_flowers102
import tqdm
import torchvision
from task2a import AutoEncoderTrainer
from PIL import Image
import random

class UNetDenoiser(torch.nn.Module):
    def __init__(self):
        super(UNetDenoiser, self).__init__()

        # -----Downsampling----- #
        # Input: 12*12*64
        self.down1_1 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 12*12*128

        self.down1_2 = torch.nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 12*12*128

        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # Output: 6*6*128


        # Input: 6*6*128
        self.down2_1 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 6*6*256

        self.down2_2 = torch.nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 6*6*256

        self.pool2 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # Output: 3*3*256


        # -----Bottleneck----- #
        self.bottleneck1 = torch.nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 3*3*512
        
        self.bottleneck2 = torch.nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 3*3*256


        # -----Upsampling----- #
        self.up2_1 = torch.nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = (3, 3), stride = 2, padding = 1, output_padding = 1)
        # Output: 6*6*256

        # Input: the output of down2_2 concatenate the output of up2_1
        self.up2_2 = torch.nn.Conv2d(in_channels = (256 + 256), out_channels = 128, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 6*6*128

        self.up1_1 = torch.nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = (3, 3), stride = 2, padding = 1, output_padding = 1)
        # Output: 12*12*128

        # Input: the output of down1_2 concatenate the output of up1_1
        self.up1_2 = torch.nn.Conv2d(in_channels = (128 + 128), out_channels = 64, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 12*12*64

        self.output = torch.nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (3, 3), stride = 1, padding = 1)
        # Output: 12*12*64
    
    
    def forward(self, x):
        is_batch = x.ndim == 4
        if not is_batch:
            x = x.unsqueeze(0) # (batch_size(1), channels, w, h)

        # -----Downsampling----- #
        x = self.down1_1(x)
        x = torch.relu(x)

        x = self.down1_2(x)
        x = torch.relu(x)
        x1_2 = x

        x = self.pool1(x)


        x = self.down2_1(x)
        x = torch.relu(x)

        x = self.down2_2(x)
        x = torch.relu(x)
        x2_2 = x

        x = self.pool2(x)


        # -----Bottleneck----- #
        x = self.bottleneck1(x)
        x = torch.relu(x)

        x = self.bottleneck2(x)
        x = torch.relu(x)


        # -----Upsampling----- #
        x = self.up2_1(x)
        x = torch.relu(x)

        concat_dim = 1
        if x.ndim == 3: # Only process one latent image, x.shape like (256, 6, 6)
            concat_dim = 0
        elif x.ndim == 4: # Batch processing, x.shape like (batch_size, 256, 6, 6)
            concat_dim = 1
        x = self.up2_2(torch.concat((x2_2, x), dim = concat_dim)) # Skip connection: pass the output of x2_2 to up2_2
        x = torch.relu(x)


        x = self.up1_1(x)
        x = torch.relu(x)

        x = self.up1_2(torch.concat((x1_2, x), dim = concat_dim)) # Skip connection: pass the output of x1_2 to up1_2
        x = torch.relu(x)


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
        save_base_name = os.path.join(path_to_save_folder, "oxford_flowers")
        # self.saved_weights = save_base_name + "_torch_denoiser_1e4_20_random_step.weights.h5"
        # self.saved_weights = save_base_name + "_torch_denoiser_1e4_20_brightloss_std05.weights.h5"
        self.saved_weights = save_base_name + "_torch_denoiser_1e4_20_brightloss_std005i+1.weights.h5"
        # self.saved_weights = save_base_name + "_torch_denoiser_1e4_20.weights.h5"
        # self.saved_weights = save_base_name + "_torch_denoiser_1e4_20_brightloss_std005i+1_randomstep.weights.h5"
        # self.saved_weights = save_base_name + "_torch_denoiser_test_2.weights.h5"
        # self.saved_weights = save_base_name + "_torch_denoiser_test.weights.h5"
        self.saved_images_path = os.path.join(path_to_save_folder, "task2_images")
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
        std = std
        mean = mean # 0: Pure noise, no brightness change
        noise = torch.randn(image.size(), device = self.device) * std + mean
        return image + noise

    def train(self, load_from_file = False, epochs = 50, learning_rate = 0.001):
        self.denoiser_model = UNetDenoiser()
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

                # # Test training the first image only
                # image_samples = [
                #                   self.training_set[0][0].to(self.device),
                #                   self.training_set[100][0].to(self.device),
                #                   self.training_set[200][0].to(self.device),
                #                   self.training_set[300][0].to(self.device),
                #                   self.training_set[400][0].to(self.device),
                #                   self.training_set[600][0].to(self.device)
                #                   ]
                
                # inputs = []
                # labels = []
                # for image_sample in image_samples:
                #     # Train only 1 image, 10 time steps (from very noisy to less noisy)
                #     noisy_latent_images = []
                #     noisy_images = self.create_noisy_images_tensor(image_sample)
                #     for noisy_image in noisy_images:
                #         noisy_latent_image = self.ae_encoder(noisy_image)
                #         noisy_latent_images.append(noisy_latent_image)

                #     for i in range(1):
                #         for j in range(len(noisy_latent_images) - 1):
                #             input = noisy_latent_images[j + 1]
                #             label = noisy_latent_images[j]

                #             inputs.append(input)
                #             labels.append(label)

                # Train only 1 image, fixed 1 input and 1 output
                # batch_loss = 0
                # input_img_path = os.path.join(self.saved_images_path, f"task2b_test_input.jpg")
                # input = torchvision.transforms.ToTensor()(Image.open(input_img_path).convert('RGB'))
                # input = input.to(self.device)
                # input = self.ae_encoder(input)
                # inputs.append(input)

                # label_img_path = os.path.join(self.saved_images_path, f"task2b_test_label.jpg")
                # label = torchvision.transforms.ToTensor()(Image.open(label_img_path).convert('RGB'))
                # label = label.to(self.device)
                # label = self.ae_encoder(label)
                # labels.append(label)



                # batch_loss = 0
                
                # inputs = torch.stack(inputs)
                # labels = torch.stack(labels)

                # outputs = self.denoiser_model(inputs)

                # loss_value = loss(outputs, labels)
                # batch_loss += loss_value

                # optimizer.zero_grad()
                # batch_loss.backward() # Calculate gradient
                # optimizer.step() # # Update weights and biases

                # total_loss_training += batch_loss.item()

                

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
                    
                # Test every 10 epochs
                if epoch % 10 == 0:
                    self.test()

                num_batches_training = len(self.training_data) # The number of mini-batches in training data

                average_loss_in_epoch_training = total_loss_training / num_batches_training
                
                # Validation after each epoch
                average_loss_validation = self.get_val_or_test_loss(self.validation_data)

                print(f"loss: {average_loss_in_epoch_training:.4f}")
                print(f"validation loss: {average_loss_validation:.4f}")
            
            # Save model to file
            print(f"Saving model to {self.saved_weights}...")
            torch.save(self.denoiser_model.state_dict(), self.saved_weights)
    
    # Get loss on validation or test dataset. dataset: self.validation_set or self.test_set
    def get_val_or_test_loss(self, dataset):
        print("get_val_or_test_loss() dataset: ", dataset)
        self.denoiser_model.eval()

        loss = torch.nn.MSELoss()

        total_loss = 0
        with torch.no_grad(): # In evaluation mode, avoid calculating gradient and save computational cost
            for x_batch, y_batch in dataset:
                x_batch = x_batch.to(self.device)

                inputs = []
                labels = []
                batch_loss = 0
                for image in x_batch:
                    # For each image, generate 10 noisy images, from a little bit noisy to almost pure random noise
                    noisy_latent_images = []
                    noisy_images = self.create_noisy_images_tensor(image)
                    for noisy_image in noisy_images:
                        noisy_latent_image = self.ae_encoder(noisy_image) # Encode to latent image
                        noisy_latent_images.append(noisy_latent_image)
                
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

                total_loss += batch_loss.item()
        
        num_batches = len(dataset)
        average_loss_in_epoch = total_loss / num_batches
        return average_loss_in_epoch

    # Test on test dataset
    def test(self):
        print("Start testing...")
        average_loss_test = self.get_val_or_test_loss(self.test_data)
        print(f"test loss: {average_loss_test:.4f}")

    # Some images are dark, preserve brightness when denoising
    def preserve_brightness(self, original_image, generated_image):
        original_mean = original_image.mean()
        current_mean = generated_image.mean()
        generated_image = generated_image * (original_mean / current_mean)
        return generated_image

    def generate_images_test(self):
        self.denoiser_model.eval()

        # image_sample = self.training_set[0][0].to(self.device)
        image_sample = torch.randn(3, self.imsize, self.imsize, device = self.device)
        
        noisy_images = self.create_noisy_images_tensor(image_sample)

        noisy_latent_images = []
        for noisy_image in noisy_images:
            noisy_latent_image = self.ae_encoder(noisy_image)
            noisy_latent_images.append(noisy_latent_image)
        input = noisy_latent_images[-1]
        input.to(self.device)
        # input = self.ae_encoder(input)
        
        # input_img_path = os.path.join(self.saved_images_path, f"task2b_test_input.jpg")
        # input = torchvision.transforms.ToTensor()(Image.open(input_img_path).convert('RGB'))
        # input = input.to(self.device)
        # input = self.ae_encoder(input)

        denoised_images = []
        denoised_latent_image = input
        for i in range(50):
            denoised_latent_image = self.denoiser_model(denoised_latent_image)
            denoised_latent_image = self.preserve_brightness(input, denoised_latent_image)
            denoised_image = self.ae_decoder(denoised_latent_image)
            denoised_images.append(denoised_image)
        denoised_images = torch.stack(denoised_images)

        torchvision.utils.save_image(denoised_images, os.path.join(self.saved_images_path, f"task2b_denoised_images_test.jpg"), nrow = 10)
        

    # Generate images from random noise and test the denoised images
    def generate_images(self):
        self.denoiser_model.eval()

        # Generate random noise
        all_images = []
        for i in range(self.denoise_steps):
            random_noise = torch.randn(3, self.imsize, self.imsize, device = self.device) * 2.8
            # random_noise = self.training_set[0][0].to(self.device)
            # random_noise = self.add_gaussian_noise_to_image(random_noise, std = 2.0)
            # random_noise = self.create_noisy_images_tensor(random_noise)[3]

            # Encode the random noise
            random_latent_noise = self.ae_encoder(random_noise)

            # Denoise in the latent space for multiple times
            denoised_images = [random_noise]
            current_denoised_latent_image = random_latent_noise
            for step_index in range(self.denoise_steps):
                current_denoised_latent_image = self.denoiser_model(current_denoised_latent_image)
                current_denoised_latent_image = self.preserve_brightness(random_latent_noise, current_denoised_latent_image)
                current_denoised_image = self.ae_decoder(current_denoised_latent_image) # Decode from latent space back to an image
                denoised_images.append(current_denoised_image)
            
            all_images.extend(denoised_images)
            
        # Save and check the resulting images
        image_count_in_a_row = self.denoise_steps + 1 #int(len(all_images) / (self.denoise_steps + 1))
        torchvision.utils.save_image(all_images, os.path.join(self.saved_images_path, f"task2b_denoised_images.jpg"), nrow = image_count_in_a_row)

    def save_image_test(self):
        image_sample = self.training_set[100][0].to(self.device)
        noisy_latent_images = []
        noisy_images = self.create_noisy_images_tensor(image_sample)
        for noisy_image in noisy_images:
            noisy_latent_image = self.ae_encoder(noisy_image)
            noisy_latent_images.append(noisy_latent_image)

        # for i in range(1):
        input = self.ae_decoder(noisy_latent_images[-1])
        label = self.ae_decoder(noisy_latent_images[-8])
        
        torchvision.utils.save_image(input, os.path.join(self.saved_images_path, f"task2b_test_input.jpg"), nrow = 1)
        torchvision.utils.save_image(label, os.path.join(self.saved_images_path, f"task2b_test_label.jpg"), nrow = 1)
        

    # Now this function is only for test
    def save_image(self):
        image_sample = self.training_set[600][0].to(self.device) # The first [0] is the first (image, label) pair
        # noisy_images = [image_sample]

        # noisy_image = image_sample
        # for i in range(10):
        #     noisy_image = self.add_gaussian_noise_to_image(noisy_image, std = (i + 1) * 0.2)
        #     noisy_images.append(noisy_image)

        # noisy_images = torch.stack(noisy_images)

        noisy_images = self.create_noisy_images_tensor(image_sample)

        torchvision.utils.save_image(image_sample, os.path.join(self.saved_images_path, f"task2b_original.jpg"), nrow = 1)
        torchvision.utils.save_image(noisy_images, os.path.join(self.saved_images_path, f"task2b_noisy.jpg"), nrow = len(noisy_images))

    def create_noisy_images_tensor_1(self, original_image):
        noisy_images = [original_image]
        # current_image = original_image
        for i in range(self.denoise_steps):
            std = (i + 1) * 0.1
            noisy_image = self.add_gaussian_noise_to_image(original_image, std = std)
            noisy_images.append(noisy_image)
        noisy_image = torch.stack(noisy_images)
        return noisy_images

    # Create (self.denoise_steps + 1) images, the first element is the original image, the last one is almost 'pure' random noise
    def create_noisy_images_tensor(self, original_image):
        noisy_images = [original_image]
        noisy_image = original_image
        for i in range(self.denoise_steps):
            std = 0.05 * (i + 1)
            # if i == self.denoise_steps - 1: # The last one, add super large noise
            #     std = 100
            noisy_image = self.add_gaussian_noise_to_image(noisy_image, std = std)
            noisy_images.append(noisy_image)
        noisy_image = torch.stack(noisy_images)
        return noisy_images

if __name__ == "__main__":

    load_from_file = False
    imsize = 96
    batch_size = 16
    epochs = 20
    learning_rate = 0.0001
    denoise_steps = 20 # How many steps to denoise from random noise to image

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
    trainer.train(load_from_file = load_from_file, 
                  epochs = epochs, 
                  learning_rate = learning_rate)
    # # trainer.test()
    trainer.generate_images_test()
    trainer.generate_images()
    # trainer.save_image_test()
    trainer.save_image()

    print_hyper_params()
