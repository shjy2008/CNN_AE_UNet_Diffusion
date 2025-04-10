import torch
from load_oxford_flowers102 import load_oxford_flowers102
import os
import tqdm
import torchvision

class AE_Encoder(torch.nn.Module):

    def __init__(self, in_channels = 3, imsize = 96):
        super(AE_Encoder, self).__init__()

        # Because we don't use max pooling in AutoEncoder, we use stride = 2 instead of 1 to do downsampling
        # Input: 96*96*3
        self.conv1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = 16, kernel_size = (3, 3), stride = 2, padding = 1)
        # Output: 48*48*16

        self.conv2 = torch.nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = (3, 3), stride = 2, padding = 1)
        # Output: 24*24*32

        self.conv3 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = (3, 3), stride = 2, padding = 1)
        # Output: 12*12*64


    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        x = torch.relu(x)

        x = self.conv3(x)

        return x

class AE_Decoder(torch.nn.Module):

    def __init__(self, out_channels = 3, imsize = 96):
        super(AE_Decoder, self).__init__()

        # Input: 12*12*64
        self.conv_transpose1 = torch.nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = (3, 3), stride = 2, padding = 1, output_padding=1)
        # Output: 24*24*32

        self.conv_transpose2 = torch.nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = (3, 3), stride = 2, padding = 1, output_padding=1)
        # Output: 48*48*16

        self.conv_transpose3 = torch.nn.ConvTranspose2d(in_channels = 16, out_channels = out_channels, kernel_size = (3, 3), stride = 2, padding = 1, output_padding=1)
        # Output: 96*96*3

    def forward(self, x):
        x = self.conv_transpose1(x)
        x = torch.relu(x)

        x = self.conv_transpose2(x)
        x = torch.relu(x)

        x = self.conv_transpose3(x)

        return x


class AutoEncoderTrainer(object):

    def __init__(self, imsize = 96):
        self.imsize = imsize

        # AutoEncoder models
        self.ae_encoder = None
        self.ae_decoder = None

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
        self.saved_weights_encoder = save_base_name + "_torch_autoencoder_encoder.weights.h5"
        self.saved_weights_decoder = save_base_name + "_torch_autoencoder_decoder.weights.h5"
        self.saved_images_path = os.path.join(path_to_save_folder, "task2_images")

        # Device, gpu, mps or cpu
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
    
    def load_local_model(self):
        if self.ae_encoder == None:
            self.ae_encoder = AE_Encoder(in_channels = 3)
            self.ae_encoder.to(self.device)
        if self.ae_decoder == None:
            self.ae_decoder = AE_Decoder(out_channels = 3)
            self.ae_decoder.to(self.device)
        
        print(f"Loading weights from {self.saved_weights_encoder} and {self.saved_weights_decoder}")
        self.ae_encoder.load_state_dict(torch.load(self.saved_weights_encoder, weights_only = True, map_location = self.device))
        self.ae_decoder.load_state_dict(torch.load(self.saved_weights_decoder, weights_only = True, map_location = self.device))

    def train(self, load_from_file = False, epochs = 50, learning_rate = 0.001):
        ae_encoder = AE_Encoder(in_channels = 3, imsize = self.imsize)
        ae_encoder.to(self.device)
        ae_decoder = AE_Decoder(out_channels = 3, imsize = self.imsize)
        ae_decoder.to(self.device)

        self.ae_encoder = ae_encoder
        self.ae_decoder = ae_decoder
        
        if load_from_file and os.path.isfile(self.saved_weights_encoder) and os.path.isfile(self.saved_weights_decoder):
            # Load previous models
            self.load_local_model()
        else:
            # Optimizer
            optimizer = torch.optim.Adam(list(ae_encoder.parameters()) + list(ae_decoder.parameters()), lr = learning_rate)

            loss = torch.nn.MSELoss()

            for epoch in range(1, epochs + 1):
                # Switch to train mode, activate BatchNorm and Dropout
                ae_encoder.train()
                ae_decoder.train()

                total_loss_training = 0
                total_sample_count = 0

                print(f"Epoch {epoch}/{epochs}")
                batches_loop = tqdm.tqdm(enumerate(self.training_data), total = len(self.training_data), bar_format = '{n_fmt}/{total_fmt} [{bar}] - ETA: {remaining}{postfix}', ascii = ' >=')

                for i, (x_batch, y_batch) in batches_loop:
                    x_batch = x_batch.to(self.device)

                    latent_data = ae_encoder(x_batch)
                    reconstructed_imgs = ae_decoder(latent_data)

                    # Calculate loss
                    loss_value = loss(reconstructed_imgs, x_batch)

                    # Update gradients and backpropagation update weights
                    optimizer.zero_grad()
                    loss_value.backward() # Calculate gradient
                    optimizer.step() # Update weights and biases

                    total_loss_training += loss_value.item()

                    total_sample_count += len(x_batch)

                    batches_loop.set_postfix_str(f"loss: {total_loss_training/(i + 1):.4f}")
                    
                # Test every 10 epochs
                if epoch % 10 == 0:
                    self.test()

                num_batches_training = len(self.training_data) # The number of mini-batches in training data
                num_samples_training = len(self.training_set) # The total number of samples in training data

                average_loss_in_epoch_training = total_loss_training / num_batches_training

                # Validation after each epoch
                average_loss_validation = self.get_val_or_test_loss(self.validation_data)

                print(f"loss: {average_loss_in_epoch_training:.4f}")
                print(f"validation loss: {average_loss_validation:.4f}")

            # Save model to file
            print(f"Saving model to {self.saved_weights_encoder} and {self.saved_weights_decoder}...")
            torch.save(ae_encoder.state_dict(), self.saved_weights_encoder)
            torch.save(ae_decoder.state_dict(), self.saved_weights_decoder)
    
    # Get loss on validation or test dataset. dataset: self.validation_set or self.test_set
    def get_val_or_test_loss(self, dataset):
        # Switch to evaluation mode, deactivate BatchNorm and Dropout
        self.ae_encoder.eval()
        self.ae_decoder.eval()

        loss = torch.nn.MSELoss()

        total_loss = 0
        with torch.no_grad():  # In evaluation mode, don't calculate gradient, save computational cost
            for x_batch, y_batch in dataset:
                x_batch = x_batch.to(self.device)

                latent_data = self.ae_encoder(x_batch)
                reconstructed_imgs = self.ae_decoder(latent_data)

                loss_value = loss(reconstructed_imgs, x_batch)
                total_loss += loss_value.item()

        num_batches = len(dataset)
        average_loss_in_epoch = total_loss / num_batches
        return average_loss_in_epoch

    def test(self):
        average_loss_in_test = self.get_val_or_test_loss(self.test_data)
        print(f"test loss: {average_loss_in_test:.4f}")

    # Save recontructed images
    def save_reconstructed_images(self):
        save_image_count = 5

        for i in range(2):
            if i == 0:
                dataset = self.training_set
            elif i == 1:
                dataset = self.test_set
            
            images_sample = [dataset[i][0] for i in range(save_image_count)]
            images_sample = torch.stack(images_sample)
            images_sample = images_sample.to(self.device)

            latent_data = self.ae_encoder(images_sample)
            reconstructed_images = self.ae_decoder(latent_data)

            images_sample.to("cpu").detach()
            reconstructed_images.to("cpu").detach()

            os.makedirs(self.saved_images_path, exist_ok=True)
            torchvision.utils.save_image(images_sample, os.path.join(self.saved_images_path, f"original_{i + 1}.jpg"), nrow = save_image_count)
            torchvision.utils.save_image(reconstructed_images, os.path.join(self.saved_images_path, f"reconstructed_{i + 1}.jpg"), nrow = save_image_count)





if __name__ == "__main__":

    load_from_file = False
    imsize = 96
    batch_size = 16
    epochs = 150
    learning_rate = 0.0001 # 0.00005

    def print_hyper_params():
        print("------------------------------------")
        print(f"load_from_file:{load_from_file}")
        print(f"imsize:{imsize}")
        print(f"batch_size:{batch_size}")
        print(f"epochs:{epochs}")
        print(f"learning_rate:{learning_rate}")
        print("------------------------------------")

    print_hyper_params()

    trainer = AutoEncoderTrainer(imsize = imsize)
    trainer.load_dataset(fine_grained = False, batch_size = batch_size)
    trainer.train(load_from_file = load_from_file,
                  epochs = epochs,
                  learning_rate = learning_rate)
    trainer.test()
    trainer.save_reconstructed_images()

    print_hyper_params()
