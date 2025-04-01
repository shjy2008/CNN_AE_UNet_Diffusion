from load_oxford_flowers102 import load_oxford_flowers102
import torch
import tqdm
import os

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    print(f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%")

class ConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv_list = []
        for i in range(3):
            self.conv_list.append(torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1))

        self.pool = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        for conv in self.conv_list:
            x = conv(x)
            x = torch.relu(x)
        
        x = self.pool(x)

        return x

class CNN(torch.nn.Module):

    def __init__(self, in_channels, n_classes, reg_dropout_rate = 0, reg_batch_norm = False):
        super(CNN, self).__init__()

        # Save regularisation parameters
        self.reg_dropout_rate = reg_dropout_rate
        self.reg_batch_norm = reg_batch_norm

        self.conv_block1 = ConvBlock(in_channels, 64)
        if self.reg_batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(64)

        self.conv_block2 = ConvBlock(64, 128)
        if self.reg_batch_norm:
            self.bn2 = torch.nn.BatchNorm2d(128)
        
        self.conv_blcok3 = ConvBlock(128, 256)
        if self.reg_batch_norm:
            self.bn3 = torch.nn.BatchNorm2d(256)
        
        # # in_channels: 3 (RGB)
        # # input: 32 * 32 * 3 (width * height * in_channels)
        # self.conv1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        # # output: 32 * 32 * 64 neurons
        
        # self.pool1 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # # output: 16 * 16 * 64 neurons

        # self.conv2 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        # # output: 16 * 16 * 128 neurons
        
        # self.pool2 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # # output: 8 * 8 * 128 neurons

        # self.conv3 = torch.nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1)
        # # output: 8 * 8 * 256 neurons
        
        # self.pool3 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # # output: 4 * 4 * 256 neurons

        self.flatten = torch.nn.Flatten() # A multi-dimensional feature map -> a 1D vector
        # output: 4 * 4 * 256 = 4096 neurons

        self.fc1 = torch.nn.Linear(4096, 128)
        # output: 128 neurons

        self.fc2 = torch.nn.Linear(128, 512)
        # output: 512 neurons

        if self.reg_dropout_rate > 0:
            self.dropout1 = torch.nn.Dropout(self.reg_dropout_rate)

        self.fc3 = torch.nn.Linear(512, n_classes)
        # output: n_classes neurons

    def forward(self, x):
        x = self.conv_block1(x)
        if self.reg_batch_norm:
            x = self.bn1(x)
        
        x = self.conv_block2(x)
        if self.reg_batch_norm:
            x = self.bn2(x)
        
        x = self.conv_block3(x)
        if self.reg_batch_norm:
            x = self.bn3(x)
        

        # x = self.conv1(x)
        # x = torch.relu(x)
        
        # x = self.pool1(x)
        
        # x = self.conv2(x)
        # x = torch.relu(x)

        # x = self.pool2(x)

        # x = self.conv3(x)
        # x = torch.relu(x)

        # x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        if self.reg_dropout_rate > 0:
            x = self.dropout1(x)

        x = self.fc3(x)

        return x
    
class ModelTrainer(object):

    def __init__(self):
        self.cnn = None

        self.training_set = None
        self.validation_set = None
        self.test_set = None
        self.class_names = None

        self.training_data = None
        self.validation_data = None
        self.test_data = None

        # Save file configs
        self.load_from_file = True
        path_to_this_scripts_folder = os.path.dirname(os.path.realpath(__file__))
        path_to_save_folder = os.path.join(path_to_this_scripts_folder, "saved")
        if not os.path.isdir(path_to_save_folder):
            os.mkdir(path_to_save_folder)
        save_base_name = os.path.join(path_to_save_folder, "oxford_flowers")
        self.saved_weights = save_base_name + "_torch_cnn_net.weights.h5"

        # Device, gpu, mps or cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"device: {self.device}")

    def load_dataset(self):
        self.training_set, self.validation_set, self.test_set, self.class_names = load_oxford_flowers102(imsize = 32, fine = True)
        self.training_data = torch.utils.data.DataLoader(self.training_set, batch_size = 16, shuffle = True)
        self.validation_data = torch.utils.data.DataLoader(self.validation_set, batch_size = 16, shuffle = False)
        self.test_data = torch.utils.data.DataLoader(self.test_set, batch_size = 16, shuffle = False)

        label_to_count = {}
        for x_batch, y_batch in self.training_data:
            for y in y_batch:
                label = y.item()
                if label in label_to_count:
                    label_to_count[label] += 1
                else:
                    label_to_count[label] = 1
        
        # coarse: {8: 291, 2: 164, 3: 798, 7: 103, 4: 368, 1: 49, 9: 97, 5: 201, 6: 136, 0: 115}
        # fine-grained: {76: 231, 45: 176, 22: 71, 85: 38, 74: 100, 37: 36, 49: 72, 9: 25, 4: 45, 91: 46,
        #  28: 58, 79: 85, 51: 65, 88: 164, 67: 34, 84: 43, 36: 88, 55: 89, 94: 108, 42: 110, 80: 146, 57: 94,
        #  77: 117, 89: 62, 87: 134, 46: 47, 73: 151, 93: 142, 13: 28, 97: 62, 50: 238, 71: 76, 92: 26, 40: 107,
        #  52: 73, 64: 82, 53: 41, 27: 46, 72: 174, 18: 29, 100: 38, 7: 65, 35: 55, 2: 20, 43: 73, 31: 25, 59: 89,
        #  60: 30, 63: 32, 54: 51, 29: 65, 83: 66, 11: 67, 10: 67, 96: 46, 17: 62, 61: 35, 82: 111, 81: 92, 14: 29,
        #  75: 87, 21: 39, 86: 43, 26: 20, 47: 51, 66: 22, 90: 56, 58: 47, 16: 65, 15: 21, 62: 34, 98: 43, 19: 36,
        #  32: 26, 78: 21, 68: 34, 69: 42, 1: 40, 39: 47, 5: 25, 8: 26, 41: 39, 20: 20, 65: 41, 99: 29, 70: 58, 95: 71,
        #  48: 29, 30: 32, 56: 47, 25: 21, 23: 22, 12: 29, 24: 21, 33: 20, 3: 36, 44: 20, 34: 23, 0: 20, 38: 21, 101: 28, 6: 20}
        print(f"label_to_count: {label_to_count}")

        # TODO: weighted loss
    
    def train(self):
        cnn = CNN(in_channels = 3, n_classes = len(self.class_names))
        cnn.to(self.device)

        self.cnn = cnn

        if self.load_from_file and os.path.isfile(self.saved_weights):
            # Load previous model
            print(f"Loading weights from {self.saved_weights}")
            cnn.load_state_dict(torch.load(self.saved_weights, weights_only = True))
        else:
            optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.001)

            loss = torch.nn.CrossEntropyLoss()

            history = {"loss": [], "accuracy": [], "validation_loss": [], "validation_accuracy": []}

            epochs = 50
            for epoch in range(1, epochs + 1):

                total_loss_training = 0
                total_correct_prediction_training = 0
                total_sample_count = 0

                print(f"Epoch {epoch}/{epochs}")
                batches_loop = tqdm.tqdm(enumerate(self.training_data), total = len(self.training_data), bar_format = '{n_fmt}/{total_fmt} [{bar}] - ETA: {remaining}{postfix}', ascii = ' >=')

                for b, (x_batch, y_batch) in batches_loop:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    y_predict = cnn(x_batch)

                    loss_value = loss(y_predict, y_batch)
                    
                    optimizer.zero_grad()

                    loss_value.backward()

                    optimizer.step()

                    total_loss_training += loss_value.item()
                    y_predict = torch.argmax(y_predict, dim = 1)
                    total_correct_prediction_training += torch.sum(y_predict == y_batch).item()
                    total_sample_count += len(y_batch)

                    batches_loop.set_postfix_str(f"loss: {total_loss_training/(b + 1):.4f} - accuracy: {total_correct_prediction_training/total_sample_count:.4f}")


                num_batches_training = len(self.training_data) # The number of mini-batches in training data
                num_samples_training = len(self.training_set) # The total number of samples in training data

                average_loss_in_epoch_training = total_loss_training / num_batches_training
                accuracy_in_epoch_training = total_correct_prediction_training / num_samples_training

                # Validation
                total_loss_validation = 0
                total_correct_predictions_validation = 0
                for x_batch, y_batch in self.validation_data:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    y_predict = cnn(x_batch)
                    total_loss_validation += loss(y_predict, y_batch).item()
                    y_predict = torch.argmax(y_predict, dim = 1)
                    total_correct_predictions_validation += torch.sum(y_predict == y_batch).item()

                num_batches_validation = len(self.validation_data)
                num_samples_validation = len(self.validation_set)

                average_loss_in_epoch_validation = total_loss_validation / num_batches_validation
                accuracy_in_epoch_validation = total_correct_predictions_validation / num_samples_validation

                print(f"loss: {average_loss_in_epoch_training:.4f} - accuracy: {accuracy_in_epoch_training:.4f}")
                print(f"validation loss: {average_loss_in_epoch_validation:.4f} - validation accuracy: {accuracy_in_epoch_validation:.4f}")
        
            # Save model to file
            print(f"Saving model to {self.saved_weights}...")
            torch.save(cnn.state_dict(), self.saved_weights)

    def test(self):
        accuracy_test = 0
        for x_batch, y_batch in self.test_data:
            # Must move the data to the device, because the model is on the device
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            y_pred = self.cnn(x_batch)
            y_pred = torch.argmax(y_pred, dim=1)
            accuracy_test += torch.sum(y_pred == y_batch).item()

            accuracy_test /= len(self.test_set)
            print(f'Test accuracy : {accuracy_test:.2f}')


if __name__ == "__main__":
    trainer = ModelTrainer()
    # trainer.load_dataset()
    # trainer.train()
    # trainer.test()

    cnn = CNN(in_channels = 3, n_classes = 10, reg_dropout_rate = 0.4, reg_batch_norm = True)
    print_number_of_trainable_model_parameters(cnn) # should not be more than 15 million (15,000,000)


        
