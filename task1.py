import torch.utils
import torch.utils.data
import torchvision
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

class CNN(torch.nn.Module):

    def __init__(self, in_channels, n_classes, reg_dropout_rate = 0, reg_batch_norm = False):
        super(CNN, self).__init__()

        # Save regularisation parameters
        self.reg_dropout_rate = reg_dropout_rate
        self.reg_batch_norm = reg_batch_norm

        out_channels_1 = 64
        out_channels_2 = 128
        out_channels_3 = 256
        out_channels_4 = 128
        out_channels_5 = 256
        
        # in_channels: 3 (RGB)
        # input: 32 * 32 * 3 (width * height * in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels = in_channels, out_channels = out_channels_1, kernel_size = 3, stride = 1, padding = 1)
        # output: 32 * 32 * out_channels_1 neurons
        self.conv1_1 = torch.nn.Conv2d(in_channels = out_channels_1, out_channels = out_channels_1, kernel_size = 3, stride = 1, padding = 1)

        if self.reg_batch_norm:
            self.bn1 = torch.nn.BatchNorm2d(out_channels_1)
        
        self.pool1 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # output: 16 * 16 * out_channels_1 neurons

        self.conv2 = torch.nn.Conv2d(in_channels = out_channels_1, out_channels = out_channels_2, kernel_size = 3, stride = 1, padding = 1)
        # output: 16 * 16 * out_channels_2 neurons
        self.conv2_1 = torch.nn.Conv2d(in_channels = out_channels_2, out_channels = out_channels_2, kernel_size = 3, stride = 1, padding = 1)
        
        if self.reg_batch_norm:
            self.bn2 = torch.nn.BatchNorm2d(out_channels_2)

        self.pool2 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # output: 8 * 8 * out_channels_2 neurons

        self.conv3 = torch.nn.Conv2d(in_channels = out_channels_2, out_channels = out_channels_3, kernel_size = 3, stride = 1, padding = 1)
        # output: 8 * 8 * out_channels_3 neurons
        self.conv3_1 = torch.nn.Conv2d(in_channels = out_channels_3, out_channels = out_channels_3, kernel_size = 3, stride = 1, padding = 1)
        
        if self.reg_batch_norm:
            self.bn3 = torch.nn.BatchNorm2d(out_channels_3)

        self.pool3 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # output: 4 * 4 * out_channels_3 neurons

        # self.conv4 = torch.nn.Conv2d(in_channels = out_channels_3, out_channels = out_channels_4, kernel_size = 3, stride = 1, padding = 1)
        # # output: 8 * 8 * out_channels_4 neurons
        # self.conv4_1 = torch.nn.Conv2d(in_channels = out_channels_4, out_channels = out_channels_4, kernel_size = 3, stride = 1, padding = 1)
        
        # if self.reg_batch_norm:
        #     self.bn4 = torch.nn.BatchNorm2d(out_channels_4)

        # self.pool4 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # # output: 4 * 4 * out_channels_3 neurons

        # self.conv5 = torch.nn.Conv2d(in_channels = out_channels_4, out_channels = out_channels_5, kernel_size = 3, stride = 1, padding = 1)
        # # output: 8 * 8 * out_channels_5 neurons
        # self.conv5_1 = torch.nn.Conv2d(in_channels = out_channels_5, out_channels = out_channels_5, kernel_size = 3, stride = 1, padding = 1)
        
        # if self.reg_batch_norm:
        #     self.bn5 = torch.nn.BatchNorm2d(out_channels_5)

        # self.pool5 = torch.nn.MaxPool2d(kernel_size = 2, stride = 2)
        # # output: 4 * 4 * out_channels_3 neurons

        self.flatten = torch.nn.Flatten() # A multi-dimensional feature map -> a 1D vector
        # output: 4 * 4 * out_channels_3 = 4096 neurons

        neurons_after_conv_layers = 4096 #8192 # 512 # 4608 # 4096 # 36864
        self.fc1 = torch.nn.Linear(neurons_after_conv_layers, 128)
        # output: 128 neurons

        if self.reg_batch_norm:
            self.bn_fc1 = torch.nn.BatchNorm1d(128)

        self.fc2 = torch.nn.Linear(128, 512)
        # output: 512 neurons

        if self.reg_dropout_rate > 0:
            self.dropout1 = torch.nn.Dropout(self.reg_dropout_rate)

        self.fc3 = torch.nn.Linear(512, n_classes)
        # output: n_classes neurons

    def forward(self, x):
        x = self.conv1(x)
        if self.reg_batch_norm:
            x = self.bn1(x)
        x = torch.relu(x)
        # x = self.conv1_1(x)
        # x = torch.relu(x)
        
        x = self.pool1(x)
        
        x = self.conv2(x)
        if self.reg_batch_norm:
            x = self.bn2(x)
        x = torch.relu(x)
        # x = self.conv2_1(x)
        # x = torch.relu(x)

        x = self.pool2(x)

        x = self.conv3(x)
        if self.reg_batch_norm:
            x = self.bn3(x)
        x = torch.relu(x)
        # x = self.conv3_1(x)
        # x = torch.relu(x)

        x = self.pool3(x)
        
        # x = self.conv4(x)
        # if self.reg_batch_norm:
        #     x = self.bn4(x)
        # x = torch.relu(x)
        # # x = self.conv4_1(x)
        # # x = torch.relu(x)

        # x = self.pool4(x)
        
        # x = self.conv5(x)
        # if self.reg_batch_norm:
        #     x = self.bn5(x)
        # x = torch.relu(x)
        # # x = self.conv5_1(x)
        # # x = torch.relu(x)

        # x = self.pool5(x)

        x = self.flatten(x)

        x = self.fc1(x)
        if self.reg_batch_norm:
            x = self.bn_fc1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = torch.relu(x)

        if self.reg_dropout_rate > 0:
            x = self.dropout1(x)

        x = self.fc3(x)

        return x
    

class AugmentedDataset(torch.utils.data.Dataset):

    def __init__(self, img_and_labels):
        self.img_and_labels = img_and_labels
    
    def __getitem__(self, idx):
        return self.img_and_labels[idx]

    def __len__(self):
        return len(self.img_and_labels)
    
class ModelTrainer(object):

    def __init__(self, reg_dropout_rate = 0, reg_batch_norm = False, reg_wdecay_beta = 0):
        self.cnn = None

        # Regularisation parameters
        self.reg_dropout_rate = reg_dropout_rate
        self.reg_batch_norm = reg_batch_norm
        self.reg_wdecay_beta = reg_wdecay_beta

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
        self.saved_weights = save_base_name + "_torch_cnn_net.weights.h5"

        # Device, gpu, mps or cpu
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"device: {self.device}")

    def load_dataset(self, fine_grained = False, imsize = 96, batch_size = 16, data_augmentation = True):
        self.training_set, self.validation_set, self.test_set, self.class_names = load_oxford_flowers102(imsize = imsize, fine = fine_grained)

        # print(len(self.training_set), len(self.validation_set), len(self.test_set))

        # Data augmentation
        if data_augmentation:
            print("Start preparing data...")
            transform_func = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(), # Horizontal flip
                # torchvision.transforms.Pad(padding = 10, padding_mode = "edge"), # Pad
                torchvision.transforms.RandomRotation(degrees = 25),
                torchvision.transforms.ColorJitter(brightness = 0.3, contrast = 0.3, saturation = 0.3, hue = 0.1),
                
                torchvision.transforms.RandomAffine(degrees = 15, translate = (0.1, 0.1)), # Affine
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

                    # FOR TEST
                    # augmented_img_pil = torchvision.transforms.ToPILImage()(new_img)
                    # augmented_img_pil.save('augmented_image.jpg')
                    # pil_img.save('original_image.jpg')
            
            augmented_dataset = AugmentedDataset(img_and_labels = img_and_labels)
            self.training_set = torch.utils.data.ConcatDataset([self.training_set, augmented_dataset])

        # Use DataLoader to load data into batches
        self.training_data = torch.utils.data.DataLoader(self.training_set, batch_size = batch_size, shuffle = True)
        self.validation_data = torch.utils.data.DataLoader(self.validation_set, batch_size = batch_size, shuffle = False)
        self.test_data = torch.utils.data.DataLoader(self.test_set, batch_size = batch_size, shuffle = False)

        # TODO: for test CIFAR10
        # transform_train = torchvision.transforms.ToTensor()
        # path_to_data_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data_CIFAR10')
        # training_set = torchvision.datasets.CIFAR10(root=path_to_data_folder, train=True, download=True, transform=transform_train)
        # self.test_set = torchvision.datasets.CIFAR10(root=path_to_data_folder, train=False, download=True, transform=torchvision.transforms.ToTensor())
        # self.training_set, self.validation_set = torch.utils.data.random_split(training_set, [45000, 5000])

        # self.training_data = torch.utils.data.DataLoader(self.training_set, batch_size=100, shuffle=True)
        # self.validation_data = torch.utils.data.DataLoader(self.validation_set, batch_size=100, shuffle=False)
        # self.test_data = torch.utils.data.DataLoader(self.test_set, batch_size=100, shuffle=False)
        # self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
        #        'dog', 'frog', 'horse', 'ship', 'truck']

        # label_to_count = {}
        # for x_batch, y_batch in self.training_data:
        #     for y in y_batch:
        #         label = y.item()
        #         if label in label_to_count:
        #             label_to_count[label] += 1
        #         else:
        #             label_to_count[label] = 1
        
        # print(f"label_to_count: {label_to_count}")

        # TODO: weighted loss
    
    def train(self, load_from_file = False, epochs = 50, learning_rate = 0.001):
        cnn = CNN(in_channels = 3, 
                  n_classes = len(self.class_names),
                  reg_dropout_rate = self.reg_dropout_rate,
                  reg_batch_norm = self.reg_batch_norm)
        cnn.to(self.device)

        print_number_of_trainable_model_parameters(cnn) # should not be more than 15 million (15,000,000)

        self.cnn = cnn

        if load_from_file and os.path.isfile(self.saved_weights):
            # Load previous model
            print(f"Loading weights from {self.saved_weights}")
            cnn.load_state_dict(torch.load(self.saved_weights, weights_only = True))
        else:

            # Optimizer
            if self.reg_wdecay_beta:
                weight_decay_params = []
                no_weight_decay_params = []
                for name, p in cnn.named_parameters():
                    if "fc1.weight" in name or "fc2.weight" in name:
                        weight_decay_params.append(p)
                    else:
                        no_weight_decay_params.append(p)

                optimizer = torch.optim.Adam([{"params": weight_decay_params, "weight_decay": self.reg_wdecay_beta},
                                              {"params": no_weight_decay_params}
                                              ], lr = learning_rate)
            
            else:
                optimizer = torch.optim.Adam(cnn.parameters(), lr = learning_rate)

            loss = torch.nn.CrossEntropyLoss()

            history = {"loss": [], "accuracy": [], "validation_loss": [], "validation_accuracy": []}

            for epoch in range(1, epochs + 1):

                total_loss_training = 0
                total_correct_prediction_training = 0
                total_sample_count = 0

                print(f"Epoch {epoch}/{epochs}")
                batches_loop = tqdm.tqdm(enumerate(self.training_data), total = len(self.training_data), bar_format = '{n_fmt}/{total_fmt} [{bar}] - ETA: {remaining}{postfix}', ascii = ' >=')

                for i, (x_batch, y_batch) in batches_loop:
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    y_predict = cnn(x_batch)

                    # Calculate loss
                    loss_value = loss(y_predict, y_batch)
                    
                    # Update gradients and backpropagation update weights
                    optimizer.zero_grad()
                    loss_value.backward()
                    optimizer.step()

                    total_loss_training += loss_value.item()
                    y_predict = torch.argmax(y_predict, dim = 1)
                    total_correct_prediction_training += torch.sum(y_predict == y_batch).item()
                    total_sample_count += len(y_batch)

                    batches_loop.set_postfix_str(f"loss: {total_loss_training/(i + 1):.4f} - accuracy: {total_correct_prediction_training/total_sample_count:.4f}")

                # Test every 10 epochs
                if epoch % 10 == 0:
                    self.test()

                num_batches_training = len(self.training_data) # The number of mini-batches in training data
                num_samples_training = len(self.training_set) # The total number of samples in training data

                average_loss_in_epoch_training = total_loss_training / num_batches_training
                accuracy_in_epoch_training = total_correct_prediction_training / num_samples_training

                # Validation after each epoch
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
    cnn = CNN(in_channels = 3, n_classes = 10, reg_dropout_rate = 0.4, reg_batch_norm = True)
    print_number_of_trainable_model_parameters(cnn) # should not be more than 15 million (15,000,000)

    reg_dropout_rate = 0 # 0.3
    reg_batch_norm = False
    reg_wdecay_beta = 0 # 0.001
    fine_grained = False
    imsize = 32 # 96 # 32
    batch_size = 16
    data_augmentation = False
    epochs = 50
    learning_rate = 0.001 # 0.00005

    def print_hyper_params():
        print("------------------------------------")
        print(f"reg_dropout_rate:{reg_dropout_rate}")
        print(f"reg_batch_norm:{reg_batch_norm}")
        print(f"reg_wdecay_beta:{reg_wdecay_beta}")
        print(f"fine_grained:{fine_grained}")
        print(f"imsize:{imsize}")
        print(f"batch_size:{batch_size}")
        print(f"data_augmentation:{data_augmentation}")
        print(f"epochs:{epochs}")
        print(f"learning_rate:{learning_rate}")
        print("------------------------------------")
    
    print_hyper_params()

    trainer = ModelTrainer(reg_dropout_rate = reg_dropout_rate,
                           reg_batch_norm = reg_batch_norm,
                           reg_wdecay_beta = reg_wdecay_beta)
    
    trainer.load_dataset(fine_grained = fine_grained,
                         imsize = imsize,
                         batch_size = batch_size,
                         data_augmentation = data_augmentation)
    
    trainer.train(load_from_file = False,
                  epochs = epochs,
                  learning_rate = learning_rate)
    
    trainer.test()

    print_hyper_params()



        
