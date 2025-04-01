__author__ = "Lech Szymanski"
__organization__ = "COSC420, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import tqdm
import os

"""
The original Oxford flowers102 dataset (https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) consists of colour images of 102 
different types of flowers.

This dataaset provides a train, validation, and test split of the data in PyTorch Dataset format. You can pass each dataset into PyTorch data loader to get the images and labels.  See example in the main method for use of the load_oxford_flowers102 method.

The fine labelled set has 102 classes, one for each type of flower, while the coarse labelled set has 10 classes, each containing a group of flowers.  There are 6149/1020/1020 images in the fine labelled training/validation/test sets; there are 2322/390/390 images in the coarse labelled training/validation/test sets.


To use this script in another, just drop this file in the same folder and then, you can invoke it from the other
script like so:

from load_oxford_flowers102 import load_oxford_flowers102

train_data, validation_data, test_data, class_names = load_oxford_flowers102(imsize=96, fine=False)

See example below for use of the load_oxford_flowers102 method.

"""



"""
Labels for the Oxford flowers102 dataset (fine labelled set)
"""
flowers102_class_names = ['pink primrose','hard-leaved pocket orchid', 'canterbury bells','sweet pea','english marigold',
                          'tiger lily','moon orchid','bird of paradise','monkshood','globe thistle','snapdragon',"colt's foot",
                          'king protea','spear thistle','yellow iris','globe-flower','purple coneflower','peruvian lily',
                          'balloon flower','giant white arum lily','fire lily','pincushion flower','fritillary','red ginger',
                          'grape hyacinth','corn poppy',"prince of wales feathers",'stemless gentian','artichoke','sweet william',
                          'carnation','garden phlox','love in the mist','mexican aster','alpine sea holly','ruby-lipped cattleya',
                          'cape flower','great masterwort','siam tulip','lenten rose','barbeton daisy','daffodil','sword lily',
                          'poinsettia','bolero deep blue','wallflower','marigold','buttercup','oxeye daisy','common dandelion',
                          'petunia','wild pansy','primula','sunflower','pelargonium','bishop of llandaff','gaura','geranium',
                          'orange dahlia','pink-yellow dahlia','cautleya spicata','japanese anemone','black-eyed susan',
                          'silverbush','californian poppy','osteospermum','spring crocus','bearded iris','windflower',
                          'tree poppy','gazania','azalea','water lily','rose','thorn apple','morning glory','passion flower',
                          'lotus','toad lily','anthurium','frangipani','clematis','hibiscus','columbine','desert-rose',
                          'tree mallow','magnolia','cyclamen','watercress','canna lily','hippeastrum','bee balm','ball moss',
                          'foxglove','bougainvillea','camellia','mallow','mexican petunia','bromelia','blanket flower',
                          'trumpet creeper','blackberry lily'
]

"""
Grouping for coarse labelled set
"""
# flowers102_group_names = {
#     "Orchids": [ "hard-leaved pocket orchid","moon orchid","ruby-lipped cattleya"],
#     "Bell-shaped Flowers": [ 'balloon flower', "canterbury bells", 'fritillary', 'japanese anemone','columbine','cyclamen'],
#     "Lilies": ["tiger lily","fire lily","giant white arum lily","toad lily","canna lily",'peruvian lily','siam tulip','hippeastrum'],
#     "Tubular Flowers": ["snapdragon","trumpet creeper","foxglove","bee balm", 'sweet pea', 'red ginger','petunia','wild pansy', 
#                         'pelargonium', 'frangipani', 'clematis', 'hibiscus', 'desert-rose', 'mexican petunia'],
#     "Composite Flowers": ["sunflower","marigold","black-eyed susan","english marigold","purple coneflower",
#                           "barbeton daisy","oxeye daisy", 'globe thistle', 'spear thistle', 'globe-flower', 'mexican aster', 'alpine sea holly',
#                           'bolero deep blue', 'wallflower', 'buttercup', 'common dandelion', 'gaura', 'osteospermum', 'gazania', 'blanket flower'],
#     "Iris-like Flowers": ["yellow iris","bearded iris","blackberry lily",  "sword lily", 'bird of paradise', 'grape hyacinth', 'spring crocus', 'windflower'],
#     "Dahlia Varieties": ["orange dahlia","pink-yellow dahlia", 'bishop of llandaff'],
#     "Poppies": ["corn poppy","californian poppy", 'tree poppy'],
#     "Water Flowers": ["water lily","lotus", 'watercress'],
#     "Carnations": ["sweet william","carnation", 'garden phlox']
# }


flowers102_group_names = {
    "Orchids": [ "hard-leaved pocket orchid","moon orchid","ruby-lipped cattleya"],
    "Bell-shaped Flowers": [ 'balloon flower', "canterbury bells"],
    "Lilies": ["tiger lily","fire lily","giant white arum lily","toad lily","canna lily"],
    "Tubular Flowers": ["snapdragon","trumpet creeper","foxglove","bee balm", 
                        'pelargonium', 'frangipani', 'clematis', 'hibiscus', 'desert-rose', 'mexican petunia'],
    "Composite Flowers": ["sunflower","marigold","black-eyed susan","english marigold","purple coneflower",
                          "barbeton daisy","oxeye daisy"],
    "Iris-like Flowers": ["yellow iris","bearded iris","blackberry lily",  "sword lily"],
    "Dahlia Varieties": ["orange dahlia","pink-yellow dahlia"],
    "Poppies": ["corn poppy","californian poppy"],
    "Water Flowers": ["water lily","lotus"],
    "Carnations": ["sweet william","carnation"]
}


# Define a custom dataset wrapper to remap labels
class RelabeledFlowers102(Dataset):
    def __init__(self, original_dataset, label_map, cache_file=None):

        if cache_file is not None and os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                self.selected_samples = pickle.load(f)
        else:
            self.selected_samples = []
            for i in range(len(original_dataset)):
                _, original_label = original_dataset[i]
                if original_label in label_map:  # Keep only if in new groups
                    self.selected_samples.append((i, label_map[original_label]))
            with open(cache_file, 'wb') as f:
                pickle.dump(self.selected_samples, f)

        self.dataset = original_dataset
        self.label_map = label_map

    def __len__(self):
        return len(self.selected_samples)

    def __getitem__(self, idx):
        original_idx, new_label = self.selected_samples[idx]
        image, _ = self.dataset[original_idx]  # Get image, ignore old label
        return image, new_label


def load_oxford_flowers102(imsize=96, fine=False):
   """
   Loads the flowers102 dataset
   Arguments:
       format: a string ('numpy' (default), 'tfds', or 'pandas')
       imsize: an integer (default is 96) specifying the size of the images to be returned
       fine: a boolean (default is False) specifying whether to return the fine labelled set (True) or the coarse labelled set (False)

    Returns:

      when format=='numpy':
         Tuples (train_data, validation_data, test_data, class_names) where train_data, validation_data, and test_data are dictionaries
         that contain 'images' and 'labels', and the class_names is a list of strings containing the class names.

      when format=='tfds':
         A tuple (oxford_flowers102_train, oxford_flowers102_validation, oxford_flowers102_test) containing the original (fine labelled) train, validation and test dataset in tfds format;

      when format=='pandas':
         A tuple (oxford_flowers102_train, oxford_flowers102_validation, oxford_flowers102_test) containing the original (fine labelled) train, validation and test dataset in pandas data frame format.
   """

   # Get the absolute path to the data folder
   path_to_this_scripts_folder = os.path.dirname(os.path.realpath(__file__))
   path_to_data_folder = os.path.join(path_to_this_scripts_folder, 'data')

   transform_train = transforms.Compose([
                transforms.Resize(imsize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor()
   ])

   transform_test = transforms.Compose([
                transforms.Resize(imsize),
                transforms.CenterCrop(imsize),
                transforms.ToTensor()
   ])


   training_set = torchvision.datasets.Flowers102(path_to_data_folder, 'test', transform=transform_train, download=True)
   validation_set = torchvision.datasets.Flowers102(path_to_data_folder, 'val', transform=transform_test, 
   download=True)
   test_set = torchvision.datasets.Flowers102(path_to_data_folder, 'train', transform=transform_test, download=True)

   if fine:
       class_names = flowers102_class_names
   else:
       # Create a mapping from original labels to new labels
       new_label_mapping = {}
       for new_label, original_labels in enumerate(flowers102_group_names.values()):
           for old_label in original_labels:
               i = flowers102_class_names.index(old_label)
               new_label_mapping[i] = new_label

       training_set = RelabeledFlowers102(training_set, new_label_mapping, os.path.join(path_to_data_folder,'train_coarse.npy'))
       validation_set = RelabeledFlowers102(validation_set, new_label_mapping,  os.path.join(path_to_data_folder,'valid_coarse.npy'))
       test_set = RelabeledFlowers102(test_set, new_label_mapping,  os.path.join(path_to_data_folder,'test_coarse.npy'))
       class_names = list(flowers102_group_names.keys())

   return training_set, validation_set, test_set, class_names

def write_to_folder(images, folder_name):
    """
    Helper function that writes images to folder
    """

    from PIL import Image  

    path_to_this_scripts_folder = os.path.dirname(os.path.realpath(__file__))
    path_to_save_folder = os.path.join(path_to_this_scripts_folder, folder_name)

    if not os.path.isdir(path_to_save_folder):
        os.makedirs(path_to_save_folder, exist_ok=True)

    print("Writing images to %s..." % folder_name)
    for i in tqdm.tqdm(range(len(images))):
        im_file = f'img{i:05d}.jpg'
        im_path = os.path.join(path_to_save_folder, im_file)
        im = images[i].numpy().transpose(1,2,0)
        im = (im*255).astype(np.uint8)
        im = Image.fromarray(im)
        im.save(im_path)      

if __name__ == "__main__":
    import sys
    import torch

    path_to_this_scripts_folder = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(path_to_this_scripts_folder)
    import show_methods

    sort_by_class = True

    # Load the dataset (change the 'fine' argument to True to load the fine labelled set)
    training_set, validation_set, test_set, class_names = load_oxford_flowers102(imsize=96, fine=False)

    # Create data loader for training
    training_data = torch.utils.data.DataLoader(training_set, batch_size=16, shuffle=True)

    # You can write the images to folder to inspect them
    for x_train, y_train in training_data:
        write_to_folder(x_train, 'oxford_flowers102_train')
        break

    for x_train, y_train in training_data:
        show_methods.show_data_images(images=x_train,labels=y_train, class_names=class_names)
        

