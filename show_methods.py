__author__ = "Lech Szymanski"
__organization__ = "COSC420, University of Otago"
__email__ = "lech.szymanski@otago.ac.nz"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
   import torch
   torch_lib = True
except:
   torch_lib = False
   
import time

def show_data_attributes(inputs, labels,attribute_names,class_names):
   # Create a figure
   figure_handle = plt.figure()

   n_attributes = inputs.shape[1]
   n_plots = int(n_attributes*(n_attributes-1)/2)
   n_plot_rows = int(np.floor(np.sqrt(n_plots)))
   n_plot_columns = int(np.ceil(n_plots/n_plot_rows))

   # Plot each dimension against another dimension
   n_classes = len(class_names)
   plot_index = 1
   for i in range(n_attributes):
       for j in range(i+1,n_attributes):
           # Add subplot to the figure
           plot_handle = figure_handle.add_subplot(n_plot_rows, n_plot_columns, plot_index)
           # Do a scatter plot for each class separately (matplotlib automatically does them in different colours)
           for k in range(n_classes):
               # Find all the indexes of points of class k
               I = np.where(labels==k)[0]
               # Show a scatter plot of dimension i vs. dimension j of the points of class k
               plot_handle.scatter(inputs[I,i],inputs[I,j])
           # Increments the plot_index variable
           plot_index += 1
           # Label the x and y axis
           plot_handle.set_xlabel(attribute_names[i])
           plot_handle.set_ylabel(attribute_names[j])

   # Show the legend for class colours on the last subplot
   plot_handle.legend(class_names)


def show_data_images(images, labels=None, predictions=None, class_names=None,blocking=True):

   if torch_lib and torch.is_tensor(images):
      images = images.numpy()
      # Torch images are in the shape (N, C, H, W) where N is the number of images, C is the number of channels, H is the height, whereas the show methods expect (N, H, W, C) shape...so we transpose switching the last two dimensions
      images = np.transpose(images, (0, 2, 3, 1))
      if len(images.shape) == 4 and images.shape[3] == 1:
         images = np.squeeze(images, axis=3)

      if labels is not None and torch.is_tensor(labels):
         labels = labels.numpy()

   # Create a figure
   figure_handle = plt.figure(figsize=(6,8))
   n_images = len(images)
   n_plot_rows = int(np.floor(np.sqrt(n_images)))
   n_plot_columns = int(np.ceil(n_images/n_plot_rows))

   plt.ion()

   # Plot each dimension against another dimension
   for i in range(n_plot_rows * n_plot_columns):
      plot_handle = figure_handle.add_subplot(n_plot_rows, n_plot_columns, i + 1)
      if len(np.shape(images)) == 3:
         plot_handle.imshow(images[i, :, :], cmap='gray')
      else:
         plot_handle.imshow(images[i, :, :,:])

      plot_handle.tick_params(bottom=False, left=False)
      plot_handle.tick_params(labelbottom=False, labelleft=False)

      titleStr = ''
      if labels is not None:
         if class_names is not None:
            if len(labels.shape) == 1:
               titleStr += 'label: %s' % class_names[labels[i]]
            else:
               titleStr = ''
               for j in labels[i]:
                  if j<len(class_names):
                     titleStr += '%s, ' % class_names[j]
               titleStr = titleStr[:-2]
         elif isinstance(labels[i],list) and len(labels[i])==3:
            titleStr = ''
            firstLabel = True
            for label in labels[i]:
               if firstLabel:
                  firstLabel = False
               else:
                  titleStr += '\n'
               titleStr += '%s (%.2f)' % (label[1],label[2])






      if predictions is not None:
         if labels is not None:
            titleStr += '\n'

         titleStr += 'pred: %s' % class_names[predictions[i]]

      plot_handle.set_title(titleStr)
      plt.pause(0.01)
      plt.show()
      time.sleep(0.01)

   if blocking:
      plt.ioff()
   else:
      time.sleep(1)

   plt.show()

def show_data_classes(inputs, labels, class_names,mode='pca'):

   if len(np.shape(inputs))>2:
      n_inputs = len(inputs)
      inputs = np.reshape(inputs,(n_inputs,-1))

   # Set up PCA compression to 2D
   if mode=='pca':
      x_2D = PCA(n_components=2).fit_transform(inputs)
   elif mode=='tsne':
      x_2D = TSNE(n_components=2).fit_transform(inputs)

   # Create a new figure
   figure_handle = plt.figure()
   # Create a subplot
   plot_handle = figure_handle.add_subplot(1, 1, 1)
   # Do a scatter plot for each class separately (matplotlib automatically does them in different colours)
   for k in range(len(class_names)):
      # Find all the indexes of points of class k
      I = np.where(labels == k)[0]
      # Show a scatter plot of dimension i vs. dimension j of the points of class k
      plot_handle.scatter(x_2D[I, 0], x_2D[I, 1])

   # Show the legend for class colours
   if len(class_names) <= 10:
      plot_handle.legend(class_names)
   # Display the plot title
   plot_handle.set_title('Compressed data from %dD to 2D' % inputs.shape[1])
