
# coding: utf-8

# # SSD300 Inference Tutorial
# 
# This is a brief tutorial that shows how to use a trained SSD300 for inference on the Pascal VOC datasets. If you'd like more detailed explanations, please refer to [`ssd300_training.ipynb`](https://github.com/pierluigiferrari/ssd_keras/blob/master/ssd300_training.ipynb)

# In[ ]:


from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300_base import ssd_300_base
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator1 import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

#from VOCdevkit.prior_generator import get_prior_batch, get_priors, initialize_priors,calc_accuracy,get_Accuracy_Metrics
from Calculate_Accuracy import get_accuracy, get_accuracy_metrics
import os
import cv2
#os.environ["CUDA_VISIBLE_DEVICES"]="1,6"
#get_ipython().magic(u'matplotlib inline')


# In[2]:


# Set the image size.
img_height = 300
img_width = 300

PETS_images_dir      = 'VOCdevkit/PETS/JPEGImages/'
#initialize_priors()
# ## 1. Load a trained SSD
# 
# Either load a trained model or build a model and load trained weights into it. Since the HDF5 files I'm providing contain only the weights for the various SSD versions, not the complete models, you'll have to go with the latter option when using this implementation for the first time. You can then of course save the model and next time load the full model directly, without having to build it.
# 
# You can find the download links to all the trained model weights in the README.

# ### 1.1. Build the model and load trained weights into it

# In[3]:


# 1: Build the Keras model

K.clear_session() # Clear previous models from memory.

# model = ssd_300_base(image_size=(img_height, img_width, 3),
#                 n_classes=1,
#                 mode='inference',
#                 l2_regularization=0.0005,
#                 scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
#                 aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
#                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                                          [1.0, 2.0, 0.5],
#                                          [1.0, 2.0, 0.5]],
#                 two_boxes_for_ar1=True,
#                 steps=[8, 16, 32, 64, 100, 300],
#                 offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#                 clip_boxes=False,
#                 variances=[0.1, 0.1, 0.2, 0.2],
#                 normalize_coords=True,
#                 subtract_mean=[123, 117, 104],
#                 swap_channels=[2, 1, 0],
#                 confidence_thresh=0.5,
#                 iou_threshold=0.45,
#                 top_k=200,
#                 nms_max_output_size=400)

img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 20 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5],
                 [1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

model  = ssd_300_base(image_size=(img_height, img_width, img_channels),
                n_classes=n_classes,
                mode='inference',
                l2_regularization=0.0005,
                scales=scales,
                aspect_ratios_per_layer=aspect_ratios,
                two_boxes_for_ar1=two_boxes_for_ar1,
                steps=steps,
                offsets=offsets,
                clip_boxes=clip_boxes,
                variances=variances,
                normalize_coords=normalize_coords,
                subtract_mean=mean_color,
                swap_channels=swap_channels,
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)


# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = 'ssd300_pascal_07+12_102k_steps.h5'
#weights_path = 'ssd300_pascal_07+12_epoch-80_loss-2.5239_val_loss-2.5941.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

#priors = get_prior_batch(['VOCdevkit/PETS/JPEGImages/frame1_0319.jpg'])
#print("Priors: ",priors)
#inputA = priors[0]; inputB = priors[1]; inputC = priors[2]; inputD = priors[3]; inputE = priors[4]

# Or

# ### 1.2. Load a trained model

# In[ ]:


# TODO: Set the path to the `.h5` file of the model to be loaded.
#model_path = 'path/to/trained/model.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
#ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

#K.clear_session() # Clear previous models from memory.

#model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                               'L2Normalization': L2Normalization,
#                                               'DecodeDetections': DecodeDetections,
#                                               'compute_loss': ssd_loss.compute_loss})


# ## 2. Load some images
# 
# Load some images for which you'd like the model to make predictions.

# In[4]:


views = ["1","5","6","7","8"]
confidence_threshold = 0.5
classes = ['background','person']

# for view in views:
#     print("View: ",view)
#     for id in range(0,10):
#         orig_images = [] # Store the images here.
#         input_images = [] # Store resized versions of the images here.
#         img_path = "VOCdevkit/PETS/JPEGImages/frame" + view +"_" + '{0:04d}'.format(id)+".jpg"
#         if (not os.path.exists(img_path)):
#             continue 
#         orig_images.append(imread(img_path))
#         img = image.load_img(img_path, target_size=(img_height, img_width))
#         img = image.img_to_array(img) 
#         input_images.append(img)
#         input_images = np.array(input_images)

#         #priors = get_prior_batch([img_path])
#         #print("Priors: ",priors)
#         #inputA = priors[0]; inputB = priors[1]; inputC = priors[2]; inputD = priors[3]; inputE = priors[4]


#         y_pred = model.predict(input_images)

#         y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

#         np.set_printoptions(precision=2, suppress=True, linewidth=90)
#         print("ID: ",id)
#         #print('   class   conf xmin   ymin   xmax   ymax')
#         #print(y_pred_thresh[0])
        
#         # gts = []
#         # for box in batch_original_labels[0]:
#         #     xmin = box[1]
#         #     ymin = box[2]
#         #     xmax = box[3]
#         #     ymax = box[4]
#         #     gts.append([xmin,ymin,xmax,ymax])

#         det_content = []
#         for box in y_pred_thresh[0]:
#             # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
#             #print(orig_images[0].shape)
#             xmin = box[2] * orig_images[0].shape[1] / img_width
#             ymin = box[3] * orig_images[0].shape[0] / img_height
#             xmax = box[4] * orig_images[0].shape[1] / img_width
#             ymax = box[5] * orig_images[0].shape[0] / img_height
#             det_content.append([xmin,ymin,xmax,ymax])
#             #color = colors[int(box[0])]
#             cv2.rectangle(orig_images[0],(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),3)
#             #label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#             #current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
#             #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

#             cv2.imwrite("Output_prior/frame"+view+"_"+'{0:04d}'.format(id)+".jpg",orig_images[0])
#         #get_accuracy(gts,det_content)
# #print("****************************************FINAL ACCURACY*********************************************")
# #get_accuracy_metrics()

dataset = DataGenerator()

# TODO: Set the paths to the datasets here.

# VOC_2007_images_dir         = 'VOCdevkit/VOC2007/JPEGImages/'
# VOC_2007_annotations_dir    = 'VOCdevkit/VOC2007/Annotations/'
# VOC_2007_test_image_set_filename = 'VOCdevkit/VOC2007/ImageSets/Main/test.txt'

VOC_2007_images_dir         = 'VOCdevkit/PETS/JPEGImages/'
VOC_2007_annotations_dir    = 'VOCdevkit/PETS/Annotations/'
VOC_2007_test_image_set_filename = 'VOCdevkit/PETS/ImageSets/Main/view7_bc1.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

dataset.parse_xml(images_dirs=[VOC_2007_images_dir],
                  image_set_filenames=[VOC_2007_test_image_set_filename],
                  annotations_dirs=[VOC_2007_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=True,
                  ret=False)

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

generator = dataset.generate(batch_size=1,
                             shuffle=True,
                             transformations=[convert_to_3_channels,
                                              resize],
                             returns={'processed_images',
                                      'filenames',
                                      'inverse_transform',
                                      'original_images',
                                      'original_labels'},
                             keep_images_without_gt=False)


# In[13]:


# Generate a batch and make predictions.

confidence_threshold = 0.5 
plt.figure(figsize = (8,6))
count_arr = []
acc_arr = []
for epoch in range(1000):
  batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(generator) 

  i = 0 # Which batch item to look at 
  
#  print()
#  print(epoch)
#  print("Image:", batch_filenames[i])
#  print("Ground truth boxes:\n")
#  print(np.array(batch_original_labels[i])) 

  # Predict.  
  try:
    y_pred = model.predict(batch_images)
    # Perform confidence thresholding.
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
    # Convert the predictions for the original image.
    y_pred_thresh_inv = apply_inverse_transforms(y_pred_thresh, batch_inverse_transforms)
  except Exception as e:
    raise e
  #np.set_printoptions(precision=2, suppress=True, linewidth=90)
#  print("Predicted boxes:\n")
#  print('class   conf xmin   ymin   xmax   ymax')
#  print(y_pred_thresh_inv[i]) 
  

  # Display the image and draw the predicted boxes onto it. 

  # Set the colors for the bounding boxes
  # colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist() 

  # plt.figure(figsize=(20,12))
  # plt.imshow(batch_original_images[i])  

  # current_axis = plt.gca()  

  gts = []
  dets = [] 

  for box in batch_original_labels[i]:
      xmin = box[1]
      ymin = box[2]
      xmax = box[3]
      ymax = box[4]
      gts.append([xmin,ymin,xmax,ymax])
      # label = '{}'.format(classes[int(box[0])])
      # current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
      # current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})  

  for box in y_pred_thresh_inv[i]:
      xmin = box[2]
      ymin = box[3]
      xmax = box[4]
      ymax = box[5]
      dets.append([xmin,ymin,xmax,ymax])
      #color = colors[int(box[0])]
      # label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
      # current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
      # current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})  

  acc = get_accuracy(gts,dets)
  if(not acc):
    acc = acc_arr[epoch-1]
  count_1 = epoch + 1
  count_arr.append(count_1)
  print("Count", count_1)
  acc_arr.append(acc)
    
  ax = plt.gca()
  ax.set_xlabel('No. of images', fontsize=15)
  ax.set_ylabel('Accuracy', fontsize=15)
  ax.set_title('Accuracy Graph Full Dataset Sample Baseline Model', fontsize=15, fontweight='bold')
  ax.plot(np.array(count_arr), np.array(acc_arr), '-yo',markevery=100)   # scatter plot showing actual data

  plt.pause(0.0000001)

