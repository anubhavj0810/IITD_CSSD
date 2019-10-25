
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

from models.keras_ssd300mask import ssd_300_mask
from models.keras_ssd300_base import ssd_300_base
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

#from VOCdevkit.prior_generator import get_prior_batch, get_priors, initialize_priors,calc_accuracy,get_Accuracy_Metrics, get_prior_mask_batch
import os
import cv2
os.environ["CUDA_VISIBLE_DEVICES"]="3,4,6"
import time
times = []
#get_ipython().magic(u'matplotlib inline')

from Calculate_Accuracy import get_accuracy, get_accuracy_metrics
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

model = ssd_300_mask(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)


#print(model.summary())
# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = 'ssd300_pascal_07+12_epoch-80_loss-2.5239_val_loss-2.5941.h5'
#weights_path = 'saved_weights_prior_with_person_pets/ssd300_pascal_07+12_epoch-20_loss-2.8624_val_loss-3.8971.h5'
#weights_path = 'saved_weights/ssd300_pascal_07+12_102k_steps.h5'
model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

#priors = get_prior_batch(['VOCdevkit/PETS/JPEGImages/frame5_0600.jpg'])
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


#orig_images = [] # Store the images here.
#input_images = [] # Store resized versions of the images here.
views = ["1","5","6","7","8"]
#views = ["1"]
confidence_threshold = 0.5
classes = ['background','person']


classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

#for view in views:
#    print("View: ",view)
#    for id in range(0,1500):
#        orig_images = [] # Store the images here.
#        input_images = [] # Store resized versions of the images here.
#        img_path = "VOCdevkit/PETS/JPEGImages/frame" + view +"_" + '{0:04d}'.format(id)+".jpg"
#        if (not os.path.exists(img_path)):
#            continue 
#        orig_images.append(imread(img_path))
#        img = image.load_img(img_path, target_size=(img_height, img_width))
#        img = image.img_to_array(img) 
#        input_images.append(img)
#        input_images = np.array(input_images)

#        priors = get_prior_mask_batch([img_path])
#        #print("Priors: ",priors.shape)
#        #inputA = priors[0]; inputB = priors[1]; inputC = priors[2]; inputD = priors[3]; inputE = priors[4]


#        y_pred = model.predict([input_images,priors])
#	#y_pred = model.predict(input_images)
#        y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

#        np.set_printoptions(precision=2, suppress=True, linewidth=90)
#        print("ID: ",id)
#        #print('   class   conf xmin   ymin   xmax   ymax')
#        #print(y_pred_thresh[0])
#        det_content = []
#        for box in y_pred_thresh[0]:
#            # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
#            #print(orig_images[0].shape)
#            xmin = box[2] * orig_images[0].shape[1] / img_width
#            ymin = box[3] * orig_images[0].shape[0] / img_height
#            xmax = box[4] * orig_images[0].shape[1] / img_width
#            ymax = box[5] * orig_images[0].shape[0] / img_height
#            det_content.append([xmin,ymin,xmax,ymax])
#            #color = colors[int(box[0])]
#            cv2.rectangle(orig_images[0],(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),3)
#            #label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#            #current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
#            #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})

#            cv2.imwrite("Output_prior/frame"+view+"_"+'{0:04d}'.format(id)+".jpg",orig_images[0])
#        calc_accuracy(int(view),id,det_content)
#print("****************************************FINAL ACCURACY*********************************************")
#get_Accuracy_Metrics()


# ## 5. Make predictions on Pascal VOC 2007 Test
# 
# Let's use a `DataGenerator` to make predictions on the Pascal VOC 2007 test dataset and visualize the predicted boxes alongside the ground truth boxes for comparison. Everything here is preset already, but if you'd like to learn more about the data generator and its capabilities, take a look at the detailed tutorial in [this](https://github.com/pierluigiferrari/data_generator_object_detection_2d) repository.

# In[10]:


# Create a `BatchGenerator` instance and parse the Pascal VOC labels.

dataset = DataGenerator()

# TODO: Set the paths to the datasets here.

VOC_2007_images_dir         = 'VOCdevkit/PETS/JPEGImages/'
VOC_2007_annotations_dir    = 'VOCdevkit/PETS/Annotations/'
VOC_2007_test_image_set_filename = 'VOCdevkit/PETS/ImageSets/Main/view7_bc.txt'

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
                                      'original_labels',
                                      'masks'},
                             keep_images_without_gt=False)


# In[13]:


# Generate a batch and make predictions.
confidence_threshold = 0.5
for epoch in range(1000):
    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels,masks = next(generator)

    i = 0 # Which batch item to look at

    #print("Image:", batch_filenames[i])
    #print()
    #print("Ground truth boxes:\n")
    #print(np.array(batch_original_labels[i]))
    #y_pred = model.predict(batch_images)
    start = time.time()
    y_pred = model.predict([batch_images,masks])
    elapsed  = (time.time()-start)*1000
    times.append(elapsed)
    print("Time: ",time.time()-start)

    # Perform confidence thresholding.
    y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]

    # Convert the predictions for the original image.
    y_pred_thresh_inv = apply_inverse_transforms(y_pred_thresh, batch_inverse_transforms)

    #np.set_printoptions(precision=2, suppress=True, linewidth=90)
    #print("Predicted boxes:\n")
    #print('   class   conf xmin   ymin   xmax   ymax')
    #print(y_pred_thresh_inv[i])


    # Display the image and draw the predicted boxes onto it.

    #current_axis = plt.gca()
    gts = []
    dets = []
    for box in batch_original_labels[i]:
        xmin = box[1]
        ymin = box[2]
        xmax = box[3]
        ymax = box[4]
        gts.append([xmin,ymin,xmax,ymax])
        label = '{}'.format(classes[int(box[0])])
        #current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
        #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})
        cv2.rectangle(batch_original_images[i],(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,255,255),3)
    for box in y_pred_thresh_inv[i]:
        xmin = box[2]
        ymin = box[3]
        xmax = box[4]
        ymax = box[5]
        dets.append([xmin,ymin,xmax,ymax])
        #color = colors[int(box[0])]
        label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
        #current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
        #current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})
        cv2.rectangle(batch_original_images[i],(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0),3)

    cv2.imwrite("Output_prior/epoch_"+str(epoch)+".jpg",batch_original_images[i])
    #np.array(y_pred_thresh_inv).dump("dump_files/epoch_"+str(epoch))
    get_accuracy(gts,dets)

print("Mean: ",np.mean(np.array(times)))
print("STD: ",np.std(np.array(times)))
get_accuracy_metrics();
