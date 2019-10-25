This code section is for the Collaborative Single Shot Detector (CSSD). Single Shot Detector (SSD) is a object detection algorithm, we modify that for the collaborative architecture to present the CSSD architecture. Traditional object detection networks detect object by passing only the input image. In CSSD, object detection is benefited from not only the input frames but also another input mask which encodes possible locations of objects obtained from peer camera detections.

The code is tested on Tensorflow 1.6.0 and Keras 2.2.4. 
Training the Network######################

    - Benchmark dataset for training and testing object detection algorithms are VOC2007, VOC2012 and COCO. We have trained and tested the baseline SSD model using VOC2007 and VOC2012. You can extract VOC dataset tar files into VOCdevkit folder for training and testing purposes.
    - For testing the collaborative model we use a benchmark dataset called PETS2009 (http://www.cvg.reading.ac.uk/PETS2009/a.html). Ground truth annotations and images are available in the folder VOCdevkit/PETS
    - For training CSSD model you can either use a part of PETS dataset and finetune the network with homography mapped input masks (As implemented in data_generator/object_detection_2d_data_generator.py in function get_Priors_from_Homograph ), or you can use VOC2007 and VOC2012 dataset by passing random loose bounding boxes (As implemented in data_generator/object_detection_2d_data_generator.py in function get_prior_mask_batch)  
    - ssd300_training.py code is for training baseline SSD architecture
    - ssd300baseline_inference.py code is for testing baseline SSD architecture 
    - ssd300_training_mask_new.py code is for training CSSD architecture
    - ssd300mask_inference_py3.py code is for testing baseline CSSD architecture 

Training Models###########################

    - Training models are defined inside models/ folder. (keras_ssd300_base.py - Baseline SSD model, keras_ssd300mask.py - CSSD Model ). You can customize and build your own model by following the similar pattern
