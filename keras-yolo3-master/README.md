
## Training

### 1. Data preparation 

Use the tool to parse the data set and transfer it into voc format.
https://gitlab.aiacademy.tw/JT072/Project_JT072_09_Biodiv-Beetle/blob/master/utility/extractor.ipynb

The dataset will be organized into 4 folders:

+ train_image_folder <= the folder that contains the train images.

+ train_annot_folder <= the folder that contains the train annotations in VOC format.

+ valid_image_folder <= the folder that contains the validation images.

+ valid_annot_folder <= the folder that contains the validation annotations in VOC format.
    
There is a one-to-one correspondence by file name between images and annotations. If the validation set is empty, the training set will be automatically splitted into the training set and validation set using the ratio of 0.8.

### 2. Edit the configuration file
The configuration file is a json file, which looks like this:

```python
{
    "model" : {
        "min_input_size":       288,
        "max_input_size":       448,
        "anchors":              [17,18, 28,24, 36,34, 42,44, 56,51, 72,66, 90,95, 92,154, 139,281],
        "labels":               ["beetle"]
    },

    "train": {
        "train_image_folder":   "/project/jt072-09-biodiv-beetle/ds_train/images/",
        "train_annot_folder":   "/project/jt072-09-biodiv-beetle/ds_train/annotations/",
        "cache_name":           "beetle_train_0323_4.pkl",

        "train_times":          3,
        "batch_size":           16,
        "learning_rate":        1e-4,
        "nb_epochs":            100,
        "warmup_epochs":        3,
        "ignore_thresh":        0.5,
        "gpus":                 "1",

        "grid_scales":          [1,1,1],
        "obj_scale":            5,
        "noobj_scale":          1,
        "xywh_scale":           1,
        "class_scale":          1,

        "tensorboard_dir":      "log_beetle_0323_4",
        "saved_weights_name":   "beetle_0323_4.h5",
        "debug":                true
    },

    "valid": {
        "valid_image_folder":   "",
        "valid_annot_folder":   "",
        "cache_name":           "",

        "valid_times":          1
    }
}

```

Download pretrained weights for backend at:
/jt072-09-biodiv-beetle/yolo3_profile/backend.h5

**This weights must be put in the root folder of the repository. They are the pretrained weights for the backend only and will be loaded during model creation. The code does not work without this weights.**

### 3. Generate anchors for your dataset (optional)

`python gen_anchors.py -c config_beetle.json`

Copy the generated anchors printed on the terminal to the ```anchors``` setting in ```config_beetle.json```.

### 4. Start the training process

`python train.py -c config_beetle.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

### 5. Perform detection using trained weights on image, set of images, video, or webcam
`python predict.py -c config_beetle.json -i /path/to/image/or/video`

It carries out detection on the image and write the image with detected bounding boxes to the same folder.

