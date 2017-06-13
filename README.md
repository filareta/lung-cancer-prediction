# README #



## Analysis and experiments with early cancer detection in lung CT scans ##



### What is this repository for? ###

* Experimenting with approaches for processing the lung CT scans that are publicly available in the kaggle competition  Data Science Bowl 2017.
* Experimenting with deep neural networks for training a model that hepls earily cancer detection.


### How do I get set up? ###
 
1. The project is written in python. The easiest way to set up the environment is using Anaconda. Here is the link for downloading https://www.continuum.io/downloads. Python 3.x is required, the preferred version of Anaconda is the one using python 3.6.

2. Required python modules

   * **pydicom**
   * **scikit-image**
   * **scikit-learn**
   * **scipy**
   * **sklearn**
   * **pandas**
   * **imutils**
   * ** tensorflow**
   * ** google-api-python-client**
   * ** google-cloud-storage**
   * **opencv** module for python

   All of the modules instead of tensorflow and cv2 could be easily installed using either 
  
```
#!shell

      pip install <module_name>
```

  
   or 

 
```
#!shell

     conda install <module_name>
```
   
   Though some of the modules are recommended to be installed using conda, since Anaconda is hadling some of the environment issues and provides precompiled binaries for the modules.

   For example:

```
#!shell

     conda install scikit-image
```

   Installing opencv:
     
     
```
#!shell

     conda install -c menpo opencv3

```
  
   Before installing any modules, start with setting up the Anaconda environment suitable for
   installing the tensorflow library.
   
   Creating CPU tensorflow environment:

    
```
#!shell

     conda create --name tensorflow python=3.5
     activate tensorflow
     conda install jupyter
     conda install scipy
     pip install tensorflow
```

     

   If you are running on windows you should specify the python version to be 3.5 when creating the environment.


   Creating GPU tensorflow environment:
  
     
```
#!shell

     conda create --name tensorflow-gpu python=3.5
     activate tensorflow-gpu
     conda install jupyter
     conda install scipy
     pip install tensorflow-gpu
```
 

  To switch between environments you can simply use ***deactivate*** to deactivate the current one and ***activate tensorflow-gpu*** for
  example, if you want to switch to the tensorflow environment with gpu support.



  In order to run tensorflow with GPU support you must install Guda Toolkit and cuDNN.

  *   Installing on Windows - https://www.tensorflow.org/install/install_windows
  *   Installing on Ubuntu - https://www.tensorflow.org/install/install_linux
  *   Installing on Mac OS X - https://www.tensorflow.org/install/install_mac


  After installing tensorflow, you can simply use the requirements.txt file provided in the project. Execute the following line:


```
#!shell

      pip install -r requirements.txt
```

  In case you are unable to install some of the modules listed in the requirements
  file, do remove it and try installing it using:


```
#!shell

     conda install <module-name>
```


###  How to start data preprocessing? ###

  
To start processing the dicom files you need to run:

   
```
#!shell

     python preprocess_dicoms.py
```

  Although this step is not required, since original images are too big and data preprocessing is time consuming. First stage of image preprocessing has been already executed and the data is stored in Google Cloud using several buckets:

    
    *  Baseline - https://console.cloud.google.com/storage/browser/baseline-preprocess/baseline_preprocessing/?project=lung-cancer-tests

    *  Morphological operations segmentation - https://console.cloud.google.com/storage/browser/segmented-lungs/segmented_morph_op/?project=lung-cancer-tests

    * Waterhsed segmentation - https://console.cloud.google.com/storage/browser/segmented-lungs-watershed/segmented_watershed/?project=lung-cancer-tests
 

  To simply download the data required for the model to be trained you need to execute:


```
#!shell

     python data_collector.py
```
  
  Compressed 3D patient images will be downloaded and by default stored under ***./fetch_data/ *** directory.

  To select which model will be trained, you need to change the value of SELECTED_MODEL in ***config.py*** (simply choose one of the predefined model names and the other configurations will be changed correspondingly)

  Source and destination directories are configurable using the ***config.py***:

   * ALL_IMGS points to the directory with the original dicom files for each patient (you do not need to edit it, if you have used the download script to store the preprocessed data as mentioned in the previous step)

   * SEGMENTED_LUNGS_DIR points to the directory where the segmented lungs will be stored in a 
    .npz file for each patient (compressed numpy array). The directory is configured automaticaly depending on the selected model to be trained



### How to start model training? ###


 To start model training simply execute
 

```
#!shell

     python model_train.py
```

 Configuration and definitions of the layers for the CNN are described in python files located under ***model_definition***. Three configurations are currently available:

  * baseline.py  - Baseline configuration with three convolutional layers and two fully connected.

  * additional_layers.py - Deeper network with four convolutional and
  three fully connected layers.

  * default.py - The default configuration also has seven layers in total, but some of the filters have different sizes from those in the previous configuration.

### Other configurations? ###


Other properties that might be configured are related with storing model states and summary exported during training.
  
  * SUMMARIES_DIR - points to the directory where summary for the error, accuracy and senitivity is exported during trainig. The data can be viewed using Tensorboard.
  * RESTORE_MODEL_CKPT - point to the checkpoint file, you might want to resume training from or simply use for evaluating test examples with the saved state of the network (if you want to resume triaing set RESTORE to True and point out the START_STEP for proper counting of the epoches)
