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
   * ** tensorflow**
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
   

   Installing opencv:
     
     
```
#!shell

     conda install -c menpo opencv3

```

   
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


###  How to start data preprocessing? ###
  
To start processing the dicom files you need to run

   
```
#!shell

     python preprocess_dicoms.py
```



 Source and destination directories are configurable using the ***config.py***:

   * ALL_IMGS points to the directory with dicom files for each patient
   * SEGMENTED_LUNGS_DIR points to the directory where the segmented lungs will be stored in a 
    .npz file for each patient (compressed numpy array)



### How to start model training? ###


 To start model training simply execute
 

```
#!shell

     python model_train.py
```

 Configuration and definitions of the layers for the CNN are stored in ***model_definition.py***