# Traffic Light Classification

---

**Traffic Light Classification**

The goals/steps of this project are the following:

* Gather and label the datasets
* Transfer learning on a TensorFlow model
* Classify the state of traffic lights
* Summarize the results with a written report

[//]: # (References)
[capstone project]: https://github.com/iburris/CarND-Capstone
[bosch dataset]: https://hci.iwr.uni-heidelberg.de/node/6132
[lara dataset]: http://www.lara.prd.fr/benchmarks/trafficlightsrecognition
[alex lechner dataset]: https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0
[coldknight dataset]: https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset
[coldknight repo]: https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI
[daniel stang]: https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e
[anthony sarkis]: https://codeburst.io/self-driving-cars-implementing-real-time-traffic-light-detection-and-classification-in-2017-7d9ae8df1c58
[vatsal srivastava]: https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62
[faster rcnn]: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
[labeling img]: ./imgs/labeling.jpg
[labelImg]: https://github.com/tzutalin/labelImg
[simultaneous training]: ./imgs/simultaneous-training.jpg
[tf bad perfomance]: ./imgs/tf-bad-performance
[tfrecord file]: #23-create-a-tfrecord-file
[clifton pereira]: https://github.com/ExtEng
[ian burris]: https://github.com/iburris
[label map]: ./data/udacity_label_map.pbtxt
[set up tensorflow]: #set-up-tensorflow
[create_tf_record]: create_tf_record.py
[training section]: #training
[protobuf win]: https://github.com/google/protobuf/releases
[cdahms question]: https://stackoverflow.com/questions/48247921/tensorflow-object-detection-api-on-windows-error-modulenotfounderror-no-modu
[pythonpath win]: ./imgs/pythonpath-win.jpg
[path variable win]: ./imgs/path-win.jpg
[models zoo]: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
[simultaneous training]: ./imgs/simultaneous-training.jpg
[ssd inception]: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_11_06_2017.tar.gz 
[ssd inception 171117]: http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz
[faster rcnn inception]: http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
[faster rcnn resnet101]: http://download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
[bad performance]: ./imgs/tf-bad-performance.jpg
[model configs]: https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs
[alex lechner model configs]: ./config
[7-zip win]: https://www.7-zip.org/
[aws login]: https://console.aws.amazon.com
[spot instance]: ./imgs/aws-spot-instance.jpg
[tf setup linux]: #linux
[epratheeban github]: https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/issues/11
[aws inbound rules]: ./imgs/aws-inbound-rules.jpg
[kill memory]: ./imgs/kill-process.jpg
[ssd udacity]: ./imgs/ssd-udacity-image.jpg
[ssd simulator]: ./imgs/ssd-simulator-image.jpg
[jupyter notebook]: ./tl_classification.ipynb

---

## Table of Contents
1. [Introduction](#introduction)
2. [Set up Tensorflow][set up tensorflow]
    1. [Windows 10](#windows-10)
    2. [Linux][tf setup linux]
3. [Datasets](#datasets)
    1. [The Lazy Approach](#1-the-lazy-approach)
    2. [The Diligent Approach](#2-the-diligent-approach)
        1. [Extract images from a ROSbag file](#21-extract-images-from-a-rosbag-file)
        2. [Data labeling](#22-data-labeling)
        3. [Create a TFRecord file][tfrecord file]
4. [Training][training section]
    1. [Choosing a model](#1-choosing-a-model)
    2. [Configure the .config file of the model](#2-configure-the-config-file-of-the-model)
    3. [Setup an AWS spot instance](#3-setup-an-aws-spot-instance)
    4. [Training the model](#4-training-the-model)
    5. [Freezing the graph](#5-freezing-the-graph)
5. [Recommendation: Use SSD Inception V2](#recommendation-use-ssd-inception-v2)
    1. [Conclusion](#conclusion)
6. [Troubleshooting](#troubleshooting)
7. [Summary](#summary)


## Introduction

The goal of this project was to retrain a TensorFlow model on images of traffic lights in their different light states. The trained model was then used in the final capstone project of the Udacity Self-Driving Car Engineer Nanodegree Program as a frozen inference graph. Our project (and the implementation of the frozen graph) can be found here: [Drive Safely Capstone Project][capstone project]

The following guide is a detailed tutorial on how to set up the traffic light classification project, to (re)train the TensorFlow model and to avoid the mistakes I did. For my project I've read [Daniel Stang's][daniel stang], [Anthony Sarkis'][anthony sarkis] and [Vatsal Srivastava's][vatsal srivastava] Medium posts on traffic light classification. I encourage you to read through them as well. However, even though they were comprehensible and gave a basic understanding of the problem the authors still missed the biggest and hardest part of the project: Setting up a training environment and retrain the Tensorflow model.

I will now try to cover up all steps necessary from the beginning to the end to have a working classifier. Also, this tutorial is Windows-friendly since the project was done on Windows 10 for the most part. I suggest reading through this tutorial first before following along.

**If you run into any errors during this tutorial (and you probably will) please check the [Troubleshooting section](#troubleshooting).**

## Set up TensorFlow
If a technical recruiter ever asks me:

>_"Describe the toughest technical problem you've worked on."_

my answer definitely will be:

> _"Get TensorFlow to work!"_

Seriously, if someone from the TensorFlow team is reading this: Clean up your folder structure, use descriptive folder names, merge your READMEs and - more importantly - **fix your library!!!**

But enough of Google bashing - they're doing a good job but the library still has teething troubles (and an user-**un**friendly installation setup).

I will now show you how to install the TensorFlow 'models' repository on Windows 10 and Linux. The Linux setup is easier and if you don't have a powerful GPU on your local machine I strongly recommend you to do the training on an AWS spot instance because this will save you a lot of time. However, you can do the basic stuff like data preparation and data preprocessing on your local machine but I suggest doing the training on an AWS instance. I will show you how to set up the training environment in the [Training section][training section].

### Windows 10
1. Install TensorFlow version 1.4 by executing the following statement in the Command Prompt (this assumes you have python.exe set in your PATH environment variable)
    ```
    pip install tensorflow==1.4
    ```
2. Install the following python packages 
    ```
    pip install pillow lxml matplotlib
    ```
3. [Download protoc-3.4.0-win32.zip from the Protobuf repository][protobuf win] (It must be version 3.4.0!)
4. Extract the Protobuf .zip file e.g. to ``C:\Program Files\protoc-3.4.0-win32``
5. Create a new directory somewhere and name it ``tensorflow``
6. Clone TensorFlow's *models* repository from the ``tensorflow`` directory by executing 
    ```
    git clone https://github.com/tensorflow/models.git
    ```
7. Navigate to the ``models`` directory in the Command Prompt and execute 
    ```
    git checkout f7e99c0
    ```

    This is important because the code from the ``master`` branch won't work with TensorFlow version 1.4. Also, this commit has already fixed broken models from previous commits.

8. Navigate to the ``research`` folder and execute 
    ```cmd
    ## The quotation marks are needed!
    “C:\Program Files\protoc-3.4.0-win32\bin\protoc.exe” object_detection/protos/*.proto --python_out=. 
    ```
9. If step 8 executed without any error then execute ``python builders/model_builder_test.py``
10. In order to access the modules from the research folder from anywhere, the ``models``, ``models/research``, ``models/research/slim`` & ``models/research/object_detection`` folders need to be set as PATH variables like so:

    10.1. Go to ``System`` -> ``Advanced system settings`` -> ``Environment Variables...`` -> ``New...`` -> name the variable ``PYTHONPATH`` and add the absolute path from the folders mentioned above

    ![pythonpath][pythonpath win]

    10.2. Double-click on the ``Path`` variable and add ``%PYTHONPATH%``

    ![path variable][path variable win]

Source: [cdahms' question/tutorial on Stackoverflow][cdahms question].

### Linux
1. Install TensorFlow version 1.4 by executing 
    ```
    pip install tensorflow==1.4
    ```
2. Install the following packages 
    ```
    sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
    ```
3. Create a new directory somewhere and name it ``tensorflow``
4. Clone TensorFlow's *models* repository from the ``tensorflow`` directory by executing 
    ```
    git clone https://github.com/tensorflow/models.git
    ```
5. Navigate to the ``models`` directory in the Command Prompt and execute 
    ```
    git checkout f7e99c0
    ```

    This is important because the code from the ``master`` branch won't work with TensorFlow version 1.4. Also, this commit has already fixed broken models from previous commits.

6.  Navigate to the ``research`` folder and execute 
    ```
    protoc object_detection/protos/*.proto --python_out=.

    export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    ```
7.  If the step 6 executed without any errors then execute 
    ```
    python object_detection/builders/model_builder_test.py
    ```

## Datasets
As always in deep learning: Before you start coding you need to gather the right datasets. For this project, you will need images of traffic lights with labeled bounding boxes.
In sum there are 4 datasets you can use:
1. [Bosch Small Traffic Lights Dataset][bosch dataset]
2. [LaRA Traffic Lights Recognition Dataset][lara dataset]
3. Udacity's ROSbag file from Carla
4. Traffic lights from Udacity's simulator

I ended up using Udacity's ROSbag file from Carla only and if you carefully follow along with this tutorial the images from the ROSbag file will be enough to have a working classifier for real-world AND simulator examples. There are two approaches on how to get the data from the ROSbag file (and from Udacity's simulator):

### 1. The Lazy Approach
You can download Vatsal Srivastava's dataset and my dataset for this project. The images are already labeled and a [TFRecord file][tfrecord file] is provided as well:

1. [Vatsal's dataset][coldknight dataset]
2. [My dataset][alex lechner dataset]

Both datasets include images from the ROSbag file and from the Udacity Simulator.

My dataset is a little sparse (at least the amount of yellow traffic lights is small) but Vatsal's dataset has enough images to train. However, I encourage you to use both. For example, I used Vatsal's data for training and mine for evaluation.

### 2. The Diligent Approach
If you have enough time, love to label images, read tutorials about traffic light classification before this one or want to gather more data, then this is the way to go:

#### 2.1 Extract images from a ROSbag file
For the simulator data, my team colleagues [Clifton Pereira][clifton pereira] and [Ian Burris][ian burris] drove around the track in the simulator and recorded a ROSbag file of their rides. Because Udacity provides the students with a ROSbag file from their Car named Carla where (our and) your capstone project will be tested on the code/procedure for extracting images will be (mostly) the same. **The steps below assume you have ros-kinetic installed either on your local machine (if you have Linux as an operating system) or in a virtual environment (if you have Windows or Mac as an operating system)**

1. Open a terminal and launch ROS 
    ```sh
    roscore
    ```
2. Open another terminal (but do NOT close or exit the first terminal!) and play the ROSbag file
    ```sh
    rosbag play -l path/to/your_rosbag_file.bag
    ```
3. Create a directory where you want to save the images
4. Open another, third terminal and navigate to the newly created directory and... 
    
    1. ...execute the following statement if you have a ROSbag file from Udacity's simulator:
        ```sh
        rosrun image_view image_saver _sec_per_frame:=0.01 image:=/image_color
        ```
    
    2. ...execute the following statement if you have a ROSbag file from Udacity's Car Carla:
        ```sh
        rosrun image_view image_saver _sec_per_frame:=0.01 image:=/image_raw
        ```

    As you can see the difference is the rostopic after ``image:=``.

These steps will extract the (camera) images from the ROSbag file into the folder where the code is executed. Please keep in mind that the ROSbag file is in an infinite loop and won't stop when the recording originally ended so it will automatically start from the beginning. If you think you have enough data you should interrupt one of the open terminals.

If you can't execute step 4.1 or 4.2 you probably don't have ``image_view`` installed. To fix this install ``image_view`` with ``sudo apt-get install ros-kinetic-image-view``.

Hint: You can see the recorded footage of your ROSbag file by opening another, fourth terminal and executing ``rviz``.

#### 2.2 Data labeling
After you have your dataset you will need to label it by hand. For this process I recommend you to [download labelImg][labelImg]. It's very user-friendly and easy to set up.
1. Open labelImg, click on ``Open Dir`` and select the folder of your traffic lights
2. Create a new folder within the traffic lights folder and name it ``labels``
3. In labelImg click on ``Change Save Dir`` and choose the newly created ``labels`` folder

Now you can start labeling your images. When you have labeled an image with a bounding box hit the ``Save`` button and the program will create a .xml file with a link to your labeled image and the coordinates of the bounding boxes.

Pro tip: I'd recommend you to split your traffic light images into 3 folders: Green, Yellow, and Red. The advantage is that you can check ``Use default label`` and use e.g. ``Red`` as an input for your red traffic light images and the program will automatically choose ``Red`` as your label for your drawn bounding boxes.

![labeling a traffic light][labeling img] 

#### 2.3 Create a TFRecord file
Now that you have your labeled images you will need to create a TFRecord file in order to retrain a TensorFlow model. A TFRecord is a binary file format which stores your images and ground truth annotations. But before you can create this file you will need the following:
1. A [``label_map.pbtxt``][label map] file which contains your labels (``Red``, ``Green``, ``Yellow`` & ``off``) with an ID (IDs must start at 1 instead of 0)
2. [Setup Tenorflow][set up tensorflow]
3. A script which creates the TFRecord file for you (feel free to use my [``create_tf_record.py``][create_tf_record] file for this process)

Please keep in mind that your ``label_map.pbtxt`` file can have more than 4 labels depending on your dataset. For example, if you're using the [Bosch Small Traffic Lights Dataset][bosch dataset] you will most likely have about 13 labels.

In case you are using the dataset from Bosch, all labels and bounding boxes are stored in a .yaml file instead of a .xml file. If you are developing your own script to create a TFRecord file you will have to take care of this. If you are using my script I will now explain how to execute it and what it does:

For datasets with **.yaml** files (e.g.: Bosch dataset) execute: 
```
python create_tf_record.py --data_dir=path/to/your/data.yaml --output_path=your/path/filename.record --label_map_path=path/to/your/label_map.pbtxt
```

For datasets with **.xml** files execute: 
```
python create_tf_record.py --data_dir=path/to/green/lights,path/to/red/lights,path/to/yellow/lights --annotations_dir=labels --output_path=your/path/filename.record --label_map_path=path/to/your/label_map.pbtxt
```

You will know that everything worked fine if your .record file has nearly the same size as the sum of the size of your images. Also, you have to execute this script for your training set, your validation set (if you have one) and your test set separately.

As you can see you don't need to specify the ``annotations_dir=`` flag for .yaml files because everything is already stored in the .yaml file.

The second code snippet (for datasets with .xml files) assumes you have the following folder structure:
```
path/to
|
└─green/lights
│   │  img01.jpg
│   │  img02.jpg
│   │  ...
|   |
│   └──labels
│      │   img01.xml
│      │   img02.xml
│      │   ...
|
└─red/lights
│   │  ...
|   |
│   └──labels
│      │   ...
|
└─yellow/lights
│   │  ...
|   |
│   └──labels
│      │   ...

```

**Important note about the dataset from Bosch**: This dataset is very large in size because every image takes approximately 1 MB of space. However, I've managed to reduce the size of each image **drastically** by simply converting it from a .png file to a .jpg file (for some reason the people from Bosch saved all images as PNG). You want to know what I mean by 'drastically'? Before the conversion from PNG to JPEG, my .record file for the test set was **11,3 GB** in size. After the conversion, my .record file for the test set was only **842 MB** in size. I know... :open_mouth: :open_mouth: :open_mouth: Trust me, I've checked the code and images and tested my script multiple times until I was finally convinced. The image conversion is already implemented in the [``create_tf_record.py``][create_tf_record] file. 

## Training

### 1. Choosing a model
So far you should have a TFRecord file of the dataset(s) which you have either downloaded or created by yourself. Now it's time to select a model which you will train. You can [see the stats of and download the Tensorflow models from the model zoo][models zoo]. In sum I've trained 3 TensorFlow models and compared them based on their performance and precision:

* [SSD Inception V2 Coco (17/11/2017)][ssd inception 171117] Pro: Very fast, Con: Not good generalization on different data
* [SSD Inception V2 Coco (11/06/2017)][ssd inception] Pro: Very fast, Con: Not good generalization on different data
* [Faster RCNN Inception V2 Coco (28/01/2018)][faster rcnn inception] Pro: Good precision and generalization of different data, Con: Slow
* [Faster RCNN Resnet101 Coco (11/06/2017)][faster rcnn resnet101] Pro: Highly Accurate, Con: Very slow

Our team ended up using **SSD Inception V2 Coco (17/11/2017)** because it has good results for its performance.

You may ask yourself why the date after the model's name is important. As I've mentioned in the [TensorFlow set up section][set up tensorflow] above, it's very important to check out a specific commit from the 'models' repository because the team has fixed broken models. That's why it is important. And if you don't want to see the following results after a very long training session I encourage you to stick to the newest models or the ones I've linked above:

![bad performance][bad performance]

You get these result too if you have too few training steps. You can imagine how much time I've spent to figure this out...

After you've downloaded a model, create a new folder e.g. ``models`` and unpack the model with [7-zip on Windows][7-zip win] or ``tar -xvzf your_tensorflow_model.tar.gz`` on Linux.

### 2. Configure the .config file of the model
You will need to [download the .config file for the model you've chosen][model configs] or you can simply [download the .config files of this repository][alex lechner model configs] if you've decided to train the images on one of the models mentioned above.

If you want to configure them on your own there are some important changes you need to make. For this walkthrough, I will assume you are training on the Udacity Carla dataset with ~~Faster RCNN Inception V2~~ SSD Inception V2.

**TensorFlow model configs might differ but the following steps below are the same for every model!**

1. Change ``num_classes: 90`` to the number of labels in your ``label_map.pbtxt``. This will be ``num_classes: 4``
2. Set the default ``max_detections_per_class: 100`` and ``max_total_detections: 300`` values to a lower value for example ``max_detections_per_class: 10`` and ``max_total_detections: 10``
4. Change ``fine_tune_checkpoint: "PATH_TO_BE_CONFIGURED/model.ckpt"`` to the directory where your downloaded model is stored e.g.: ``fine_tune_checkpoint: "models/your_tensorflow_model/model.ckpt"``
5. Set ``num_steps: 200000`` down to ``num_steps: 20000``
6. Change the ``PATH_TO_BE_CONFIGURED`` placeholders in ``input_path`` and ``label_map_path`` to your .record file(s) and ``label_map.pbtxt``

For Faster RCNN Inception V2:

1. Change the default ``min_dimension: 600`` and ``max_dimension: 1024`` values to the minimum value (height) and the maximum value (width) of your images like so

    ```
    keep_aspect_ratio_resizer {
        min_dimension: 1096
        max_dimension: 1368
    }
    ```

2. You can increase ``batch_size: 1`` to ``batch_size: 3`` or even higher


If you don't want to use evaluation/validation in your training, simply remove those blocks from the config file. However, if you do use it make sure to set ``num_examples`` in the ``eval_config`` block to the sum of images in your .record file.

You can [take a look at the .config files of this repsoitory][alex lechner model configs] for reference. I've configured a few things like batch size and dropout as well. As I've mentioned earlier I've used [Vatsal's dataset][coldknight dataset] for training and my dataset for validation so don't get confused by the filename of my .record file ``jpg_udacity_train.record``.

### 3. Setup an AWS spot instance
For training, I recommend setting up an AWS spot instance. Training will be much faster and you can train multiple models simultaneously on different spot instances (like I did):

![simultaneous training][simultaneous training]
_Left: Training Faster RCNN Inception V2 Coco, Right: Training SSD Inception V2 Coco_

To set up an AWS spot instance do the following steps:
1. [Login to your Amazon AWS Account][aws login]
2. Navigate to ``EC2`` -> ``Instances`` -> ``Spot Requests`` -> ``Request Spot Instances``
3. Under ``AMI`` click on ``Search for AMI``, type ``udacity-carnd-advanced-deep-learning`` in the search field, choose ``Community AMIs`` from the drop-down and select the AMI (**This AMI is only available in US Regions so make sure you request a spot instance from there!**)
4. Delete the default instance type, click on ``Select`` and select the ``p2.xlarge`` instance
5. Uncheck the ``Delete`` checkbox under ``EBS Volumes`` so your progress is not deleted when the instance get's terminated
6. Set ``Security Groups`` to ``default``
7. Select your key pair under ``Key pair name`` (if you don't have one create a new key pair)
8. At the very bottom set ``Request valid until`` to about 5 - 6 hours and set ``Terminate instances at expiration`` as checked (You don't have to do this but keep in mind to receive a very large bill from AWS if you forget to terminate your spot instance because the default value for termination is set to 1 year.)
9. Click ``Launch``, wait until the instance is created and then connect to your instance via ssh

![spot instance][spot instance]

### 4. Training the model
1. When you're connected with the instance execute the following statements consecutively:
    ```sh
    sudo apt-get update
    pip install --upgrade dask
    pip install tensorflow-gpu==1.4
    ```

2. [Set up TensorFlow for Linux][tf setup linux] (**but skip step one because we've already installed tensorflow-gpu!**)
3. Clone your classification repository and create the folders ``models`` & ``data`` (in your project folder) if they are not tracked by your VCS.
4. Upload the datasets to the ``data`` folder 
    1. If you're using my dataset you can simply execute the following statements in the ``data`` folder: 

        ```sh
        wget https://www.dropbox.com/s/vaniv8eqna89r20/alex-lechner-udacity-traffic-light-dataset.zip?dl=0
        unzip alex-lechner-udacity-traffic-light-dataset.zip?dl=0 ## Don't miss the ``?dl=0`` part when unzipping!
        ``` 

5. Navigate to the ``models`` folder in your project folder and download your tensorflow model with 
    ```sh
    wget http://download.tensorflow.org/models/object_detection/your_tensorflow_model.tar.gz
    tar -xvzf your_tensorflow_model.tar.gz
    ```

6. Copy the file ``train.py`` from the ``tensorflow/models/research/object_detection`` folder to the root of your project folder
7. Train your model by executing the following statement in the root of your project folder
    ```
    python train.py --logtostderr --train_dir=./models/train --pipeline_config_path=./config/your_tensorflow_model.config
    ```

### 5. Freezing the graph
When training is finished the trained model needs to be exported as a frozen inference graph. Udacity's Carla has TensorFlow Version 1.3 installed. However, the minimum version of TensorFlow needs to be Version 1.4 in order to freeze the graph but note that this does not raise any compatibility issues. 
If you've trained the graph with a higher version of TensorFlow than 1.4, don't panic! As long as you downgrade Tensorflow to version 1.4 before running the script to freeze the graph you should be fine.
To freeze the graph:
1. Copy ``export_inference_graph.py`` from the ``tensorflow/models/research/object_detection`` folder to the root of your project folder
2. Now freeze the graph by executing
    ```
    python export_inference_graph.py --input_type image_tensor --pipeline_config_path ./config/your_tensorflow_model.config --trained_checkpoint_prefix ./models/train/model.ckpt-20000 --output_directory models
    ```

    This will freeze and output the graph as ``frozen_inference_graph.pb``.

## Recommendation: Use SSD Inception V2
At first, our team was using Faster RCNN Inception V2 model. This model takes about 2.9 seconds to classify images which is - besides the name of the model - not that fast. The advantage about training the Faster RCNN Inception V2 is the generalization of the model to new, different & unseen images which means the model was only trained on the image data of Udacity's parking lot and was able to classify the light state of the traffic lights in the simulator too. So why did we change the model to SSD Inception V2? 

Our code was successfully tested on Carla but it failed in the simulator. This might sound funny - and it actually is - but the reason why it failed is that the frequency of changing lights in the simulator is set ridiculously high so the light was changing every 2 - 3 seconds. The configuration of our traffic light detector node in our project is set to 3 consecutive images of traffic lights until the final state (Red, Green, Yellow or Unknown) and action is passed to the agent/car. That's the reason why we changed the model from Faster RCNN Inception V2 to SSD Inception V2.

The good thing about SSD Inception V2 is its speed and performance. Sometimes the SSD model misses to classify an image with over 50% certainty but in general, it is doing a good job for its performance. However, unlike the Faster RCNN Inception V2 the model does not a good job of classifying new, different images. For example, I've trained the SSD model first on Udacity's parking lot data with 10.000 steps and it did a good job on classifying the parking lot traffic lights but the model did not classify a single image from the simulator data. After the training, I did transfer learning on the simulator data with 10.000 steps as well. After the training something interesting happened: The model was able to classify the simulator data BUT the model "forgot" about its previous training on the Udacity parking lot data and therefore only classified 2 out of 10 images from the Udacity parking lot dataset.

### Conclusion
Our team is using now 2 trained SSD Inception V2 models for [our Capstone project][capstone project]:

* 1 SSD model for real-world data
* 1 SSD model for simulator data
  
If you are using this approach as well I recommend you to train 2 SSD models simultaneously on an [AWS instance](#3-setup-an-aws-spot-instance). Because the SSD model "forgets" about the old trained data you don't have to do transfer learning and you can safely train 1 model on simulator data and 1 model on real-world data separately (and simultaneously) which will save you a tremendous amount of time.

SSD trained on parking lot images  |  SSD trained on simulator images
:---------------------------------:|:---------------------------------:
![ssd udacity][ssd udacity]        | ![ssd simulator][ssd simulator]

**[Take a look at the Jupyter Notebook][jupyter notebook] to see the results.**

**UPDATE:** At first, I've trained both SSD models with "only" 10.000 steps and the results were okay. In order to have better results, I've trained it for another 10.000 steps so I'd recommend training both models with 20.000 steps in sum. To give you an example: Both SSD models had a problem to classify traffic lights which were far away in the first 10.000 steps session. After training them for another 10.000 steps this problem was solved (and they had a higher certainty in classifying the light state as well).

## Troubleshooting
In case you're running into any of the errors listed below, the solutions provided will fix it:

* _ValueError: Tried to convert 't' to a tensor and failed. Error: Argument must be a dense tensor: range(0, 3) - got shape [3], but wanted []._

Go to ``tensorflow/models/research/object_detection/utils`` and edit the ``learning_schedules.py`` file. Go to the line 167 and replace it with:
```python
rate_index = tf.reduce_max(tf.where(tf.greater_equal(global_step, boundaries),
                                      list(range(num_boundaries)),
                                      [0] * num_boundaries))
```

source: [epratheeban's answer on GitHub][epratheeban github]

* _ValueError: Protocol message RewriterConfig has no "optimize_tensor_layout" field._

Go to ``tensorflow/models/research/object_detection/`` and edit the  ``exporter.py`` file. Go to line 71 and change ``optimize_tensor_layout`` to ``layout_optimizer``.

If the same error occurs with the message _[...] has no "layout_optimizer" field._ then you have to change ``layout_optimizer`` to ``optimize_tensor_layout``.

* Can't ssh into the AWS instance because of _port 22: Resource temporarily unavailable_

Go to ``Network & Security`` -> ``Security Groups`` -> right click on the security group that is used on your spot instance (propably ``default``) -> ``Edit inbound rules`` and set ``Source`` of SSH and Custom TCP to ``Custom`` and ``0.0.0.0/0`` like so:

![aws inbound rules][aws inbound rules]

* Can't install packages on Linux because of _dpkg: error: dpkg status database is locked by another process_

This error will probably occur when trying to execute ``sudo apt-get install protobuf-compiler python-pil python-lxml python-tk`` on the AWS spot instance after upgrading tensorflow-gpu to Version 1.4. Execute the following lines and try installing the packages again: 
```sh
sudo rm /var/lib/dpkg/lock
sudo dpkg --configure -a
```

* _tensorflow.python.framework.errors_impl.InternalError: Dst tensor is not initialized._

This error occurs when you don't have enough free available memory on your GPU to train. To fix this execute ``sudo fuser -v /dev/nvidia*`` and look for the process that is currently using your memory from the GPU.

![kill memory][kill memory]

Then kill the process by executing ``sudo kill -9 <PID-to-kill>``


## Summary
If you are using Vatsal's and my dataset you only need to:
1. [Download the datasets](#1-the-lazy-approach)
2. [Set up TensorFlow **only on the training instance**, do the training and export the model][training section]

If you are using your own dataset you need to:
1. [Set up TensorFlow locally][set up tensorflow] (because of creating TFRecord files)
2. [Create your own datasets](#2-the-diligent-approach)
3. [Set up TensorFlow again on a training instance (if the training instance is not your local machine), do the training and export the model][training section]


Training instance = System, where you train the TensorFlow model (probably an AWS instance and not your local machine)