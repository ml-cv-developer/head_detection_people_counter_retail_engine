## Packages

        pip install opencv-python
        pip install tensorflow==1.15.*


## Description

Train the model using head dataset and detect the heads. Based on this, count the incoming and outgoing in the retail.
Support 2 models, ssd_mobile_net_v1 and faster_rcnn_resnet50 (this isn't committed to repository)

![img](sample.jpg)

## Running method

- running script

        python3 main.py [video_file]
        
    if you ignore video_file parameter, then will load the video or camera streaming from setting.py automatically.
