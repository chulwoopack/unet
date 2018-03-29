# Unet
Semantic Segmentation neural net based on Unet [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597). Batch norms and dropouts are added to the network as well as weighted cross entropy loss for multi-class segmentation.

<img src="Images/framework.png" width="800px"/>

### Disclaimer
Most of this code is from kimoktm. Please find the original code from here (https://github.com/kimoktm/U-Net)

### Objective
This is a class and reserach project to convert a grayscale/color image to binary image. The domain of image is specified on historical machine-printed/hand-written document image.

### Dependencies
- python 2.7
- [TensorFlow >=1.0.0](https://www.tensorflow.org/get_started/os_setup)
- In addition, please `pip install -r requirements.txt` to install the following packages:
    - `Pillow`
    - `numpy`
    - `tensorflow>=1.0.0`

### Data Preprocessing
Image data will be loaded from a given path. The dataset images should be in the same folder (im1_image.png, im1_label.png) with PNG or JPG format. The label images must be 1 channel images.
- To load dataset find unet_train.ipynb and run cell 1 and 2.

### Training
- To train Unet find unet_train.ipynb and run cell 1 and 2.
   
### Evaluation
- To be done...

### Citing Unet
Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical
image segmentation. In: International Conference on Medical Image Computing
and Computer-Assisted Intervention. pp. 234–241. Springer (2015) [pdf](https://arxiv.org/abs/1505.04597).

    @inproceedings{fusenet2016accv,
     author    = "Olaf Ronneberger, Philipp Fischer, and Thomas Brox",
     title     = "U-Net: Convolutional Networks for Biomedical Image Segmentation",
     booktitle = "Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015",
     year      = "2015",
     month     = "October",
    }
