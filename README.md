# Semantic_Segmentation
This repo shows a sample code to implement semantic segmenation (people segmentation) using unet + mobilenetv2.
The dataset I used was coco2017. 


## Getting Started

### Prerequisites
* Keras 2.4.3
* Tensorflow 2.2.0
* Coco API
* Opencv for python


## Usage via command line

### Training the Model 
python unet_semantic_seg_train.py

### Training Stats
The training set is coco/train2017 and the validation set is coco/val2017. The pretained model under model is getting 96.22% accuracy for training set and 92.54% accuracy for validation set. 

### Inference
python unet_semantic_seg_inference.py

The testing set used for inference is test2017. Some results are shown below:  
![](result.png)


## References
1)https://towardsdatascience.com/master-the-coco-dataset-for-semantic-image-segmentation-part-1-of-2-732712631047
2)https://www.tensorflow.org/tutorials/images/segmentation 
3)https://cocodataset.org/#download
