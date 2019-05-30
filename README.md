# Hair detection, segmentation, and hairstyle classification in the wild
Created by Muhammad Umar Riaz - University of Brescia (2016)

## Introduction

This is the code for the paper:  

[Hair detection, segmentation, and hairstyle classification in the wild](https://www.sciencedirect.com/science/article/pii/S0262885618300143)  
U.R. Muhammad, M. Svanera, R. Leonardi, and S. Benini  
Image and Vision Computing, 2018. 

[Project page](http://www.eecs.qmul.ac.uk/~urm30/Hair.html)

## Cite
If you find this code useful in your research, please, consider citing our paper:
```
@article{umar2018hair,
  title={Hair detection, segmentation, and hairstyle classification in the wild},
  author={Muhammad, Umar Riaz and Svanera, Michele and Leonardi, Riccardo and Benini, Sergio},
  journal={Image and Vision Computing},
  year={2018},
  publisher={Elsevier}
}
```

## Related work
This is the continuation of our previous work ([project page](http://www.eecs.qmul.ac.uk/~urm30/Figaro.html), [paper](http://ieeexplore.ieee.org/document/7532494/)):
```
@inproceedings{svanera2016figaro,
  title={Figaro, hair detection and segmentation in the wild},
  author={Svanera, Michele and Muhammad, Umar Riaz and Leonardi, Riccardo and Benini, Sergio},
  booktitle={International Conference on Image Processing (ICIP)},
  year={2016},
  organization={IEEE}
}
```
## Dataset 
[Figaro-1k](http://www.eecs.qmul.ac.uk/~urm30/HairMaterial/Figaro-1k.zip): It contains 1050 unconstrained view images with persons, subdivided into seven different hairstyles classes (straight, wavy, curly, kinky, braids, dreadlocks, short), where each image is provided with the related manually segmented hair mask.  
The 7 classes are distributed in this order:  
- straight: frame00001-00150  
- wavy: frame00151-00300  
- curly: frame00301-00450  
- kinky: frame00451-00600  
- braids: frame00601-00750  
- dreadlocks: frame00751-00900  
- short-men: frame00091-01050  

## Demo
You can run the demo on any jpg format image (that must be placed in folder *Data*) by running main.py file.  
N.B. you need to download [model_caffenet.caffemodel](http://www.eecs.qmul.ac.uk/~urm30/HairMaterial/model_caffenet.caffemodel) and place it in *Tools/CaffeNet/*.  


