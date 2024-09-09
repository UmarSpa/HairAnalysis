# Hair detection, segmentation, and hairstyle classification in the wild
Created by Muhammad Umar Riaz - University of Brescia (2016)

## Introduction

This is the code for the paper:  

[Hair detection, segmentation, and hairstyle classification in the wild](https://www.sciencedirect.com/science/article/pii/S0262885618300143)  
U.R. Muhammad, M. Svanera, R. Leonardi, and S. Benini  
Image and Vision Computing, 2018. 

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
This is the continuation of our previous work ([paper](http://ieeexplore.ieee.org/document/7532494/)):
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
[Figaro-1k](https://www.dropbox.com/scl/fi/bkzbwgobxayoaeqohgvrp/Figaro-1k.zip?rlkey=qahueoko45prpzmsadzmus5ga&dl=0): It contains 1050 unconstrained view images with persons, subdivided into seven different hairstyles classes (straight, wavy, curly, kinky, braids, dreadlocks, short), where each image is provided with the related manually segmented hair mask.  
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
N.B. you need to download [model_caffenet.caffemodel](https://www.dropbox.com/scl/fi/r6jp05fo8g9srlfg01cjo/model_caffenet.caffemodel?rlkey=pmiouw039notr0d7fgbrsdy3j&st=ndvremo7&dl=0) and place it in *Tools/CaffeNet/*.  


