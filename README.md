# cv-project: Sign Language Translation with a Pure Transformer Architecture
Computer Vision Master's Project 2022/2023
Sign Language Recognition and Translation

This repo contains the code for different architectures realising training and evaluation of Sign Language Translation. For more information read ```CVProject_FinalPaper.pdf```.  

Main model versions: Swin 2d/3d Encoder.

## Requirements

Install required packages in a with conda environment using the ```requirements.txt``` file.
```
$ conda create --name <env> --file <this file>
```
(Created from Windows 64Bit)  


Alternative: 

Install using the ```requirementspip.txt``` file.
```
pip install -r requirementspip.txt
```





## Training 

Go to ```slt``` folder of the model version you want to train.  

This command will initiate the training process using the training dataset, perform validation using the validation dataset, and save essential model parameters, vocabularies, and validation results. Ensure that all necessary information is properly configured in the data, training, and model sections of the configuration file.

```
python -m signjoey train configs/sign.yaml
```

---------------------------------------------------------------------------------------------------------------------

## Implementations of Different Encoders
Swin2dEncoder: Uses the Swin Transformer as encoder from the paper Liu, Z. et al. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–1002.  

Swin3dEncoder: Uses the Video Swin Transformer as encoder from the paper Liu, Z. et al. (2022). Video swin transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3202–321.


## Implementations of Different Backbones for Feature Extraction
SLTCNNBackbone: Original implementation of baseline model from the paper Camgöz, N. C. et al. (2020b). Sign language transformers: Joint end-to-end sign language recognition and translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.  

EfficientB7Backbone: Uses the EfficientnetB7 (or alternatively EfficientnetB0) as backbone from the paper Tan, M. et al. (2019). Efficientnet: Re-thinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR.  

ViTBackbone: Uses the Vision Transformer as backbone from the paper Dosovitskiy, A. et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale.  

Swin2DBackbone: Uses the Swin Transformer as backbone from the paper Liu, Z. et al. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–1002.


All implementations are based on the Sign Language Transformer implementation by Camgöz et al. (https://github.com/neccam/slt).

---------------------------------------------------------------------------------------------------------------------

Relevant Papers:
- Camgöz: Joint End-to-End Sign Language Recognition and Translation, 2020
- Liu: Video Swin Transformer, 2022

Other Related Papers:
- Camgöz: Multi-channel Transformers for Multi-articulatory Sign Language Translation, 2020
- ANANTHANARAYANA: Deep Learning Methods for SLT, 2021
- Selva: Video Transformers: A Survey, 2022
- Dosovitskiy: Vision Transformer, AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE, 2021
- Lu: ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks, 2019

Dataset:
- RWTH-PHOENIX-Weather 2014 T: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/

Other SLT Datasets:
- RWTH-PHOENIX-Weather: https://www-i6.informatik.rwth-aachen.de/~forster/database-rwth-phoenix.php
- DGS Corpus: https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start_de.html
  -> how to use: https://www.sign-lang.uni-hamburg.de/dgs-korpus/index.php/publications.html
- Content4All: SWISSTXT-NEWS, SWISSTXT-WEATHER, VRT-NEWS: https://www.cvssp.org/data/c4a-news-corpus/
