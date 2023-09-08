# cv-project
Computer Vision Master's Project 2022/2023
Sign Language Recognition and Translation

This repo contains the code for different architectures realising training and evaluation of Sign Language Translation.

## Implementations of Different Backbones for Feature Extraction
SLTCNNBackbone: Original implementation of baseline model from the paper Camgöz, N. C. et al. (2020b). Sign language transformers: Joint end-to-end sign language recognition and translation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
EfficientB7Backbone: Uses the EfficientnetB7 (or alternatively EfficientnetB0) as backbone from the paper Tan, M. et al. (2019). Efficientnet: Re-thinking model scaling for convolutional neural networks. In International conference on machine learning, pages 6105–6114. PMLR.
ViTBackbone: Uses the Vision Transformer as backbone from the paper Dosovitskiy, A. et al. (2020). An image is worth 16x16 words: Transformers for image recognition at scale.
Swin2DBackbone: Uses the Swin Transformer as backbone from the paper Liu, Z. et al. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–1002.

## Implementations of different Encoders
Swin2dEncoder: Uses the Swin Transformer as encoder from the paper Liu, Z. et al. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 10012–1002.
Swin3dEncoder: Uses the Video Swin Transformer as encoder from the paper Liu, Z. et al. (2022). Video swin transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3202–321.



All implementations are based on the Sign Language Transformer implementation by Camgöz et al. (https://github.com/neccam/slt).

---------------------------------------------------------------------------------------------------------------------
Relevant Papers:
- Camgöz: Joint End-to-End Sign Language Recognition and Translation, 2020
- Camgöz: Multi-channel Transformers for Multi-articulatory Sign Language Translation, 2020
- ANANTHANARAYANA: Deep Learning Methods for SLT, 2021

General Background Papers:
- Selva: Video Transformers: A Survey, 2022
- Liu: Video Swin Transformer, 2022
- Dosovitskiy: Vision Transformer, AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE, 2021
- Lu: ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks, 2019

Relevant Datasets:
- RWTH-PHOENIX-Weather: https://www-i6.informatik.rwth-aachen.de/~forster/database-rwth-phoenix.php
- RWTH-PHOENIX-Weather 2014 T: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/
- DGS Corpus: https://www.sign-lang.uni-hamburg.de/meinedgs/ling/start_de.html
  -> how to use: https://www.sign-lang.uni-hamburg.de/dgs-korpus/index.php/publications.html
- Content4All: SWISSTXT-NEWS, SWISSTXT-WEATHER, VRT-NEWS: https://www.cvssp.org/data/c4a-news-corpus/
