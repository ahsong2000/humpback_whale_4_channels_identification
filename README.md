# humpback_whale_4_channels_identification
Can you identify a whale by its tail?

Author: Chuen-Song (Ahsong) Chen    |     Email: ahsong.chen@gmail.com    

![GitHub Logo](/images/happy_whale.png)

Siamese Neural Networks:
1. Twin networks (CNN branch model) accept distinct images and learn the structural feature representation
2. Output of twin networks are joined by an energy function at the top
3. This energy function computes some metrics between twin network's feature representation.
4. Head model take the computed metrics and learn the relationship between two images.
4. Siamese NNs are popular for finding similarity or a relationship between two comparable images.

In this repos, Siamese NNs are trained by 4 different channels of images (red,green,blue, and grey scale).
The maximum score/probability is selected for final prediction.

Dataset Link: https://www.kaggle.com/c/humpback-whale-identification/data
