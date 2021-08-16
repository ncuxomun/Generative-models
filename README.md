# Generative-models

In this repository, I have prepared examples of classical generative models - simple convolutional autoencoder (AE) and vanilla convolutional generative adversarial model (GAN). The objectives are two-fold:
 1. Use autoencoder to reconstruct input maps as close as possible.
 2. Use GAN to generate channelized examples given random input noise. 
 3. In both cases, I used (unpacked) [channelized (binary) maps](https://github.com/ncuxomun/CNN-proxy/blob/main/channels_uncond_10k.7z). 
 4. Furthermore, the models are too constructed using PyTorch Lighning which nicely and neatly wraps PyTorch. 
 5. The AE model can be accessed [here](https://github.com/ncuxomun/Generative-models/blob/main/ae_m.py), which comes with saved weights that can be loaded using [pre-trained weights](https://github.com/ncuxomun/Generative-models/blob/main/m_ae_model).
 6. The GAN model can be accessed [here](https://github.com/ncuxomun/Generative-models/blob/main/gan_m_small.py), which comes with saved weights that can be loaded using [pre-trained weights](https://github.com/ncuxomun/Generative-models/blob/main/gan_small).
 7. Examples of reconstructed maps can be found below. First, I present test set that is followed by predicted maps using AE model. Then, I present a set of examples generated using GAN.

Test set - True Maps

![myimage-alt-tag](https://github.com/ncuxomun/Generative-models/blob/main/true.png)

Test set - Reconstructed Maps using AE

![myimage-alt-tag](https://github.com/ncuxomun/Generative-models/blob/main/pred.png)

Generated Maps using GAN

![myimage-alt-tag](https://github.com/ncuxomun/Generative-models/blob/main/gan.png)
