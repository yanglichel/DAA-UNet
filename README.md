# DAA-UNet
#DAA-UNet
The software development environment based on PyTorch used PyCharm Community Edition 2024 and Anaconda3 2024 includes 
    python3.12, 
    torchvision0.18.1,
    torch2.3.1, 
    cuda121, 
    numpy1.26.3, and others.
 The training model employs the RMSprop optimizer for all optimizations, with the Lovasz-Softmax as the loss function.
 The training is conducted for 300 epochs, starting with an initial learning rate of 0.0001, which is decayed by a factor of 0.8 every 20 epochs.
 The highest mean Dice Similarity Coefficient (mDSC) on the test set is recorded, and the corresponding model parameters are saved.


Copyright Notice
Copyright (c) [2024] [Tianhan Hu，Li Yang]. All rights reserved.
   This code is developed and maintained by [Tianhan Hu，Li Yang]. Without explicit written permission, no organization or individual is allowed to copy, modify, distribute,
 or use this code in any other way.This code is intended solely for educational and research purposes. It must not be used for commercial purposes. 
If you wish to use this code in a commercial project, please contact [Tianhan Hu，Li Yang] to obtain the necessary authorization.
Please note that this code may contain third-party libraries or components, which may be protected by their own independent copyright and licensing agreements. 
When using these libraries or components, please ensure that you comply with their respective licensing agreements.



Instructions
    1.Data location(train_data_path)
        Dataset original image storage address: D:\pw\x1\DAA-UNet\data\train
        Label data storage address:E:\pw_2024\PycharmProjects\data\seg
    2.Model:  DAAUNet.py
    3.Run train.py directly, where the train_image folder stores the effect images during the training process
    4.Save weights in train_data_path
