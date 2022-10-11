# Description
This software classifies knee osteoarthritis severity from plain radiographs with Kellgren-Lawrence scale using ResNet-34 network. Built with a UI easy to use

![Example](https://user-images.githubusercontent.com/15198470/135897906-d5503242-900e-420e-ade7-a9520ff7a51c.gif)

# How to run
- Download the model from https://bit.ly/3lGaeWj.

- Place the model in the main folder.

- Run ``main.py``
    
#  Disclaimer
This software is not a medical device, is not made for diagnostic purposes and intended for the use in research settings only. Commercial use is not allowed by any means.

# How to cite 

    @Article{diagnostics12102362,
    AUTHOR = {Cueva, Joseph Humberto and Castillo, Darwin and Espinós-Morató, Héctor and Durán, David and Díaz, Patricia and Lakshminarayanan, Vasudevan},
    TITLE = {Detection and Classification of Knee Osteoarthritis},
    JOURNAL = {Diagnostics},
    VOLUME = {12},
    YEAR = {2022},
    NUMBER = {10},
    ARTICLE-NUMBER = {2362},
    URL = {https://www.mdpi.com/2075-4418/12/10/2362},
    ISSN = {2075-4418},
    ABSTRACT = {Osteoarthritis (OA) affects nearly 240 million people worldwide. Knee OA is the most common type of arthritis, especially in older adults. Physicians measure the severity of knee OA according to the Kellgren and Lawrence (KL) scale through visual inspection of X-ray or MR images. We propose a semi-automatic CADx model based on Deep Siamese convolutional neural networks and a fine-tuned ResNet-34 to simultaneously detect OA lesions in the two knees according to the KL scale. The training was done using a public dataset, whereas the validations were performed with a private dataset. Some problems of the imbalanced dataset were solved using transfer learning. The model results average of the multi-class accuracy is 61%, presenting better performance results for classifying classes KL-0, KL-3, and KL-4 than KL-1 and KL-2. The classification results were compared and validated using the classification of experienced radiologists.},
    DOI = {10.3390/diagnostics12102362}
    }
