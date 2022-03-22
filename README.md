# Deep-Energy: Unsupervised Training of Deep Neural Networks

This is a TensorFlow implementation of paper:

[Unsupervised Single Image Dehazing using Dark Channel Prior Loss](https://arxiv.org/abs/1812.07051)

Based on :[repository](https://github.com/AlonaGolts/Deep_Energy)

Abstract: Single image dehazing is a critical stage in many modern-day autonomous vision applications. Early prior-based methods often involved a time-consuming minimization of a hand-crafted energy function. Recent learning-based approaches utilize the representational power of deep neural networks (DNNs) to learn the underlying transformation between hazy and clear images. Due to inherent limitations in collecting matching clear and hazy images, these methods resort to training on synthetic data; constructed from indoor images and corresponding depth information. This may result in a possible domain shift when treating outdoor scenes. We propose a completely unsupervised method of training via minimization of the well-known, Dark Channel Prior (DCP) energy function. Instead of feeding the network with synthetic data, we solely use real-world outdoor images and tune the network's parameters by directly minimizing the DCP. Although our "Deep DCP" technique can be regarded as a fast approximator of DCP, it actually improves its results significantly. This suggests an additional regularization obtained via the network and learning process. Experiments show that our method performs on par with large-scale
supervised methods.

## Getting Started

This repository contains: 

- Test function for each application: `Test_Dehaze.py`
- Implementations of all energy functions: `Dehazing_Loss.py`
- Neural network model we use throughout all experiments: `Models.py`
- Training function to train your own model: `Deep_Energy.py`
- Implementations of Our Contribution to the Dehazing Algorithm: `our_dehaze.py` 
- Utilities: `train_utils.py`, `Utils.py`
- Configuration parameters file: `params.ini`


## Prerequisites

To perform test on saved models, you need the following:

- TensorFlow
- Numpy
- Scipy
- configargparse
- Matplotlib

## Usage Example

To perform simple test, open `Test_Dehaze.py` in you IDE and run the script. Alternatively, from the `\Code` folder in the main repository, enter the following in the command:

`python Test_Dehaze.py`


## Saved Models

The model parameters used in our papers are located in the `\Results` folder, under each application. 

Some applications, for example single image dehazing contain several checkpoints: 27,30,33. For mild haze use 27 and for heavier amounts of haze use 30 and 33. 

When new training will be performed, other sub folders in `\Results` will be created with updated timestamps.

#### Single Image Dehazing: 

The datasets we use are all linked below:
HSTS, SOTS indoor and SOTS outdoor can be found in the following [link](https://sites.google.com/view/reside-dehaze-datasets/reside-v0). 

## Directory structure

    .
	├─ README.md
	├─ Code
	├─ params.ini (Configuration parameters file) 
	│   └── report.pdf
	├─ Datasets (source images)
	│   └──hazy
	│       └── ... (input images)
	├─ Results (the results)
    │   ├─ checkpoints (saved models)
    │   └── Qual_output (output images)
    │       └── paper ...
    │       └── our ...
	└─ Code (the python source code)
        ├── Test_Dehaze.py (dehazing using the dark channel prior, generate the results for the report)
        ├── Models.py (Neural network model we use throughout all experiments)
        ├── Dehazing_Loss.py (Implementations of Dehazing energy function)
        ├── our_dehaze.py (Implementations of Our Contribution to the Dehazing Algorithm)
        ├── Utils.py (utilities)
        └── train_utils.py (utilities)
