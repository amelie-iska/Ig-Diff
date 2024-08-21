# IgDiff
This repository contains code for the paper "De novo antibody design with SE(3) diffusion". It is adapted from the original [FrameDiff code](https://github.com/jasonkyuyim/se3_diffusion).

To download the model weights and dataset visit [the HuggingFace repo](https://huggingface.co/AmelieSchreiber/IgDiff/tree/main), or run 
```bash
git clone https://huggingface.co/AmelieSchreiber/IgDiff
```
Then move the weights into the weights folder, and the samples into the igdiff_samples folder. 

## Installation
To install the package, run
```
pip install -e .
```

If needed, you can create a conda environment with all the required dependencies using
```
conda env create -f se3.yml
```
## Usage
You can generate unconditioned antibodies with
```
python experiments/inference_se3_diffusion.py
```

which will create a folder in the `inference` directory with generated structures.
