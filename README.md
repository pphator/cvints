[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://python.org)


# cvints

The library for solving frequently encountered problems in the field of computer vision


## Installation

### Development version

With [pip](https://pip.pypa.io/en/stable/)
```
pip3 install git+https://github.com/VasilyBoychuk/cvints.git
```

## Library structure

The library has several main entities:

- [Model](https://github.com/VasilyBoychuk/cvints/blob/master/cvints/model.py)
- [Dataset](https://github.com/VasilyBoychuk/cvints/blob/master/cvints/dataset.py)
- [Experiment](https://github.com/VasilyBoychuk/cvints/blob/master/cvints/experiment.py)

### Dataset

You can store information about train/validation/test dataset in the Dataset-like classes. 
To do this, specify the paths to images and annotation files. 

### Model 

You can define model using training/validation data, test data and results on test data.

### Experiment

You can think of the entire process of evaluating the model's performance with data as an experiment 
in which all actions with other entities take place


## Examples

Check dataset before start testing model with it ([rus](https://github.com/VasilyBoychuk/cvints/blob/master/examples/rus/Check%20dataset.ipynb)|eng)