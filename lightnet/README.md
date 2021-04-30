# Pascal VOC
[![Python][python-badge]][python-url]
[![PyTorch][pytorch-badge]][pytorch-url]
[![Lightnet][lightnet-badge]][lightnet-url]
[![Brambox][brambox-badge]][brambox-url]

For more information about how to use this repository,
check out our [tutorial](https://eavise.gitlab.io/lightnet/notes/03-A-pascal_voc.html).


## Dependencies
This project requires a few python packages in order to work.  
First install [PyTorch](https://pytorch.org/get-started/locally) according to your machine specifications.
For this project we used PyTorch v1.7.0, but any higher version should work.  
Once PyTorch is installed, you can install the other dependencies by running the following command:
```bash
pip install -r requirements.txt
```


## Data setup
The scripts in this repository expect to find the Pascal VOC data in `./data/VOCdevkit/`
and the annotations in a [brambox pandas format](http://eavise.gitlab.io/brambox/api/generated/brambox.io.parser.box.PandasParser.html) in `./data/trainvaltest/`.
Once you have downloaded the Pascal VOC data, you can run `./bin/labels.py` to generate the brambox annotations.


## Running scripts
Once the data and annotations are in the `./data/` folder, you can use the scripts in the `./bin/` folder to train, prune, test and benchmark your models.
You can run `./bin/<script>.py --help` for more information about the arguments required to run a certain script.

Note that all scripts require a *config* file to be passed.
These can be found in the `./cfg/` directory and are split into training and pruning config files.


## Makefile
If you have a docker image with all the necessary packages installed, you can run all the scripts with the provided `Makefile`.
Each script has its associated Make target, which requires specific arguments.

This makefile was build to more easily run multiple training/pruning pipelines and is thus only tested on one machine.  
It is provided as is and without any guarantees of working!


[python-badge]: https://img.shields.io/badge/python-3.6+-99CCFF.svg
[python-url]: https://python.org
[pytorch-badge]: https://img.shields.io/badge/PyTorch-1.7.0-F05732.svg
[pytorch-url]: https://pytorch.org
[lightnet-badge]: https://img.shields.io/badge/Lightnet-2.0.0-00BFD8.svg
[lightnet-url]: https://eavise.gitlab.io/lightnet
[brambox-badge]: https://img.shields.io/badge/Brambox-3.2.0-007EC6.svg
[brambox-url]: https://eavise.gitlab.io/brambox
