# DeepRank
[![Documentation Status](https://readthedocs.org/projects/deeprank/badge/?version=latest)](http://deeprank.readthedocs.io/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3735042.svg)](https://doi.org/10.5281/zenodo.3735042)
![Build](https://github.com/DeepRank/deeprank/workflows/Build/badge.svg)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/4254dd4798bf4cfa9f8f6fe0079de144)](https://www.codacy.com/gh/DeepRank/deeprank/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=DeepRank/deeprank&amp;utm_campaign=Badge_Grade)
[![Coverage Status](https://coveralls.io/repos/github/DeepRank/deeprank/badge.svg?branch=master)](https://coveralls.io/github/DeepRank/deeprank?branch=master)


### Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Tutorial](#Tutorial)
- [Documentation](https://deeprank.readthedocs.io/)
- [License](./LICENSE)
- [Issues & Contributing](#Issues-and-Contributing)

## Overview
![alt-text](./pics/deeprankv.2.0.png)

DeepRank is a general, configurable deep learning framework for predicting pathogenicity of missense variants using 3D convolutional neural networks (CNNs).

DeepRank contains useful APIs for pre-processing protein structural data, computing features for atoms/residues surrounding the missense variant, as well as training and testing CNN models.

#### Features:

- Predefined atom-level and residue-level feature types
   - *e.g. atomic density, vdw energy, residue contacts, PSSM, etc.*
- Flexible definition of new features
- 3D grid feature mapping
- Efficient data storage in HDF5 format
- Support both classification and regression (based on PyTorch)

## Installation

DeepRank requires a Python version 3.7 or 3.8 on Linux and MacOS.

#### Stable Release

DeepRank is available in stable releases on [PyPI](https://pypi.org/project/deeprank/):
-  Install the module `pip install deeprank`

#### Development Version

You can also install the under development source code from the branch `development`

- Clone the repository `git clone --branch development https://github.com/DeepRank/DeepRank_VariantEffect-Pred.git`
- Go there             `cd deeprank`
- Install the package  `pip install -e ./`

To check if installation is successful, you can run a test
- Go into the test directory `cd test`
- Run the test suite         `pytest`


## Tutorial

We give here the tutorial like introduction to the DeepRank machinery. More informatoin can be found in the documentation <http://deeprank.readthedocs.io/en/latest/>.  We quickly illsutrate here the two main steps of Deeprank:

-   the generation of the data
-   running deep leaning experiments.

### A . Generate the data set (using MPI)

The generation of the data require only require PDBs files of decoys and their native and the PSSM if needed. All the features/targets and mapped features onto grid points will be auomatically calculated and store in a HDF5 file.

```python
from deeprank.models.variant import *
from deeprank.generate import *
from mpi4py import MPI

comm = MPI.COMM_WORLD

# let's put this sample script in the test folder, so the working path will be deeprank/test/
# name of the hdf5 to generate
hdf5_path = '1ak4.hdf5'

# where to find the native conformation
pdb_path = "test/1AK4/native/1AK4.pdb"

# The variant data
chain_id = "C"
pdb_resnum = 10
amino_acid = "T"

# where to find the pssm
pssm_paths = {"C": "test/1AK4/pssm_new/1AK4.C.pssm",
              "D": "test/1AK4/pssm_new/1AK4.D.pssm"}

variant_class = VariantClass.BENIGN

variant = PdbVariantSelection(pdb_path, chain_id, pdb_resnum, amino_acid, pssm_paths, variant_class)


# initialize the database
database = DataGenerator(variants=[variant],
    data_augmentation=10,
    compute_targets=['deeprank.targets.variant_class'],
    compute_features=[
        'deeprank.features.atomic_contacts',
        'deeprank.features.neighbour_profile',
        'deeprank.features.accessibility'],
    hdf5=hdf5_path,
    mpi_comm=comm)


# create the database
# compute features/targets for all complexes
database.create_database(prog_bar=True)


# define the 3D grid
 grid_info = {
   'number_of_points': [30,30,30],
   'resolution': [1.,1.,1.],
   'atomic_densities': {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8},
 }

# Map the features
database.map_features(grid_info,try_sparse=True, time=False, prog_bar=True)

```

This script can be exectuted using for example 4 MPI processes with the command:

```
    NP=4
    mpiexec -n $NP python generate.py
```

In  the first part of the script we define the path where to find the PDBs of the native structure of interest. We then specify position of variant in the structure and the variant amino acid, to enable the code to compute features of the amino acids surrounding the variant. 
We then initialize the `DataGenerator` object. This object (defined in `deeprank/generate/DataGenerator.py`) needs a few input parameters:

-   variants: a selection of variant objects that make up the dataset
-   compute_targets: the module used to compute the target: 'Benign' or 'Pathogenic'
-   compute_features: list of modules used to compute the features
-   hdf5: Name of the HDF5 file to store the data set

We then create the data base with the command `database.create_database()`. This function automatically creates an HDF5 files where the pdb has its own group. In each group we can find the pdb, its calculated features and the target. We can now mapped the features to a grid. This is done via the command `database.map_features()`. As you can see this method requires a dictionary as input. The dictionary contains the instruction to map the data.

-   number_of_points: the number of points in each direction
-   resolution: the resolution in Angs
-   atomic_densities: {'atom_name': vvdw_radius} the atomic densities required

The atomic densities are mapped following the [protein-ligand paper](https://arxiv.org/abs/1612.02751). The other features are mapped to the grid points using a Gaussian function (other modes are possible but somehow hard coded)

#### Visualization of the mapped features

To explore the HDf5 file and vizualize the features you can use the dedicated browser <https://github.com/DeepRank/DeepXplorer>. This tool allows to dig through the hdf5 file and to directly generate the files required to vizualize the features in VMD or PyMol. An iPython comsole is also embedded to analyze the feature values, plot them etc ....

### B . Deep Learning

The HDF5 files generated above can be used as input for deep learning experiments. You can take a look at the file `test/test_learn.py` for some examples. We give here a quick overview of the process.

```python
from deeprank.learn import *
from deeprank.learn.model3d import cnn_reg
import torch.optim as optim
import numpy as np

# input database
hdf5_path = '1ak4.hdf5'

# output directory
out = './my_DL_test/'

# declare the dataset instance
data_set = DataSet(database,
            grid_info={
                'number_of_points': (10, 10, 10),
                'resolution': (3, 3, 3)},
            select_feature='all',
            select_target='class')


# create the network
model = NeuralNet(data_set,cnn_reg,model_type='3d',task='reg',
                  cuda=False,plot=True,outdir=out)

# change the optimizer (optional)
model.optimizer = optim.SGD(model.net.parameters(),
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=0.005)

# start the training
model.train(nepoch=50, divide_trainset=0.8, train_batch_size=5, num_workers=0)
```

In the first part of the script we create a Torch database from the HDF5 file. We can specify one or several HDF5 files and even select some conformations using the `dict_filter` argument. Other options of `DataSet` can be used to specify the features/targets the normalization, etc ...

We then create a `NeuralNet` instance that takes the dataset as input argument. Several options are available to specify the task to do, the GPU use, etc ... We then have simply to train the model. Simple !

## Issues and Contributing

If you have questions or find a bug, please report the issue in the [Github issue channel](https://github.com/DeepRank/deeprank/issues).

If you want to change or further develop DeepRank code, please check the [Developer Guideline](./developer_guideline.md) to see how to conduct further development.
