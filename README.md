# DeepRank


### Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Tutorial](#Tutorial)
- [License](./LICENSE)
- [Issues & Contributing](#Issues-and-Contributing)

## Overview

DeepRank-mut is a general, configurable deep learning framework for predicting pathogenicity of missense variants using 3D convolutional neural networks (CNNs).

DeepRank-mut contains useful APIs for pre-processing protein structural data, computing features for atoms/residues surrounding the missense variant,
as well as training and testing CNN models.

#### Features:

- Predefined atom-level and residue-level feature types
   - *e.g. atomic density, vdw energy, residue contacts, PSSM, etc.*
- Flexible definition of new features
- 3D grid feature mapping
- Efficient data storage in HDF5 format

## Installation

DeepRank requires a Python version 3.7 or 3.8 on Linux and MacOS.

#### Development Version

You can also install the under development source code from the branch `development`

- Clone the repository `git clone --branch development https://github.com/DeepRank/DeepRank-mut.git`
- Go there             `cd deeprank-mut`
- Install the package  `pip install -e ./`

To check if installation is successful, you can run a test
- Run the test suite         `pytest`


## Tutorial

We give here the tutorial like introduction to the DeepRank machinery. We quickly illsutrate here the three main steps of Deeprank-mut:

-   the generation of the data
-   training a model from the data
-   using the model to predict unseen data

### A . Generate the data set (using MPI)

For data generation, PDB files must be stored locally. The user can optionally provide PSSM files, mapped to those PDB files.
Together with the PDB files, variant data must be available. Here's an example of a table, containing variant data:


| PDB ID | CHAIN | RESIDUE | WILDTYPE | VARIANT | CLASS  |
|--------|-------|---------|----------|---------|--------|
| 101m   | A     | 25      | GLY      | ALA     | BENIGN |
| 101m   | A     | 21      | VAL      | SER     | BENIGN |


The `CLASS` column would typically be omitted from unclassified variants.
In this example, we store the table in CSV format.

All the features/targets and mapped features onto grid points will be auomatically calculated and store in a HDF5 file.

```python
import pandas

from deeprank.models.variant import *
from deeprank.generate import *
from mpi4py import MPI
from deeprank.domain.amino_acid import amino_acids_by_code
from deeprank.models.environment import Environment

comm = MPI.COMM_WORLD

# let's put this sample script in the test folder, so the working path will be ./test/
# name of the hdf5 to generate
hdf5_path = 'train_data.hdf5'

environment = Environment(pdb_root="path/to/pdb/files/",
                          pssm_root="path/to/pssm/files/")
table = pandas.read_csv("table.csv")

variants = []
for _, row in table.iterrows():
    wildtype_amino_acid = amino_acids_by_code[row["WILDTYPE"]]
    variant_amino_acid = amino_acids_by_code[row["VARIANT"]]

    variant = PdbVariantSelection(
        row["PDB ID"], row["CHAIN"], row["RESIDUE"],
        wildtype_amino_acid=wildtype_amino_acid,
        variant_amino_acid=variant_amino_acid,
        variant_class=VariantClass.parse(row["CLASS"]),  # variant class is optional
    )

    variants.append(variant)


# initialize the database
database = DataGenerator(environment, variants=variants,
    data_augmentation=10,
    compute_targets=['deeprank.targets.variant_class'],  # omit this, if the variant class is not provided
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

# preprocessed input data
hdf5_path = 'train_data.hdf5'

# output directory
out = './my_deeplearning_train/'

# declare the dataset instance
data_set = DataSet(
    hdf5_path,
    grid_info={
        'number_of_points': (10, 10, 10),
        'resolution': (10, 10, 10)
    },
    select_feature='all',
    select_target='class',
)


# create the network
model = NeuralNet(data_set,cnn_reg,model_type='3d',task='reg',
                  cuda=False)

# change the optimizer (optional)
model.optimizer = optim.SGD(model.net.parameters(),
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=0.005)

# start the training, this will generate a model file named `best_valid_model.pth.tar`.
model.train(nepoch=50, divide_trainset=0.8, train_batch_size=5, num_workers=0)
```

In the first part of the script we create a Torch database from the HDF5 file. We can specify one or several HDF5 files and even select some conformations using the `dict_filter` argument. Other options of `DataSet` can be used to specify the features/targets the normalization, etc ...

We then create a `NeuralNet` instance that takes the dataset as input argument. Several options are available to specify the task to do, the GPU use, etc ... We then have simply to train the model. Simple !

### C. Predicting Unclassified Data

Prediction is almost similar to training, apart from the fact that there are no class labels.

```
from deeprank.learn import *
from deeprank.learn.model3d import cnn_reg
from deeprank.models.metrics import OutputExporter
import torch.optim as optim
import numpy as np

# preprocessed input data
hdf5_path = 'unseen_data.hdf5'

# output directory
out = './my_deeplearning_train/'

# declare the dataset instance
data_set = DataSet(
    hdf5_path,
    grid_info={
        'number_of_points': (10, 10, 10),
        'resolution': (10, 10, 10)
    },
    select_feature='all',
    select_target='class',
)

# create the network
model = NeuralNet(data_set, cnn_reg, model_type='3d', task='reg',
                  pretrained_model="best_valid_model.pth.tar",
                  metrics_exporters=[OutputExporter(out)],
                  cuda=False)

# change the optimizer (optional)
model.optimizer = optim.SGD(model.net.parameters(),
                            lr=0.001,
                            momentum=0.9,
                            weight_decay=0.005)

# do the prediction
model.test()
```

## Issues and Contributing

If you have questions or find a bug, please report the issue in the [Github issue channel](https://github.com/DeepRank/deeprank-mut/issues).

If you want to change or further develop DeepRank-mut code, please check the [Developer Guideline](./developer_guideline.md) to see how to conduct further development.
