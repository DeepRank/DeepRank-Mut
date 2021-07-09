import os
import datetime
import sys
sys.path.insert(0, "./")

import torch.distributed as dist

from deeprank.models.variant import *
from deeprank.generate import *
from deeprank.learn import DataSet
from deeprank.learn.NeuralNetDDP import NeuralNetDDP
from deeprank.learn.model3d import cnn_class


def spmd_main():
    env_dict ={
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }

    #the address of env_dict["MASTER_ADDR"] has to be the ip of the main server
    env_dict["MASTER_ADDR"] = "127.0.0.1" 
    env_dict["MASTER_PORT"] =1080
    rank = int(env_dict["RANK"])
    world_size = int(env_dict["WORLD_SIZE"])

    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
    dist.init_process_group(backend="nccl",init_method=f'tcp://{env_dict["MASTER_ADDR"]}:{env_dict["MASTER_PORT"]}',rank=rank, world_size=world_size, timeout=datetime.timedelta(0,400))

    train(rank)


def train(rank):
    print(f"Running basic DDP example on rank {rank}.")

    database = 'test/bioprodict_dataset/bioprodict-variants.hdf5'
    data_set = DataSet(database,
                        grid_info={
                            'number_of_points': (10, 10, 10),
                            'resolution': (3, 3, 3)},
                        select_feature='all',
                        select_target='class')

    model = NeuralNetDDP(data_set, cnn_class, model_type='3d', task='class', cuda=True, plot=True, outdir='test/output', save_classmetrics=True)
    model.train(nepoch=2, divide_trainset=0.8, train_batch_size=5, num_workers=2, rank=rank)

if __name__ == "__main__":
    spmd_main()
