import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg
from flearn.trainer.ifca import IFCA
from flearn.trainer.fesem import FeSEM
from flearn.trainer.fedgroup import FedGroup

# Modify these settings as you need
def main_fedavg():
    config = TrainConfig('mnist', 'mlp', 'fedavg')
    trainer = FedAvg(config)
    trainer.train()
    #trainer.train_locally()

def main_ifca():
    config = TrainConfig('femnist', 'mlp', 'ifca')
    config.trainer_config['dynamic'] = False # whether migrate clients
    config.trainer_config['num_rounds'] = 20
    trainer = IFCA(config)
    trainer.train()
    #trainer.train_locally()

def main_fesem():
    config = TrainConfig('mnist', 'mlp', 'fesem')
    config.trainer_config['dynamic'] = False
    trainer = FeSEM(config)
    trainer.train()
    #trainer.train_locally()

def main_flexcfl():
    config = TrainConfig('femnist', 'mlp', 'fedgroup')
    config.trainer_config['dynamic'] = True
    config.trainer_config['num_rounds'] = 30
    config.trainer_config['shift_type'] = "all"
    config.trainer_config['swap_p'] = 0.05
    trainer = FedGroup(config)
    trainer.train()


''' # Uncomment these codes as you need '''

# main_fedavg()
# main_flexcfl()
# main_fesem()
main_ifca()