from data import get_MSSEG
from handlers import Messidor_Handler, MSSEG_Handler_2d
from nets import Net, MSSEG_model
from query_strategies import RandomSampling, EntropySampling, EntropySamplingDropout, BALDDropout, AdversarialAttack,AdversarialAttack_efficient, KCenterGreedy
from seed import setup_seed

# important settings
setup_seed(42)
params = {
    'MSSEG':
        {'n_epoch': 200,
         'train_args': {'batch_size': 32,'shuffle':True, 'num_workers': 2,'drop_last':False},
         'val_args': {'batch_size': 512,'shuffle':False, 'num_workers': 2,'drop_last':False},
         'test_args': {'batch_size': 512,'shuffle':False, 'num_workers': 2,'drop_last':False},
         'optimizer_args': {'lr': 0.001}},  
}


# Get data loader
def get_handler(name):
    if name == 'Messidor':
        return Messidor_Handler
    elif name == 'MSSEG':
        return MSSEG_Handler_2d


# Get dataset
def get_dataset(name,supervised):
    if name == 'MSSEG':
        if supervised == True:
            return get_MSSEG(get_handler(name),supervised = True)
        else:
            return get_MSSEG(get_handler(name))
    else:
        raise NotImplementedError


# define network for specific dataset
def get_net(name, device, init=False):
    if name == 'MSSEG':
        return Net(MSSEG_model, params[name], device)

    else:
        raise NotImplementedError

# get strategies
def get_strategy(name):
    if name == "RandomSampling":
        return RandomSampling
    elif name == "EntropySampling":
        return EntropySampling
    elif name == "EntropySamplingDropout":
        return EntropySamplingDropout
    elif name == "BALDDropout":
        return BALDDropout
    elif name == "AdversarialAttack":
        return AdversarialAttack
    elif name == "KCenterGreedy":
        return KCenterGreedy
    else:
        raise NotImplementedError