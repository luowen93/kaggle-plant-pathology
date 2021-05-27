import torch.optim as optim
import torch.nn.functional as F

def define_optimizer(parameters):
    optimizer = optim.SGD(parameters,lr=1e-3)
    return optimizer

def define_scheduler():
    # scheduler = optim.lr_scheduler
    return None

# Return the loss function
def define_loss():
    return F.nll_loss


