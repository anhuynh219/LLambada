import torch 

def get_optimizer(
    params,
    optmizer_option: str,
    learning_rate,
):
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    return optimizer