import torch 

def get_scheduler(
    optimizer,
    scheduler_option: str,
    scheduler_params: dict,
):
    if scheduler_option == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_option == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
    elif scheduler_option == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_option == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_option == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Invalid scheduler option: {scheduler_option}")
    return scheduler