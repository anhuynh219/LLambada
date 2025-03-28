from torch.optim import AdamW, Adam, lr_scheduler

def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []
    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)
    return wd_params, no_wd_params

def get_optimizer(
    params,
    lr = 1e-4,
    wd = 1e-2,
    betas = (0.9, 0.99),
    eps = 1e-8,
    filter_by_requires_grad = False,
    group_wd_params = True,
    **kwargs
):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    if wd == 0:
        return Adam(params, lr = lr, betas = betas, eps = eps)

    if group_wd_params:
        wd_params, no_wd_params = separate_weight_decayable_params(params)

        params = [
            {'params': wd_params},
            {'params': no_wd_params, 'weight_decay': 0},
        ]

    return AdamW(params, lr = lr, weight_decay = wd, betas = betas, eps = eps)

def get_linear_scheduler(
    optimizer,
    total_iters=10000,
    start_factor=1e-6,
):
    return lr_scheduler.LinearLR(optimizer=optimizer, start_factor=start_factor, end_factor=1., total_iters=total_iters, verbose = True)
    # return lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_iters, verbose = True)