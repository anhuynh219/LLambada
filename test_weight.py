import torch

state_dict = torch.load("/workspace/continuation/test/open-singsong/results/llambada_semantic/llambada.transformer.77000.pt")
print(state_dict.keys())
print(state_dict["transformer.layers.0.0.to_out.0.weight"].shape)