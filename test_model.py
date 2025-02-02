from models.Llambada.builder import build_llambada_model
import json
import torch

# cfg = json.load(open("/workspace/singsong_upgrade/SG-SingSong-Internal/configs/model_config/sg_singsong_tiny.json"))

semantic_cfg = json.load(open("/workspace/singsong_upgrade/SG-SingSong-Internal/configs/model_config/llambada_tiny_cfg/semantic_stage.json"))
print(semantic_cfg)
coarse_cfg = json.load(open("/workspace/singsong_upgrade/SG-SingSong-Internal/configs/model_config/llambada_tiny_cfg/coarse_stage.json"))

model = build_llambada_model(
    semantic_cfg,
    coarse_cfg,
    semantic_weight = "/workspace/continuation/test/open-singsong/results/llambada_semantic/llambada.transformer.77000.pt",
    coarse_weight = "/workspace/singsong_upgrade/SG-SingSong-Internal/coarse.transformer.17400.pt",
    semantic_cross_entropy_loss_weights =[0.0, 0.0, 1.0],
    coarse_cross_entropy_loss_weights =[0.0, 0.0, 1.0]
)
