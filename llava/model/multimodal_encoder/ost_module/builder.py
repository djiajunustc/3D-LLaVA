from llava.pc_utils.registry import Registry

MODELS = Registry("models")

def build_model(cfg):
    """Build models."""
    return MODELS.build(cfg)


LOSSES = Registry("losses")

def build_criteria(cfg):
    return LOSSES.build(cfg)

TASK_UTILS = Registry("task_utils")