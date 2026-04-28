import json
import os


def assert_stage2_checkpoint(model_path: str, task_name: str = "box evaluation"):
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    training_stage = config.get("training_stage", 2)
    if training_stage == 1:
        raise RuntimeError(
            f"{task_name} requires a stage2 checkpoint, but {model_path} is marked as training_stage=1."
        )
