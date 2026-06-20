from pathlib import Path

import hydra
import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import DictConfig

from psychology_state_analyzer.models.main_model import PsychologicalStateModel


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def export_to_onnx(cfg: DictConfig):
    best_model_path = cfg.export.checkpoint_path
    device = "cpu"

    model = PsychologicalStateModel.load_from_checkpoint(
        best_model_path,
        root_path=cfg.data.root_path,
        tokenizer_name=cfg.model.backbone,
        batch_size=cfg.data.batch_size,
        max_len=cfg.data.max_len,
    )
    model.eval()
    model.to(device)

    dummy_input_ids = torch.randint(
        0, 100, (cfg.data.batch_size, cfg.data.max_len), dtype=torch.long
    )
    dummy_attention_mask = torch.ones(
        (cfg.data.batch_size, cfg.data.max_len), dtype=torch.long
    )

    onnx_path = Path(cfg.export.output_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            cfg.export.output_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "seq_len"},
                "attention_mask": {0: "batch_size", 1: "seq_len"},
                "logits": {0: "batch_size"},
            },
            opset_version=cfg.export.opset_version,
            do_constant_folding=cfg.export.do_constant_folding,
        )
        torch_output = (
            model(dummy_input_ids, dummy_attention_mask).detach().cpu().numpy()
        )

    onnx_model = onnx.load(cfg.export.output_path)
    onnx.checker.check_model(onnx_model)

    if cfg.export.verify:
        ort_session = ort.InferenceSession(
            cfg.export.output_path,
            providers=["CPUExecutionProvider"],
        )
        ort_inputs = {
            "input_ids": dummy_input_ids.numpy(),
            "attention_mask": dummy_attention_mask.numpy(),
        }
        ort_output = ort_session.run(["logits"], ort_inputs)[0]

        max_abs_diff = np.max(np.abs(torch_output - ort_output))
        print(f"Максимальная абсолютная разница: {max_abs_diff:.6f}")

    print(f"ONNX модель сохранена в {cfg.export.output_path}")


if __name__ == "__main__":
    export_to_onnx()
