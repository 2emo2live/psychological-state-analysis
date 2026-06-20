from pathlib import Path

import hydra
import tensorrt as trt
from omegaconf import DictConfig


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def export_to_tensorrt(cfg: DictConfig):
    onnx_path = Path(cfg.export.onnx_output_path)
    plan_path = Path(cfg.export.tensorrt_output_path)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)

    network = builder.create_network()

    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 ГБ

    max_len = cfg.data.max_len
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, max_len), (8, max_len), (16, max_len))
    profile.set_shape("attention_mask", (1, max_len), (8, max_len), (16, max_len))
    config.add_optimization_profile(profile)

    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    with open(plan_path, "wb") as f:
        f.write(engine)
    print(f"TensorRT engine saved to {plan_path}")
