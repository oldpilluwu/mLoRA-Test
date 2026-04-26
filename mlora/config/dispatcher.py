from typing import Dict

from .config import DictConfig


class DispatcherConfig(DictConfig):
    name_: str
    concurrency_num_: int
    seq_bucket_size_: int
    max_padding_waste_: float
    fairness_wait_steps_: int
    memory_safety_ratio_: float
    estimated_base_memory_bytes_: float
    estimated_bytes_per_padded_token_: float

    __params_map: Dict[str, str] = {
        "name_": "name",
        "concurrency_num_": "concurrency_num",
    }

    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.init(self.__params_map, config)

        self.concurrency_num_ = int(self.concurrency_num_)
        self.seq_bucket_size_ = int(config.get("seq_bucket_size", 256))
        self.max_padding_waste_ = float(config.get("max_padding_waste", 0.20))
        self.fairness_wait_steps_ = int(config.get("fairness_wait_steps", 0))
        self.memory_safety_ratio_ = float(config.get("memory_safety_ratio", 0.90))
        self.estimated_base_memory_bytes_ = float(
            config.get("estimated_base_memory_bytes", 0.0)
        )
        self.estimated_bytes_per_padded_token_ = float(
            config.get("estimated_bytes_per_padded_token", 0.0)
        )
