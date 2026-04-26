import math
from typing import Any, Dict, List, Tuple, override

import torch

from mlora.config.dispatcher import DispatcherConfig
from mlora.executor.task import Task
from mlora.model.args import MLoRAData

from .dispatcher import Dispatcher


class ElasticDispatcher(Dispatcher):
    seq_bucket_size_: int
    max_padding_waste_: float
    fairness_wait_steps_: int
    memory_safety_ratio_: float
    estimated_base_memory_bytes_: float
    estimated_bytes_per_padded_token_: float
    ready_wait_rounds_: Dict[str, int]

    def __init__(self, config: DispatcherConfig) -> None:
        super().__init__(config)
        self.seq_bucket_size_ = config.seq_bucket_size_
        self.max_padding_waste_ = config.max_padding_waste_
        self.fairness_wait_steps_ = config.fairness_wait_steps_
        self.memory_safety_ratio_ = config.memory_safety_ratio_
        self.estimated_base_memory_bytes_ = config.estimated_base_memory_bytes_
        self.estimated_bytes_per_padded_token_ = (
            config.estimated_bytes_per_padded_token_
        )
        self.ready_wait_rounds_ = {}

    @override
    def info(self) -> Dict[str, Any]:
        ret = super().info()
        ret.update(
            {
                "seq_bucket_size": self.seq_bucket_size_,
                "max_padding_waste": self.max_padding_waste_,
                "fairness_wait_steps": self.fairness_wait_steps_,
                "memory_safety_ratio": self.memory_safety_ratio_,
                "estimated_base_memory_bytes": self.estimated_base_memory_bytes_,
                "estimated_bytes_per_padded_token": (
                    self.estimated_bytes_per_padded_token_
                ),
            }
        )
        return ret

    def _task_token_lengths(self, task: Task) -> List[int]:
        lengths = task.peek_next_token_lengths()
        if lengths is not None and len(lengths) > 0:
            return lengths

        cutoff_len = getattr(task.config_, "cutoff_len_", 1024)
        mini_batch_size = getattr(task.config_, "mini_batch_size_", 1)
        return [cutoff_len] * mini_batch_size

    def _group_padding_stats(self, tasks: List[Task]) -> Tuple[int, int, float]:
        token_lengths: List[int] = []
        for task in tasks:
            token_lengths.extend(self._task_token_lengths(task))

        if len(token_lengths) == 0:
            return 0, 0, 0.0

        aligned_len = math.ceil(max(token_lengths) / 8) * 8
        padded_tokens = aligned_len * len(token_lengths)
        useful_tokens = sum(token_lengths)
        padding_waste = 1.0 - (float(useful_tokens) / float(padded_tokens))

        return useful_tokens, padded_tokens, padding_waste

    def _memory_limit_bytes(self) -> int | None:
        if not torch.cuda.is_available():
            return None

        try:
            _, total_bytes = torch.cuda.mem_get_info()
        except Exception:
            return None

        return int(total_bytes * self.memory_safety_ratio_)

    def _memory_safe(self, tasks: List[Task]) -> bool:
        if self.estimated_bytes_per_padded_token_ <= 0:
            return True

        memory_limit = self._memory_limit_bytes()
        if memory_limit is None:
            return True

        _, padded_tokens, _ = self._group_padding_stats(tasks)
        estimated_memory = self.estimated_base_memory_bytes_ + (
            self.estimated_bytes_per_padded_token_ * padded_tokens
        )

        return estimated_memory <= memory_limit

    def _select_next_task(self) -> int | None:
        if len(self.ready_) == 0:
            return None

        if len(self.running_) == 0:
            return 0

        best_idx: int | None = None
        best_score: Tuple[float, int] | None = None

        for idx, task in enumerate(self.ready_):
            candidate_group = [*self.running_, task]
            _, _, padding_waste = self._group_padding_stats(candidate_group)
            if padding_waste > self.max_padding_waste_:
                continue
            if not self._memory_safe(candidate_group):
                continue

            # Prefer the group with the least padding waste, then preserve FIFO order.
            score = (padding_waste, idx)
            if best_score is None or score < best_score:
                best_idx = idx
                best_score = score

        if best_idx is not None:
            return best_idx

        if self.fairness_wait_steps_ <= 0:
            return None

        for idx, task in enumerate(self.ready_):
            wait_rounds = self.ready_wait_rounds_.get(task.task_name(), 0)
            if wait_rounds < self.fairness_wait_steps_:
                continue
            candidate_group = [*self.running_, task]
            if self._memory_safe(candidate_group):
                return idx

        return None

    @override
    def _dispatch_task_in(self):
        terminate_task = [task for task in self.ready_ if task.is_terminate()]
        self.ready_ = [task for task in self.ready_ if not task.is_terminate()]

        for task in terminate_task:
            self.ready_wait_rounds_.pop(task.task_name(), None)
            self.terminate_event_.notify(task)

        for task in self.ready_:
            task_name = task.task_name()
            self.ready_wait_rounds_[task_name] = (
                self.ready_wait_rounds_.get(task_name, 0) + 1
            )

        assert len(self.running_) <= self.concurrency_num_

        while len(self.running_) < self.concurrency_num_ and len(self.ready_) > 0:
            idx = self._select_next_task()
            if idx is None:
                break

            task = self.ready_.pop(idx)
            self.ready_wait_rounds_.pop(task.task_name(), None)
            self.running_.append(task)
            self.running_event_.notify(task)

    @override
    def observe_iteration(
        self, data: MLoRAData, memory_bytes: Dict[str, int | None]
    ) -> None:
        max_reserved = memory_bytes.get("max_reserved")
        if max_reserved is None or max_reserved <= self.estimated_base_memory_bytes_:
            return

        padded_tokens = sum(len(tokens) for tokens in data.batch_tokens_)
        if padded_tokens <= 0:
            return

        observed_bytes_per_token = (
            max_reserved - self.estimated_base_memory_bytes_
        ) / padded_tokens
        self.estimated_bytes_per_padded_token_ = max(
            self.estimated_bytes_per_padded_token_, observed_bytes_per_token
        )
