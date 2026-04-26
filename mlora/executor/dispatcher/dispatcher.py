import math
from typing import Any, Callable, Dict, List, Tuple

from mlora.config.dispatcher import DispatcherConfig
from mlora.config.task import TaskConfig
from mlora.executor.task import TASK_CLASS, Task
from mlora.model.args import Masks, MLoRAData, MLoRADataConfig, Tokens


class DispatcherEvent:
    callback_list_: List[Callable]

    def __init__(self):
        self.callback_list_ = []

    def register(self, func: Callable):
        self.callback_list_.append(func)

    def notify(self, task: Task) -> None:
        for func in self.callback_list_:
            func(task)


class Dispatcher:
    name_: str

    ready_: List[Task]
    running_: List[Task]

    init_event_: DispatcherEvent
    running_event_: DispatcherEvent
    ready_event_: DispatcherEvent
    done_event_: DispatcherEvent
    step_event_: DispatcherEvent
    terminate_event_: DispatcherEvent

    concurrency_num_: int = 2

    def __init__(self, config: DispatcherConfig) -> None:
        self.name_ = config.name_
        self.concurrency_num_ = config.concurrency_num_

        self.ready_ = []
        self.running_ = []

        self.init_event_ = DispatcherEvent()
        self.running_event_ = DispatcherEvent()
        self.ready_event_ = DispatcherEvent()
        self.done_event_ = DispatcherEvent()
        self.step_event_ = DispatcherEvent()
        self.terminate_event_ = DispatcherEvent()

    def info(self) -> Dict[str, Any]:
        return {"name": self.name_, "concurrency_num": self.concurrency_num_}

    def __task_trace_info(self, task: Task) -> Dict[str, Any]:
        adapter = getattr(task.config_, "adapter_", None)
        adapters = []
        if adapter is not None:
            adapters.append(adapter)

        reference = getattr(task.config_, "reference_", None)
        if reference is not None:
            adapters.append(reference)

        return {
            "name": task.task_name(),
            "type": task.task_type(),
            "adapters": [
                {
                    "name": getattr(item, "name_", ""),
                    "type": getattr(item, "type_", ""),
                    "rank": getattr(item, "r_", None),
                    "target_modules": [
                        key
                        for key, value in getattr(item, "target_", {}).items()
                        if value
                    ],
                }
                for item in adapters
            ],
            "mini_batch_size": getattr(task.config_, "mini_batch_size_", None),
            "batch_size": getattr(task.config_, "batch_size_", None),
            "now_step": getattr(task, "now_step_", None),
            "now_epoch": getattr(task, "now_epoch_", None),
        }

    def trace_state(self, data: MLoRAData | None = None) -> Dict[str, Any]:
        ret: Dict[str, Any] = {
            "dispatcher": self.info(),
            "ready_count": len(self.ready_),
            "running_count": len(self.running_),
            "ready_tasks": [self.__task_trace_info(task) for task in self.ready_],
            "running_tasks": [self.__task_trace_info(task) for task in self.running_],
        }

        if data is None:
            return ret

        token_lengths = [len(tokens) for tokens in data.batch_tokens_]
        useful_token_counts = [
            sum(1 for mask in masks if not mask) for masks in data.batch_mask_
        ]
        padded_tokens = sum(token_lengths)
        useful_tokens = sum(useful_token_counts)

        ret["batch"] = {
            "combined_batch_size": data.batch_size(),
            "token_len": data.token_len(),
            "padded_tokens": padded_tokens,
            "useful_tokens": useful_tokens,
            "padding_waste": (
                0.0
                if padded_tokens == 0
                else 1.0 - (float(useful_tokens) / float(padded_tokens))
            ),
            "data_configs": [
                {
                    "task_name": item.task_name_,
                    "adapter_name": item.adapter_name_,
                    "adapter_type": item.adapter_type_,
                    "batch_start_idx": item.batch_start_idx_,
                    "batch_end_idx": item.batch_end_idx_,
                    "row_count": item.batch_end_idx_ - item.batch_start_idx_,
                }
                for item in data.data_config_
            ],
        }

        return ret

    def observe_iteration(
        self, data: MLoRAData, memory_bytes: Dict[str, int | None]
    ) -> None:
        pass

    def register_hook(self, name: str, cb: Callable) -> None:
        event_map = {
            "init": self.init_event_,
            "running": self.running_event_,
            "ready": self.ready_event_,
            "done": self.done_event_,
            "step": self.step_event_,
            "terminate": self.terminate_event_,
        }

        assert name in event_map

        event_map[name].register(cb)

    def add_task(self, config: TaskConfig, llm_name: str):
        assert config.type_ in TASK_CLASS
        task = TASK_CLASS[config.type_](config, llm_name)
        self.init_event_.notify(task)
        self.ready_.append(task)

    def notify_terminate_task(self, task_name: str):
        for task in [*self.running_, *self.ready_]:
            if task.task_name() != task_name:
                continue
            task.notify_terminate()

    def is_done(self) -> bool:
        return len(self.running_) == 0 and len(self.ready_) == 0

    def _dispatch_task_in(self):
        # ready task to terminate
        terminate_task = [task for task in self.ready_ if task.is_terminate()]
        self.ready_ = [task for task in self.ready_ if not task.is_terminate()]

        for task in terminate_task:
            self.terminate_event_.notify(task)

        # ready task to running task
        assert len(self.running_) <= self.concurrency_num_
        if len(self.running_) == self.concurrency_num_:
            return

        while len(self.running_) < self.concurrency_num_ and len(self.ready_) > 0:
            task = self.ready_.pop(0)
            self.running_.append(task)
            self.running_event_.notify(task)

    def _dispatch_task_out(self):
        # running task to terminate
        terminate_task = [task for task in self.running_ if task.is_terminate()]
        self.running_ = [task for task in self.running_ if not task.is_terminate()]
        for task in terminate_task:
            self.terminate_event_.notify(task)

        # running task to ready
        done_task = [task for task in self.running_ if task.is_done()]
        self.running_ = [task for task in self.running_ if not task.is_done()]
        for task in done_task:
            self.done_event_.notify(task)

    def _align_batch_tokens(
        self, batch_tokens: List[Tokens], configs: List[MLoRADataConfig]
    ) -> Tuple[List[Tokens], List[Masks]]:
        max_seq_len = max(map(lambda x: len(x), batch_tokens))
        max_seq_len = math.ceil(max_seq_len / 8) * 8

        batch_masks: List[Masks] = []

        for data_config in configs:
            s_idx = data_config.batch_start_idx_
            e_idx = data_config.batch_end_idx_
            batch_tokens[s_idx:e_idx], masks = data_config.expand_fn_(
                batch_tokens[s_idx:e_idx], max_seq_len
            )
            batch_masks.extend(masks)

        return batch_tokens, batch_masks

    def data(self) -> MLoRAData | None:
        self._dispatch_task_in()

        batch_tokens: List[Tokens] = []
        batch_masks: List[Masks] = []
        data_configs: List[MLoRADataConfig] = []

        # get all train data
        start_idx: int = 0
        for task in self.running_:
            data, data_config = task.data(start_idx)
            data_configs.extend(data_config)
            batch_tokens.extend(data)
            start_idx = start_idx + len(data)

        # post process this batch data
        batch_tokens, batch_masks = self._align_batch_tokens(batch_tokens, data_configs)

        return MLoRAData(
            batch_tokens=batch_tokens, batch_mask=batch_masks, data_config=data_configs
        )

    def step(self):
        for _, task in enumerate(self.running_):
            task.step()
            self.step_event_.notify(task)

        self._dispatch_task_out()
