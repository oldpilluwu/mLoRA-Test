from typing import Dict, Type

from .backend_dispatcher import BackendDispatcher
from .dispatcher import Dispatcher
from .elastic_dispatcher import ElasticDispatcher
from .pipe_dispatcher import PipeDispatcher

DISPATCHER_CLASS: Dict[str, Type[Dispatcher]] = {
    "default": Dispatcher,
    "backend": BackendDispatcher,
    "elastic": ElasticDispatcher,
    "pipe": PipeDispatcher,
}

__all__ = [
    "Dispatcher",
    "BackendDispatcher",
    "ElasticDispatcher",
    "PipeDispatcher",
    "DISPATCHER_CLASS",
]
