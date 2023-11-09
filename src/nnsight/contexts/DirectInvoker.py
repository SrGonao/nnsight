from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

from .Invoker import Invoker
from .Runner import Runner

if TYPE_CHECKING:
    from ..models.AbstractModel import AbstractModel


class DirectInvoker(Runner, Invoker):
    def __init__(
        self, model: "AbstractModel", *args, fwd_args: Dict[str, Any] = None, **kwargs
    ):
        if fwd_args is None:
            fwd_args = dict()

        Runner.__init__(self, model, **fwd_args)

        Invoker.__init__(self, self, *args, **kwargs)

    def __enter__(self) -> DirectInvoker:

        Invoker.__enter__(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        Invoker.__exit__(self, exc_type, exc_val, exc_tb)

        Runner.__exit__(self, exc_type, exc_val, exc_tb)
