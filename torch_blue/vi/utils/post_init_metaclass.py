from abc import ABCMeta
from typing import Any, Type, TypeVar

T = TypeVar("T")


class PostInitCallMeta(ABCMeta):
    r"""Metaclass calling ``__post_init__`` after ``__init__``."""

    def __call__(cls: Type[T], *args: Any, **kwargs: Any) -> T:
        r"""Run __call__ as usual, then call __post_init__."""
        class_object = ABCMeta.__call__(cls, *args, **kwargs)
        class_object.__post_init__()
        return class_object
