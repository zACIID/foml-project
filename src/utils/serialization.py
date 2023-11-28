import os
from typing import Type, TypeVar, Callable

import joblib
from loguru import logger


# NOTE1: scikit-learn suggests using joblib over pickle
#   https://scikit-learn.org/stable/model_persistence.html
# This is because joblib seems to be generally better for numpy
#   stuff, because of specific optimizations and better compression,
#   meaning that objects (i.e. sklearn models) with internal
#   numpy representations are pickled more efficiently
# NOTE2: But Pickle seems to be faster with sklearn models?
#   https://mljar.com/blog/save-load-scikit-learn-model/


def serialize(obj: object, filepath: str):
    """
    Serializes the provided object

    :param obj: object to serialize
    :param filepath: path to store the serialized object at
    """

    joblib.dump(obj, filename=filepath)


_T = TypeVar("_T")


def deserialize(type_: Type[_T], filepath: str | os.PathLike) -> _T:
    """
    Deserializes the provided file and casts the retrieved object to the provided type

    :param type_: type to cast the deserialized object to
    :param filepath: file to deserialize
    :return: deserialized object of the provided type
    """

    return joblib.load(filepath)


def deserialize_or_save_object(
        type_: Type[_T],
        filepath: str | os.PathLike,
        builder: Callable[[], _T],
        overwrite: bool = False
) -> _T:
    """
    Retrieves an existing object of the provided type, at the provided path,
    or creates a new one from scratch.

    :param type_: type of the object to retrieve
    :param filepath: path to serialization file containing the object to deserialize
    :param builder: function that creates a new abject from scratch, used for serialization
    :param overwrite: if True, ignores any existing serialized classifier and creates a new one
    :return: object of the provided type
    """

    if os.path.exists(filepath) and not overwrite:
        logger.info(f"Loading object from {filepath}...")
        return deserialize(type_=type_, filepath=filepath)
    else:
        logger.info(f"Serializing (saving) object at {filepath}...")
        new_object = builder()
        serialize(new_object, filepath)

        return new_object
