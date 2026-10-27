from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Tensor(_message.Message):
    __slots__ = ("dtype", "shape", "data")
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    dtype: str
    shape: _containers.RepeatedScalarFieldContainer[int]
    data: bytes
    def __init__(self, dtype: _Optional[str] = ..., shape: _Optional[_Iterable[int]] = ..., data: _Optional[bytes] = ...) -> None: ...

class NamedTensor(_message.Message):
    __slots__ = ("name", "tensor")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TENSOR_FIELD_NUMBER: _ClassVar[int]
    name: str
    tensor: Tensor
    def __init__(self, name: _Optional[str] = ..., tensor: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class StateDict(_message.Message):
    __slots__ = ("entries",)
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    entries: _containers.RepeatedCompositeFieldContainer[NamedTensor]
    def __init__(self, entries: _Optional[_Iterable[_Union[NamedTensor, _Mapping]]] = ...) -> None: ...

class RunConfig(_message.Message):
    __slots__ = ("mode", "num_clients", "partition", "alpha", "seed", "batch_size", "lr", "dataset", "max_batches")
    MODE_FIELD_NUMBER: _ClassVar[int]
    NUM_CLIENTS_FIELD_NUMBER: _ClassVar[int]
    PARTITION_FIELD_NUMBER: _ClassVar[int]
    ALPHA_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    LR_FIELD_NUMBER: _ClassVar[int]
    DATASET_FIELD_NUMBER: _ClassVar[int]
    MAX_BATCHES_FIELD_NUMBER: _ClassVar[int]
    mode: str
    num_clients: int
    partition: str
    alpha: float
    seed: int
    batch_size: int
    lr: float
    dataset: str
    max_batches: int
    def __init__(self, mode: _Optional[str] = ..., num_clients: _Optional[int] = ..., partition: _Optional[str] = ..., alpha: _Optional[float] = ..., seed: _Optional[int] = ..., batch_size: _Optional[int] = ..., lr: _Optional[float] = ..., dataset: _Optional[str] = ..., max_batches: _Optional[int] = ...) -> None: ...

class JoinRequest(_message.Message):
    __slots__ = ("client_id", "shard")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    SHARD_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    shard: int
    def __init__(self, client_id: _Optional[str] = ..., shard: _Optional[int] = ...) -> None: ...

class JoinReply(_message.Message):
    __slots__ = ("config",)
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    config: RunConfig
    def __init__(self, config: _Optional[_Union[RunConfig, _Mapping]] = ...) -> None: ...

class RoundRequest(_message.Message):
    __slots__ = ("client_id", "last_round")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    LAST_ROUND_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    last_round: int
    def __init__(self, client_id: _Optional[str] = ..., last_round: _Optional[int] = ...) -> None: ...

class RoundInfo(_message.Message):
    __slots__ = ("round", "done", "global_state")
    ROUND_FIELD_NUMBER: _ClassVar[int]
    DONE_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_STATE_FIELD_NUMBER: _ClassVar[int]
    round: int
    done: bool
    global_state: StateDict
    def __init__(self, round: _Optional[int] = ..., done: _Optional[bool] = ..., global_state: _Optional[_Union[StateDict, _Mapping]] = ...) -> None: ...

class Activations(_message.Message):
    __slots__ = ("client_id", "round", "activations", "labels")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    ACTIVATIONS_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    round: int
    activations: Tensor
    labels: Tensor
    def __init__(self, client_id: _Optional[str] = ..., round: _Optional[int] = ..., activations: _Optional[_Union[Tensor, _Mapping]] = ..., labels: _Optional[_Union[Tensor, _Mapping]] = ...) -> None: ...

class Gradients(_message.Message):
    __slots__ = ("grad", "loss", "correct")
    GRAD_FIELD_NUMBER: _ClassVar[int]
    LOSS_FIELD_NUMBER: _ClassVar[int]
    CORRECT_FIELD_NUMBER: _ClassVar[int]
    grad: Tensor
    loss: float
    correct: int
    def __init__(self, grad: _Optional[_Union[Tensor, _Mapping]] = ..., loss: _Optional[float] = ..., correct: _Optional[int] = ...) -> None: ...

class StreamSummary(_message.Message):
    __slots__ = ("batches", "samples", "loss_sum", "correct")
    BATCHES_FIELD_NUMBER: _ClassVar[int]
    SAMPLES_FIELD_NUMBER: _ClassVar[int]
    LOSS_SUM_FIELD_NUMBER: _ClassVar[int]
    CORRECT_FIELD_NUMBER: _ClassVar[int]
    batches: int
    samples: int
    loss_sum: float
    correct: int
    def __init__(self, batches: _Optional[int] = ..., samples: _Optional[int] = ..., loss_sum: _Optional[float] = ..., correct: _Optional[int] = ...) -> None: ...

class Update(_message.Message):
    __slots__ = ("client_id", "round", "num_samples", "state")
    CLIENT_ID_FIELD_NUMBER: _ClassVar[int]
    ROUND_FIELD_NUMBER: _ClassVar[int]
    NUM_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    client_id: str
    round: int
    num_samples: int
    state: StateDict
    def __init__(self, client_id: _Optional[str] = ..., round: _Optional[int] = ..., num_samples: _Optional[int] = ..., state: _Optional[_Union[StateDict, _Mapping]] = ...) -> None: ...

class UpdateAck(_message.Message):
    __slots__ = ("accepted", "current_round")
    ACCEPTED_FIELD_NUMBER: _ClassVar[int]
    CURRENT_ROUND_FIELD_NUMBER: _ClassVar[int]
    accepted: bool
    current_round: int
    def __init__(self, accepted: _Optional[bool] = ..., current_round: _Optional[int] = ...) -> None: ...
