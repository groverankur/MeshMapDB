import json
from typing import Annotated, Any, Callable, TypeVar, Union

import uuid_utils as uuid
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.hashes import SHA256, Hash
from pydantic import BaseModel, BeforeValidator, Field, PlainSerializer

try:
    from blosc2 import compress2, decompress

    def compress_data(data: bytes) -> bytes:
        return compress2(data)

    def decompress_data(data: bytes) -> bytes:
        return decompress(data)
except ImportError as e:
    raise ImportError("Dependencies not satisfied: pip install blosc2") from e

try:
    import orjson

    def json_dumps(obj):
        try:
            return orjson.dumps(
                obj, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_UUID
            )
        except orjson.JSONDecodeError as e:
            print(e)
            raise ValueError(
                "Failed to decode data using orjson. Possibly not decrypted or decompressed."
            ) from e

    def json_loads(b):
        if isinstance(b, str):
            b = b.encode("utf-8")  # Convert str to bytes
        elif not isinstance(b, (bytes, bytearray)):
            raise TypeError("orjson.loads expects bytes or str")

        try:
            return orjson.loads(b)
        except orjson.JSONDecodeError as e:
            raise ValueError(
                "Failed to decode data using orjson. Possibly not decrypted or decompressed."
            ) from e


except ImportError:
    import json

    def json_dumps(obj):
        return json.dumps(obj).encode("utf-8")

    def json_loads(b):
        try:
            if isinstance(b, bytes):
                return json.loads(b.decode("utf-8"))
            elif isinstance(b, str):
                return json.loads(b)
            else:
                raise TypeError("json_loads expects bytes or str.")
        except UnicodeDecodeError as e:
            raise ValueError(
                "Failed to decode input as UTF-8. Possibly not decrypted or decompressed."
            ) from e


ModelType = TypeVar("ModelType", bound=BaseModel)
ModelID = Union[int, uuid.UUID]
EventHook = Callable[..., Any]

BetterUUID = Annotated[
    uuid.UUID,
    BeforeValidator(lambda x: uuid.UUID(x) if isinstance(x, str) else x),
    PlainSerializer(lambda x: str(x)),
    Field(
        description="Better annotation for UUID, parses from string format. Serializes to string format."
    ),
]


# Helper: Generate a unique ID for a record.
def generate_id() -> uuid.UUID:
    return uuid.uuid7()


# Helper: Default shard hash function using SHA256.
def default_shard_hash(shard_key: str, num_shards: int) -> int:
    hash_digest = Hash(SHA256(), default_backend())
    hash_digest.update(shard_key.encode("utf-8"))
    return int(hash_digest.finalize().hex(), 16) % num_shards
