from uuid import UUID

import pytest
from pydantic import BaseModel

from meshmapdb.session import Session
from meshmapdb.shardeddatabases import CombinedEnhancedJsonDatabase


class Item(BaseModel):
    id: UUID | None = None
    name: str
    bucket: str


@pytest.fixture
def db(tmp_path):
    return CombinedEnhancedJsonDatabase(
        base_filename=str(tmp_path / "s.json"),
        tables={"items": Item},
        num_shards=2,
        shard_key_field="bucket",
    )


def test_session_commit(db):
    with Session(db) as sess:
        i = Item(name="one", bucket="A")
        sess.add("items", i)
        sess.update("items", i.id, {"name": "uno"})
    # After commit, record is visible
    rec = db.get("items", i.id, shard_key="A")
    assert rec.name == "uno"


def test_session_rollback(db):
    with pytest.raises(RuntimeError):
        with Session(db) as sess:
            i = Item(name="bad", bucket="B")
            sess.add("items", i)
            raise RuntimeError("fail")
    # rollback => record not persisted
    with pytest.raises(ValueError):
        db.get("items", i.id, shard_key="B")
