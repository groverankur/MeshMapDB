from uuid import UUID

import pytest
from pydantic import BaseModel

from meshmapdb.shardeddatabases import ShardedEnhancedJsonDatabase


class User(BaseModel):
    id: UUID | None = None
    name: str
    region: str


@pytest.fixture
def cdb(tmp_path):
    base = str(tmp_path / "sharded.json")
    return ShardedEnhancedJsonDatabase(
        base_filename=base,
        tables={"users": User},
        num_shards=3,
        shard_key_field="region",
    )


def test_sharded_add_query(cdb):
    u1 = User(name="A", region="us")
    u2 = User(name="B", region="eu")
    cdb.add("users", u1)
    cdb.add("users", u2)

    # only one shard scanned for each
    out = cdb.query("users", {"region": "us"})
    assert len(out) == 1 and out[0].name == "A"

    # global query
    all_users = cdb.query("users", {})
    assert {u.name for u in all_users} == {"A", "B"}


def test_get_all(cdb):
    u = User(name="X", region="apac")
    cdb.add("users", u)
    lst = cdb.get_all("users")
    assert len(lst) == 1 and lst[0].region == "apac"
