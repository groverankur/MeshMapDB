from uuid import UUID

import pytest
from pydantic import BaseModel

from meshmapdb.database import EnhancedJsonDatabase


class User(BaseModel):
    id: UUID | None = None
    name: str
    age: int


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "testdb.json")


@pytest.fixture
def db(db_path):
    return EnhancedJsonDatabase(
        filename=db_path,
        tables={"users": User},
        file_encryption_key=None,
        field_encryption_key=None,
        secondary_index_fields={"users": ["age"]},
    )


def test_add_get(db):
    u = User(name="Alice", age=30)
    db.add("users", u)
    got = db.get("users", u.id)
    assert got.name == "Alice"
    assert got.age == 30


def test_query(db):
    db.add("users", User(name="Bob", age=20))
    db.add("users", User(name="Carol", age=25))
    res = db.query("users", {"age": {"$gt": 21}})
    assert len(res) == 1
    assert res[0].name == "Carol"


def test_optimized_query(db):
    # age is indexed
    for i in range(5):
        db.add("users", User(name=f"U{i}", age=30))
    out = db.optimized_query("users", {"age": 30})
    assert len(out) == 5


def test_update_delete(db):
    u = User(name="Dan", age=40)
    db.add("users", u)
    db.update("users", u.id, {"age": 41})
    assert db.get("users", u.id).age == 41
    db.delete("users", u.id)
    with pytest.raises(ValueError):
        db.get("users", u.id)
