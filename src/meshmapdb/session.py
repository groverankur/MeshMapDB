import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .helper import ModelType
from .shardeddatabases import ShardedEnhancedJsonDatabase

if TYPE_CHECKING:
    from typing import Dict


class Session:
    """
    A unit of work wrapper over ShardedEnhancedJsonDatabase.
    """

    def __init__(self, db: ShardedEnhancedJsonDatabase):
        self.db = db
        self._gtx = db.global_transaction()

    def __enter__(self):
        self._gtx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._gtx.__exit__(exc_type, exc_val, exc_tb)

    async def __aenter__(self):
        # async entry
        return await self._gtx.__aenter__() or self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # async exit
        await self._gtx.__aexit__(exc_type, exc_val, exc_tb)

    def add(self, table: str, instance: ModelType):
        return self.db.add(table, instance)

    def query(self, table: str, **kwargs):
        return self.db.query(table, kwargs)

    def get(self, table: str, id_value, shard_key=None):
        return self.db.get(table, id_value, shard_key)

    def update(self, table: str, id_value, new_data: Dict[str, Any], shard_key=None):
        return self.db.update(table, id_value, new_data, shard_key)

    def delete(self, table: str, id_value, shard_key=None):
        return self.db.delete(table, id_value, shard_key)

    def commit(self):
        self._gtx.__exit__(None, None, None)

    def rollback(self):
        self._gtx.__exit__(Exception, Exception("rollback"), None)

    async def add_async(self, table: str, instance: Any) -> Any:
        return await self.db.async_add(table, instance)

    async def query_async(
        self, table: str, query: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        return await self.db.async_query(table, query)

    async def get_async(
        self, table: str, id_value: Any, shard_key: Optional[str] = None
    ) -> Any:
        return await self.db.async_get(table, id_value, shard_key)

    async def update_async(
        self,
        table: str,
        id_value: Any,
        new_data: Dict[str, Any],
        shard_key: Optional[str] = None,
    ) -> None:
        return await self.db.async_update(table, id_value, new_data, shard_key)

    async def delete_async(
        self, table: str, id_value: Any, shard_key: Optional[str] = None
    ) -> None:
        return await self.db.async_delete(table, id_value, shard_key)

    async def commit_async(self) -> None:
        # explicit async commit
        await asyncio.to_thread(self._gtx.__exit__, None, None, None)

    async def rollback_async(self) -> None:
        # explicit async rollback
        await asyncio.to_thread(
            self._gtx.__exit__, Exception, Exception("rollback"), None
        )
