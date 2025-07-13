import asyncio
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from .database import EnhancedJsonDatabase
from .helper import ModelID, ModelType, default_shard_hash
from .logs import logger


class ShardedEnhancedJsonDatabase:
    """
    Wraps multiple EnhancedJsonDatabase instances (one per shard).
    """

    def __init__(
        self,
        base_filename: str,
        tables: Dict[str, Type[ModelType]],
        num_shards: int = 1,  # Use 1 for no horizontal sharding.
        shard_key_field: str = "shard_key",
        file_encryption_key: Optional[str] = None,
        field_encryption_key: Optional[str] = None,
        use_compression: bool = False,
        shard_hash_func: Callable[[str, int], int] = default_shard_hash,
        secondary_index_fields: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.num_shards = num_shards
        self.shard_key_field = shard_key_field
        self.shard_hash_func = shard_hash_func
        self.log = logger.bind(component="CombinedDB", shards=self.num_shards)

        # If only one shard, we use a single EnhancedJsonDatabase; otherwise, build a shard dict.
        if self.num_shards == 1:
            self.single_db = EnhancedJsonDatabase(
                filename=base_filename,
                tables=tables,
                file_encryption_key=file_encryption_key,
                field_encryption_key=field_encryption_key,
                secondary_index_fields=secondary_index_fields,
                use_compression=use_compression,
            )
            self.shards = None
        else:
            self.single_db = None
            self.shards: Dict[int, EnhancedJsonDatabase] = {}
            for shard_id in range(num_shards):
                shard_filename = self._get_shard_filename(base_filename, shard_id)
                self.shards[shard_id] = EnhancedJsonDatabase(
                    filename=shard_filename,
                    tables=tables,
                    file_encryption_key=file_encryption_key,
                    field_encryption_key=field_encryption_key,
                    secondary_index_fields=secondary_index_fields,
                    use_compression=use_compression,
                )

    @staticmethod
    def _get_shard_filename(base_filename: str, shard_id: int) -> str:
        base, ext = os.path.splitext(base_filename)
        return f"{base}_shard_{shard_id}{ext}"

    def _select_shard(self, record: Dict[str, Any]) -> int:
        if self.shard_key_field not in record:
            raise ValueError(
                f"Record must include '{self.shard_key_field}' for sharding."
            )
        shard_key = str(record[self.shard_key_field])
        return self.shard_hash_func(shard_key, self.num_shards)

    def add(self, table: str, instance: ModelType) -> ModelType:
        record = instance.model_dump()
        if self.num_shards == 1:
            return self.single_db.add(table, instance)
        shard_id = self._select_shard(record)
        return self.shards[shard_id].add(table, instance)

    def query(
        self, table: str, query: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        if self.num_shards == 1:
            return self.single_db.query(table, query)
        # If the query includes the shard key, we only search that shard.
        if query and self.shard_key_field in query:
            shard_id = self.shard_hash_func(
                str(query[self.shard_key_field]), self.num_shards
            )
            return self.shards[shard_id].query(table, query)
        else:
            results = []
            for shard in self.shards.values():
                results.extend(shard.query(table, query))
            return results

    def optimized_query(
        self, table: str, query: Optional[Dict[str, Any]] = None
    ) -> List[ModelType]:
        if self.num_shards == 1:
            return self.single_db.optimized_query(table, query)
        if query and self.shard_key_field in query:
            shard_id = self.shard_hash_func(
                str(query[self.shard_key_field]), self.num_shards
            )
            return self.shards[shard_id].optimized_query(table, query)
        results = []
        for shard in self.shards.values():
            results.extend(shard.optimized_query(table, query))
        return results

    def get_all(self, table: str, shard_key: Optional[str] = None) -> List[ModelType]:
        """
        Retrieve every record in `table`.
        - If single‐shard, delegate.
        - If multi‐shard and shard_key given, only that shard.
        - Otherwise, aggregate across all shards.
        """
        # single-shard case
        if self.num_shards == 1:
            return self.single_db.get_all(table)

        # constrained to one shard
        if shard_key is not None:
            shard_id = self.shard_hash_func(str(shard_key), self.num_shards)
            return self.shards[shard_id].get_all(table)

        # global: aggregate across all shards
        results: List[ModelType] = []
        for shard in self.shards.values():
            results.extend(shard.get_all(table))
        return results

    def get(
        self, table: str, id_value: ModelID, shard_key: Optional[str] = None
    ) -> ModelType:
        """
        Retrieve one record by its primary key (UUID or int).
        """
        if self.num_shards == 1:
            return self.single_db.get(table, id_value)  # single_db.get accepts uuid

        if shard_key is not None:
            shard_id = self.shard_hash_func(str(shard_key), self.num_shards)
            return self.shards[shard_id].get(table, id_value)

        # fallback scan
        for shard in self.shards.values():
            try:
                return shard.get(table, id_value)
            except Exception:
                continue
        raise ValueError(f"Record with ID {id_value!r} not found in any shard.")

    def update(
        self,
        table: str,
        id_value: int,
        new_data: Dict[str, Any],
        shard_key: Optional[str] = None,
    ) -> None:
        if self.num_shards == 1:
            self.single_db.update(table, id_value, new_data)
            return
        if shard_key is not None:
            shard_id = self.shard_hash_func(str(shard_key), self.num_shards)
            self.shards[shard_id].update(table, id_value, new_data)
        else:
            updated = False
            for shard in self.shards.values():
                try:
                    shard.update(table, id_value, new_data)
                    updated = True
                    break
                except Exception:
                    continue
            if not updated:
                raise ValueError(f"Record with ID {id_value} not found in any shard.")

    def delete(
        self, table: str, id_value: int, shard_key: Optional[str] = None
    ) -> None:
        if self.num_shards == 1:
            self.single_db.delete(table, id_value)
            return
        if shard_key is not None:
            shard_id = self.shard_hash_func(str(shard_key), self.num_shards)
            self.shards[shard_id].delete(table, id_value)
        else:
            deleted = False
            for shard in self.shards.values():
                try:
                    shard.delete(table, id_value)
                    deleted = True
                    break
                except Exception:
                    continue
            if not deleted:
                raise ValueError(f"Record with ID {id_value} not found in any shard.")

    def bulk_add(self, table: str, instances: List[ModelType]) -> List[ModelType]:
        if not instances:
            return []

        if self.num_shards == 1:
            return self.single_db.bulk_add(table, instances)

        # Group instances by shard
        shard_groups = defaultdict(list)
        for instance in instances:
            record = instance.model_dump()
            if self.shard_key_field not in record:
                raise ValueError(
                    f"Record must contain the shard key field '{self.shard_key_field}'."
                )
            shard_key = str(record[self.shard_key_field])
            shard_id = self.shard_hash_func(shard_key, self.num_shards)
            shard_groups[shard_id].append(instance)

        results = []
        for shard_id, group in shard_groups.items():
            results.extend(self.shards[shard_id].bulk_add(table, group))

        return results

    def bulk_update(
        self,
        table: str,
        updates: List[Tuple[int, Dict[str, Any]]],
        shard_keys: Optional[List[str]] = None,
    ) -> None:
        if not updates:
            return

        if self.num_shards == 1:
            self.single_db.bulk_update(table, updates)
            return

        if shard_keys and len(shard_keys) != len(updates):
            raise ValueError("Length of shard_keys must match length of updates.")

        # Group updates by shard
        shard_batches = defaultdict(list)
        for i, (id_value, new_data) in enumerate(updates):
            if shard_keys:
                shard_key = shard_keys[i]
            else:
                # Attempt to infer the shard by searching all shards (slower)
                shard_key = None
                for shard_id, shard in self.shards.items():
                    try:
                        record = shard.get(table, id_value)
                        shard_key = getattr(record, self.shard_key_field)
                        break
                    except Exception:
                        continue
                if shard_key is None:
                    raise ValueError(
                        f"Record with ID {id_value} not found in any shard."
                    )

            shard_id = self.shard_hash_func(str(shard_key), self.num_shards)
            shard_batches[shard_id].append((id_value, new_data))

        for shard_id, batch in shard_batches.items():
            self.shards[shard_id].bulk_update(table, batch)

    def bulk_delete(
        self, table: str, id_list: List[int], shard_keys: Optional[List[str]] = None
    ) -> None:
        if not id_list:
            return

        if self.num_shards == 1:
            self.single_db.bulk_delete(table, id_list)
            return

        if shard_keys and len(shard_keys) != len(id_list):
            raise ValueError("Length of shard_keys must match length of id_list.")

        shard_batches = defaultdict(list)

        for i, id_value in enumerate(id_list):
            if shard_keys:
                shard_key = shard_keys[i]
            else:
                # Infer shard key by scanning
                shard_key = None
                for shard_id, shard in self.shards.items():
                    try:
                        record = shard.get(table, id_value)
                        shard_key = getattr(record, self.shard_key_field)
                        break
                    except Exception:
                        continue
                if shard_key is None:
                    raise ValueError(
                        f"Record with ID {id_value} not found in any shard."
                    )

            shard_id = self.shard_hash_func(str(shard_key), self.num_shards)
            shard_batches[shard_id].append(id_value)

        for shard_id, ids in shard_batches.items():
            self.shards[shard_id].bulk_delete(table, ids)

    # Global Transaction Coordinaton: open a transaction on every shard.
    class GlobalTransaction:
        def __init__(self, combined_db: "ShardedEnhancedJsonDatabase"):
            self.combined = combined_db
            self.transactions = {}

        def __enter__(self):
            if self.combined.num_shards == 1:
                self.transactions[0] = self.combined.single_db.transaction()
                self.transactions[0].__enter__()
            else:
                for shard_id, db in self.combined.shards.items():
                    tx = db.transaction()
                    tx.__enter__()
                    self.transactions[shard_id] = tx
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            import traceback

            if exc_type:
                # 1) Log the exception and traceback
                tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
                self.log.error(f"[GlobalTransaction] Aborted due to: {exc_val!r}\n{tb}")

                # 2) Fire an error hook if you have one
                if hasattr(self.combined, "hooks"):
                    # sync‐trigger your hook manager
                    self.combined.hooks.trigger(
                        "transaction_error", exception=exc_val, traceback=tb
                    )

                # 3) Roll back all shards
                for tx in self.transactions.values():
                    try:
                        tx.rollback()
                    except Exception as e:
                        self.log.error(f"Error rolling back shard transaction: {e}")

            else:
                # Commit all shards
                for tx in self.transactions.values():
                    tx.commit()

            # Returning False => re-raise the original exception (unless you explicitly return True)
            return False

        async def __aenter__(self):
            # asynchronous entry: offload blocking calls to threads
            if self.combined.num_shards == 1:
                tx = self.combined.single_db.transaction()
                await asyncio.to_thread(tx.__enter__)
                self.transactions[0] = tx
            else:
                for sid, db in self.combined.shards.items():
                    tx = db.transaction()
                    await asyncio.to_thread(tx.__enter__)
                    self.transactions[sid] = tx
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            import traceback

            if exc_type:
                tb = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
                print(f"[GlobalTransaction] Async abort due to: {exc_val!r}\n{tb}")

                if hasattr(self.combined, "hooks"):
                    # async‐trigger your hook manager
                    await self.combined.hooks.trigger(
                        "transaction_error", exception=exc_val, traceback=tb
                    )

                for tx in self.transactions.values():
                    await asyncio.to_thread(tx.rollback)
            else:
                for tx in self.transactions.values():
                    await asyncio.to_thread(tx.commit)

            return False

    def global_transaction(self) -> "ShardedEnhancedJsonDatabase.GlobalTransaction":
        return ShardedEnhancedJsonDatabase.GlobalTransaction(self)

    # Asynchronous CRUD: use asyncio.to_thread to offload synchronous calls.
    async def async_add(self, table: str, instance: ModelType) -> ModelType:
        """
        Asynchronously add an instance to the appropriate shard.
        """
        return await asyncio.to_thread(self.add, table, instance)

    async def async_query(
        self, table: str, query: Optional[dict] = None
    ) -> List[ModelType]:
        """
        Asynchronously run a (possibly global) query.
        """
        return await asyncio.to_thread(self.query, table, query)

    async def async_get(
        self, table: str, id_value: Any, shard_key: Optional[str] = None
    ) -> ModelType:
        """
        Asynchronously fetch a record by primary key (and optional shard).
        """
        return await asyncio.to_thread(self.get, table, id_value, shard_key)

    async def async_update(
        self,
        table: str,
        id_value: Any,
        new_data: Dict[str, Any],
        shard_key: Optional[str] = None,
    ) -> None:
        """
        Asynchronously update a record.
        """
        return await asyncio.to_thread(
            self.update, table, id_value, new_data, shard_key
        )

    async def async_delete(
        self, table: str, id_value: Any, shard_key: Optional[str] = None
    ) -> None:
        """
        Asynchronously delete a record.
        """
        return await asyncio.to_thread(self.delete, table, id_value, shard_key)

    async def async_bulk_add(
        self, table: str, instances: List[ModelType]
    ) -> List[ModelType]:
        return await asyncio.to_thread(self.bulk_add, table, instances)

    async def async_bulk_update(
        self,
        table: str,
        updates: List[Tuple[Any, Dict[str, Any]]],
        shard_keys: Optional[List[str]] = None,
    ) -> None:
        return await asyncio.to_thread(self.bulk_update, table, updates, shard_keys)

    async def async_bulk_delete(
        self,
        table: str,
        id_list: List[Any],
        shard_keys: Optional[List[str]] = None,
    ) -> None:
        return await asyncio.to_thread(self.bulk_delete, table, id_list, shard_keys)

    # Replication, Backup, and Fault Tolerance
    def replicate(self, backup_dir: str) -> None:
        """Replicate each shard by copying its file into backup_dir."""
        os.makedirs(backup_dir, exist_ok=True)
        if self.num_shards == 1:
            backup_path = os.path.join(
                backup_dir, os.path.basename(self.single_db.filename)
            )
            self.single_db.backup(backup_path)
        else:
            for shard in self.shards.values():
                backup_path = os.path.join(backup_dir, os.path.basename(shard.filename))
                shard.backup(backup_path)

    def backup(self) -> None:
        """Backup all shards (or the single DB) using their native backup method."""
        if self.num_shards == 1:
            self.single_db.backup()
        else:
            for shard in self.shards.values():
                shard.backup()

    def restore(self, backup_files: Dict[int, str]) -> None:
        """
        Restore shards using a dictionary mapping shard_id to its backup filename.
        For a single DB, use key 0.
        """
        if self.num_shards == 1:
            self.single_db.restore(backup_files.get(0))
        else:
            for shard_id, shard in self.shards.items():
                if shard_id in backup_files:
                    shard.restore(backup_files[shard_id])

    def rotate_keys(
        self, new_file_key: str, new_field_key: Optional[str] = None
    ) -> None:
        if self.num_shards == 1:
            self.single_db.rotate_keys(new_file_key, new_field_key)
        else:
            for shard in self.shards.values():
                shard.rotate_keys(new_file_key, new_field_key)
