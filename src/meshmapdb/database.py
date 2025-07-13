import asyncio
import copy
import datetime
import os
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from filelock import FileLock

from .cipher import AESCipher
from .helper import (
    ModelType,
    compress_data,
    decompress_data,
    generate_id,
    json_dumps,
    json_loads,
)
from .hooks import HookManager
from .logs import logger


class EnhancedJsonDatabase:
    """
    Single-file JSON DB with:
      - Multiple tables (pydantic schemas)
      - Vertical shards
      - Secondary indexes
      - Transactions
      - Versioning
      - Encryption (file + field)
      - Backup & health check
    """

    def __init__(
        self,
        filename: str,
        tables: Dict[str, Type[ModelType]],  # mapping table name to a SQLModel subclass
        file_encryption_key: Optional[str] = None,
        field_encryption_key: Optional[str] = None,
        use_compression: bool = False,
        secondary_index_fields: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.filename = filename
        self.tables_schemas = tables
        self.lock = FileLock(f"{filename}.lock")
        self.internal_lock = threading.RLock()
        self.use_compression = use_compression
        self.hooks = HookManager()
        self.log = logger.bind(component="EnhancedJsonDatabase", file=self.filename)
        self.file_cipher = (
            AESCipher(file_encryption_key) if file_encryption_key else None
        )
        self.field_cipher = (
            AESCipher(field_encryption_key) if field_encryption_key else None
        )
        self.secondary_index_fields = secondary_index_fields or {}
        self._loaded_indexes: Dict[str, Dict[str, Dict[Any, List[int]]]] = {}
        self._transaction_data: Optional[Dict[str, Any]] = None
        self._init_db_file()
        self._build_all_indexes()

    def _init_db_file(self) -> None:
        with self.lock, self.internal_lock:
            if not os.path.exists(self.filename):
                init_data = {"tables": {}}
                for table, model_class in self.tables_schemas.items():
                    table_struct = {
                        "version": 1,
                        "records": [],
                        "history": {},
                        "index_fields": self.secondary_index_fields.get(table, []),
                        "indexes": {},
                    }
                    if hasattr(model_class, "__vertical_shards__"):
                        vertical = {
                            shard: {}
                            for shard in model_class.__vertical_shards__.keys()
                        }
                        table_struct["vertical"] = vertical
                    init_data["tables"][table] = table_struct
                self._write_data(init_data)

    def _read_data(self) -> Dict[str, Any]:
        with self.lock, self.internal_lock:
            if not os.path.exists(self.filename):
                self._init_db_file()
            with open(self.filename, "rb") as f:
                content = f.read()
            if not content:
                return {"tables": {}}
            if self.file_cipher:
                content = self.file_cipher.decrypt(content)
            if self.use_compression:
                content = decompress_data(content)
            try:
                data = json_loads(content)
            except Exception:
                raise ValueError(
                    "Decryption succeeded but JSON parsing failed — possible tampering."
                )
            if not isinstance(data, dict) or "tables" not in data:
                raise ValueError("Invalid data schema post-decryption.")
            return data

    def _write_data(self, data: Dict[str, Any], rebuild_indexes: bool = True) -> None:
        with self.lock, self.internal_lock:
            raw = json_dumps(data)
            if self.file_cipher:
                raw = self.file_cipher.encrypt(raw)
            if self.use_compression:
                raw = compress_data(raw)
            with open(self.filename, "wb") as f:
                f.write(raw)
            if rebuild_indexes:
                self._build_all_indexes()

    def _ensure_index(self, table: str, field: str):
        """
        If not already loaded in memory, load or build the index for table.field.
        Persist built indexes back into the JSON file.
        """
        # 1) check manifest
        idx_fields = self._read_data()["tables"][table].get("index_fields", [])
        if field not in idx_fields:
            raise ValueError(f"Field '{field}' is not declared indexable on '{table}'")

        # 2) already loaded?
        if table in self._loaded_indexes and field in self._loaded_indexes[table]:
            return

        # 3) load from JSON if present
        data = self._read_data()
        tbl = data["tables"][table]
        disk_index = tbl.get("indexes", {}).get(field)
        if disk_index is None:
            # build from scratch
            disk_index = {}
            for rec in tbl["records"]:
                val = rec.get(field)
                if val is not None:
                    disk_index.setdefault(val, []).append(rec["id"])
            # persist back
            tbl.setdefault("indexes", {})[field] = disk_index
            self._write_data(data)

        # 4) cache
        self._loaded_indexes.setdefault(table, {})[field] = disk_index

    def _build_all_indexes(self) -> None:
        data = self._read_data()
        self.indexes = {}
        for table, table_data in data.get("tables", {}).items():
            primary_index = {}
            secondary_indexes: Dict[str, Dict[Any, List[dict]]] = {}
            # initialize empty maps for only declared fields
            for field in self.secondary_index_fields.get(table, []):
                secondary_indexes[field] = {}

            for record in table_data.get("records", []):
                pk = record.get("id")
                if pk:
                    model_class = self.tables_schemas.get(table)
                    if (
                        model_class
                        and hasattr(model_class, "__vertical_shards__")
                        and "vertical" in table_data
                    ):
                        vertical_parts = table_data["vertical"]
                        for shard_label, shard_data in vertical_parts.items():
                            if str(pk) in shard_data:
                                record.update(shard_data[str(pk)])
                    primary_index[pk] = record
                # only index declared secondary fields
                for field in secondary_indexes:
                    val = record.get(field)
                    if val is not None:
                        secondary_indexes[field].setdefault(val, []).append(record)

            self.indexes[table] = {
                "primary": primary_index,
                "secondary": secondary_indexes,
            }

    def _update_indexes_on_add(self, table: str, record: dict):
        for field in self._loaded_indexes.get(table, {}):
            val = record.get(field)
            if val is not None:
                self._loaded_indexes[table][field].setdefault(val, []).append(
                    record["id"]
                )
                # persist change
                data = self._read_data()
                data["tables"][table]["indexes"][field] = self._loaded_indexes[table][
                    field
                ]
                self._write_data(data)

    def _update_indexes_on_delete(self, table: str, record: dict):
        """
        Remove record['id'] from every in-memory bucket for each loaded index field,
        and persist the updated index map to disk.
        """
        tid = record["id"]
        data = self._read_data()

        for field, index_map in self._loaded_indexes.get(table, {}).items():
            val = record.get(field)
            if val is None:
                continue

            bucket = index_map.get(val)
            if bucket and tid in bucket:
                bucket.remove(tid)

                # persist updated index to JSON
                data["tables"][table]["indexes"].setdefault(field, index_map)
                self._write_data(data)

    def _update_indexes_on_update(self, table: str, old_record: dict, new_record: dict):
        """
        For each loaded index field, if the value changed,
        remove the ID from the old bucket and add it to the new one.
        Then persist.
        """
        tid = old_record["id"]
        data = self._read_data()

        for field, index_map in self._loaded_indexes.get(table, {}).items():
            old_val = old_record.get(field)
            new_val = new_record.get(field)

            # if it moved from one bucket to another:
            if old_val != new_val:
                # remove from old bucket
                if old_val is not None:
                    bucket = index_map.get(old_val, [])
                    if tid in bucket:
                        bucket.remove(tid)

                # add to new bucket
                if new_val is not None:
                    index_map.setdefault(new_val, []).append(tid)

                # persist updated index to JSON
                data["tables"][table]["indexes"].setdefault(field, index_map)

        self._write_data(data)

    async def _trigger(self, event: str, **ctx: Any):
        await self.hooks.trigger(event, **ctx)

    def _decrypt_fields(self, model_class: Type[ModelType], record: dict) -> dict:
        if hasattr(model_class, "__encrypted_fields__") and self.field_cipher:
            for field in model_class.__encrypted_fields__:
                if field in record and record[field] is not None:
                    try:
                        ciphertext = bytes.fromhex(record[field])
                        decrypted = self.field_cipher.decrypt(ciphertext)
                        record[field] = decrypted.decode("utf-8")
                    except Exception as e:
                        raise ValueError(f"Failed to decrypt field '{field}': {e}")
        return record

    def _encrypt_fields(self, model_class: Type[ModelType], record: dict) -> dict:
        if hasattr(model_class, "__encrypted_fields__") and self.field_cipher:
            for field in model_class.__encrypted_fields__:
                if field in record and record[field] is not None:
                    val = record[field]
                    # Avoid double encrypting if already hex string
                    if not (
                        isinstance(val, str)
                        and all(c in "0123456789abcdefABCDEF" for c in val)
                        and len(val) % 2 == 0
                    ):
                        encoded = str(val).encode("utf-8")
                        record[field] = self.field_cipher.encrypt(encoded).hex()
        return record

    def advanced_match(self, record: dict, query: dict) -> bool:
        for key, cond in query.items():
            val = record.get(key)

            if isinstance(cond, dict) and any(k.startswith("$") for k in cond):
                for op, arg in cond.items():
                    if op == "$gt" and not (val > arg):
                        return False
                    elif op == "$lt" and not (val < arg):
                        return False
                    elif op == "$gte" and not (val >= arg):
                        return False
                    elif op == "$lte" and not (val <= arg):
                        return False
                    elif op == "$ne" and not (val != arg):
                        return False
                    elif op == "$in" and val not in arg:
                        return False
                    elif op == "$eq" and not (val == arg):
                        return False
                    # TODO add $nin, regex, etc. here
                continue

            if isinstance(cond, dict):
                if not isinstance(val, dict):
                    return False
                if not self.advanced_match(val, cond):
                    return False
                continue

            if val != cond:
                return False

        return True

    def _split_vertical(
        self, model_class: Type[ModelType], record: dict
    ) -> dict | Dict[str, dict]:
        vertical_data = {}
        core_record = record.copy()
        if hasattr(model_class, "__vertical_shards__"):
            for shard_label, fields in model_class.__vertical_shards__.items():
                shard_part = {}
                for field in fields:
                    if field in core_record:
                        shard_part[field] = core_record.pop(field)
                vertical_data[shard_label] = shard_part
        return core_record, vertical_data

    def add(self, table: str, instance: ModelType) -> ModelType:
        data = self._read_data()
        if table not in data.get("tables", {}):
            raise ValueError(f"Table {table} does not exist.")
        model_class = self.tables_schemas.get(table)
        if not isinstance(instance, model_class):
            raise ValueError(
                f"Instance must be of type {model_class.__name__} for table '{table}'."
            )
        record = instance.model_dump()
        if record.get("id") is None:
            record["id"] = generate_id()
            instance.id = record["id"]
            # setattr(instance, "id", record["id"])
        else:
            if record["id"] in self.indexes.get(table, {}).get("primary", {}):
                raise ValueError(
                    f"Record with ID {record['id']} already exists in table '{table}'."
                )
        record = self._encrypt_fields(model_class, record)
        vertical_data = {}
        if hasattr(model_class, "__vertical_shards__"):
            record, vertical_data = self._split_vertical(model_class, record)
        asyncio.run(self.hooks.trigger("before_insert", table=table, record=record))
        data["tables"][table]["records"].append(record)
        if vertical_data:
            table_vertical = data["tables"][table].setdefault("vertical", {})
            for shard_label, shard_part in vertical_data.items():
                table_vertical.setdefault(shard_label, {})
                table_vertical[shard_label][str(record["id"])] = shard_part
        self._write_data(data)
        asyncio.run(self.hooks.trigger("after_insert", table=table, record=record))
        self._update_indexes_on_add(table, record)
        return instance

    def query(self, table: str, query: Optional[dict] = None) -> List[ModelType]:
        data = self._read_data()
        if table not in data["tables"]:
            raise ValueError(f"Table '{table}' does not exist.")

        records = data["tables"][table]["records"]
        model_cls = self.tables_schemas[table]

        vertical_fields = {
            f
            for fields in getattr(model_cls, "__vertical_shards__", {}).values()
            for f in fields
        }
        needs_vert = bool(query and vertical_fields & set(query.keys()))
        vertical = data["tables"][table].get("vertical", {}) if needs_vert else {}

        full = []
        for base in records:
            rec = copy.deepcopy(base)
            if needs_vert:
                rid = str(rec["id"])
                for shard_label, shard_data in vertical.items():
                    if rid in shard_data:
                        rec.update(shard_data[rid])
            full.append(rec)

        matched = (
            full if not query else [r for r in full if self.advanced_match(r, query)]
        )

        out = []
        for r in matched:
            # make sure vertical is always included
            if hasattr(model_cls, "__vertical_shards__"):
                rid = str(r["id"])
                for shard_label, shard_data in vertical.items():
                    if rid in shard_data:
                        r.update(shard_data[rid])
            r = self._decrypt_fields(model_cls, r)
            out.append(model_cls(**r))

        return out

    def optimized_query(
        self, table: str, query: Optional[dict] = None
    ) -> List[ModelType]:
        """
        Use a secondary index when the query is a single top‐level equality on an indexed field.
        Nested or operator queries always fall back to full scan.
        """
        if not query or len(query) != 1:
            return self.query(table, query)

        field, cond = next(iter(query.items()))

        if not (not isinstance(cond, dict)):
            return self.query(table, query)

        if field in self.secondary_index_fields.get(table, []):
            self._ensure_index(table, field)
            ids = self._loaded_indexes[table][field].get(cond, [])
            results = []
            for _id in ids:
                rec = self.indexes[table]["primary"].get(_id)
                if rec and self.advanced_match(rec, query):
                    results.append(self.tables_schemas[table](**rec))
            return results

        return self.query(table, query)

    def get(self, table: str, id_value: int) -> ModelType:
        idx = self.indexes.get(table, {}).get("primary", {})
        if id_value in idx:
            record = copy.deepcopy(idx[id_value])
            model_class = self.tables_schemas.get(table)
            if hasattr(model_class, "__vertical_shards__"):
                vertical_data = self._read_data()["tables"][table].get("vertical", {})
                for shard_label, shard in vertical_data.items():
                    if str(id_value) in shard:
                        record.update(shard[str(id_value)])
            record = self._decrypt_fields(model_class, record)
            return model_class(**record)
        raise ValueError(f"Record with ID {id_value} not found in table '{table}'.")

    def get_all(self, table: str) -> List[ModelType]:
        """
        Retrieve every record in `table`, merging vertical shards and decrypting fields.
        """
        data = self._read_data()
        tbl = data.get("tables", {}).get(table)
        if tbl is None:
            raise ValueError(f"Table '{table}' does not exist.")

        model_class = self.tables_schemas[table]
        vertical_data = tbl.get("vertical", {})

        results: List[ModelType] = []
        for raw in tbl.get("records", []):
            rec = copy.deepcopy(raw)

            if hasattr(model_class, "__vertical_shards__"):
                for shard_label, shard_map in vertical_data.items():
                    if str(rec["id"]) in shard_map:
                        rec.update(shard_map[str(rec["id"])])

            rec = self._decrypt_fields(model_class, rec)

            results.append(model_class(**rec))

        return results

    def update(self, table: str, id_value: int, new_data: Dict[str, Any]) -> None:
        data = self._read_data()
        table_data = data["tables"].get(table, {})
        records = table_data.get("records", [])
        found = False
        for i, record in enumerate(records):
            if record.get("id") == id_value:
                model_class = self.tables_schemas.get(table)
                vertical_updates = {}
                if hasattr(model_class, "__vertical_shards__"):
                    for shard_label, fields in model_class.__vertical_shards__.items():
                        part = {}
                        for field in fields:
                            if field in new_data:
                                part[field] = new_data.pop(field)
                        if part:
                            vertical_updates[shard_label] = part
                decrypted_record = self._decrypt_fields(
                    model_class, copy.deepcopy(record)
                )
                asyncio.run(
                    self.hooks.trigger("before_update", table=table, record=record)
                )

                history = table_data.setdefault("history", {})
                historical_versions = history.setdefault(str(id_value), [])
                historical_versions.append(
                    {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "record": decrypted_record,
                    }
                )

                record.update(new_data)
                if vertical_updates:
                    table_vertical = table_data.setdefault("vertical", {})
                    for shard_label, update_part in vertical_updates.items():
                        table_vertical.setdefault(shard_label, {})
                        if str(id_value) in table_vertical[shard_label]:
                            table_vertical[shard_label][str(id_value)].update(
                                update_part
                            )
                        else:
                            table_vertical[shard_label][str(id_value)] = update_part

                asyncio.run(
                    self.hooks.trigger("after_update", table=table, record=record)
                )
                found = True
                break
        if not found:
            raise ValueError(f"Record with ID {id_value} not found in table '{table}'.")
        self._write_data(data)
        self._update_indexes_on_update(table, record)

    def delete(self, table: str, id_value: int) -> None:
        data = self._read_data()
        table_data = data["tables"].get(table, {})
        records = table_data.get("records", [])
        for i, record in enumerate(records):
            if record.get("id") == id_value:
                asyncio.run(
                    self.hooks.trigger("before_delete", table=table, record=record)
                )
                records.pop(i)
                if "vertical" in table_data:
                    for shard_label in table_data["vertical"].keys():
                        if str(id_value) in table_data["vertical"][shard_label]:
                            del table_data["vertical"][shard_label][str(id_value)]
                asyncio.run(
                    self.hooks.trigger("after_delete", table=table, record=record)
                )
                self._write_data(data)
                self._update_indexes_on_delete(table, record)
                return
        raise ValueError(f"Record with ID {id_value} not found in table '{table}'.")

    def bulk_add(self, table: str, instances: List[ModelType]) -> List[ModelType]:
        if not instances:
            return []

        data = self._read_data()
        if table not in data["tables"]:
            raise ValueError(f"Table '{table}' does not exist.")

        model_class = self.tables_schemas[table]

        for field in self.secondary_index_fields.get(table, []):
            self._ensure_index(table, field)

        records_to_add = []
        vertical_batch: Dict[str, Dict[str, dict]] = {}

        for inst in instances:
            if not isinstance(inst, model_class):
                raise ValueError(
                    f"All instances must be of type {model_class.__name__}"
                )

            rec = inst.model_dump()
            # assign ID if missing
            if rec.get("id") is None:
                rec["id"] = generate_id()
                inst.id = rec["id"]
            # duplicate-ID check
            elif rec["id"] in self.indexes.get(table, {})["primary"]:
                raise ValueError(f"Duplicate ID '{rec['id']}' in table '{table}'.")

            rec = self._encrypt_fields(model_class, rec)

            if hasattr(model_class, "__vertical_shards__"):
                rec, vert = self._split_vertical(model_class, rec)
                for label, part in vert.items():
                    vertical_batch.setdefault(label, {})[str(rec["id"])] = part

            asyncio.run(self.hooks.trigger("before_insert", table=table, record=rec))
            records_to_add.append(rec)

        data["tables"][table]["records"].extend(records_to_add)
        if vertical_batch:
            tv = data["tables"][table].setdefault("vertical", {})
            for label, chunk in vertical_batch.items():
                tv.setdefault(label, {}).update(chunk)

        self._write_data(data)

        for rec in records_to_add:
            self._update_indexes_on_add(table, rec)
            asyncio.run(self.hooks.trigger("after_insert", table=table, record=rec))

        return instances

    def bulk_update(
        self, table: str, updates: List[Tuple[int, Dict[str, Any]]]
    ) -> None:
        if not updates:
            return

        data = self._read_data()
        tbl = data["tables"].get(table)
        if not tbl:
            raise ValueError(f"Table '{table}' does not exist.")
        records = tbl["records"]
        model_class = self.tables_schemas[table]

        for field in self.secondary_index_fields.get(table, []):
            self._ensure_index(table, field)

        id_map = {r["id"]: r for r in records}

        vertical_updates: Dict[str, Dict[str, dict]] = {}

        for rec_id, new_data in updates:
            if rec_id not in id_map:
                raise ValueError(f"Record with ID {rec_id} not found.")

            old_rec = copy.deepcopy(id_map[rec_id])
            rec = id_map[rec_id]

            asyncio.run(
                self.hooks.trigger(
                    "before_update",
                    table=table,
                    record=old_rec,
                    **{"new_data": new_data},
                )
            )
            hist = tbl.setdefault("history", {}).setdefault(str(rec_id), [])
            hist.append(
                {"timestamp": datetime.datetime.utcnow().isoformat(), "record": old_rec}
            )

            if hasattr(model_class, "__vertical_shards__"):
                for label, fields in model_class.__vertical_shards__.items():
                    part = {f: new_data.pop(f) for f in fields if f in new_data}
                    if part:
                        vertical_updates.setdefault(label, {})[str(rec_id)] = part

            rec.update(new_data)

            asyncio.run(self.hooks.trigger("before_update", table=table, record=rec))

            self._update_indexes_on_update(table, old_rec, rec)

        if vertical_updates:
            tv = tbl.setdefault("vertical", {})
            for label, chunk in vertical_updates.items():
                tv.setdefault(label, {}).update(chunk)

        self._write_data(data)

    def bulk_delete(self, table: str, id_list: List[int]) -> None:
        if not id_list:
            return

        data = self._read_data()
        tbl = data["tables"].get(table)
        if not tbl:
            raise ValueError(f"Table '{table}' does not exist.")

        records = tbl["records"]
        ids = set(id_list)

        for field in self.secondary_index_fields.get(table, []):
            self._ensure_index(table, field)

        to_delete, to_keep = [], []
        for rec in records:
            if rec["id"] in ids:
                to_delete.append(rec)
            else:
                to_keep.append(rec)

        if not to_delete:
            raise ValueError("None of the specified records were found.")

        if "vertical" in tbl:
            for label, shard_map in tbl["vertical"].items():
                for rec_id in ids:
                    shard_map.pop(str(rec_id), None)

        tbl["records"] = to_keep
        self._write_data(data)

        for rec in to_delete:
            asyncio.run(self.hooks.trigger("before_delete", table=table, record=rec))
            self._update_indexes_on_delete(table, rec)
            asyncio.run(self.hooks.trigger("after_delete", table=table, record=rec))

    async def async_add(self, table: str, instance: ModelType) -> ModelType:
        """
        Asynchronously add an instance to `table`.
        """
        return await asyncio.to_thread(self.add, table, instance)

    async def async_query(
        self, table: str, query: Optional[dict] = None
    ) -> List[ModelType]:
        """
        Asynchronously run query on `table`.
        """
        return await asyncio.to_thread(self.query, table, query)

    async def async_get(self, table: str, id_value: Any) -> ModelType:
        """
        Asynchronously get a record by its primary key.
        """
        return await asyncio.to_thread(self.get, table, id_value)

    async def async_update(
        self, table: str, id_value: Any, new_data: Dict[str, Any]
    ) -> None:
        """
        Asynchronously update a record.
        """
        return await asyncio.to_thread(self.update, table, id_value, new_data)

    async def async_delete(self, table: str, id_value: Any) -> None:
        """
        Asynchronously delete a record.
        """
        return await asyncio.to_thread(self.delete, table, id_value)

    async def async_bulk_add(
        self, table: str, instances: List[ModelType]
    ) -> List[ModelType]:
        return await asyncio.to_thread(self.bulk_add, table, instances)

    async def async_bulk_update(
        self, table: str, updates: List[Tuple[Any, Dict[str, Any]]]
    ) -> None:
        return await asyncio.to_thread(self.bulk_update, table, updates)

    async def async_bulk_delete(self, table: str, id_list: List[Any]) -> None:
        return await asyncio.to_thread(self.bulk_delete, table, id_list)

    def _migrate_schema(
        self, table: str, migration_function: Callable[[dict], dict], new_version: int
    ) -> None:
        data = self._read_data()
        table_data = data["tables"].get(table, {})
        if not table_data:
            raise ValueError(f"Table '{table}' does not exist.")
        current_version = table_data.get("version", 1)
        if new_version <= current_version:
            raise ValueError("New version must be greater than the current version.")
        for idx, record in enumerate(table_data.get("records", [])):
            table_data["records"][idx] = migration_function(record)
        table_data["version"] = new_version
        self._write_data(data)

    def migrate_schema(
        self, table: str, migration_function: Callable[[dict], dict], new_version: int
    ) -> None:
        data = self._read_data()
        table_data = data["tables"].get(table, {})
        if not table_data:
            raise ValueError(f"Table '{table}' does not exist.")

        current_version = table_data.get("version", 1)
        if new_version <= current_version:
            raise ValueError("New version must be greater than the current version.")

        model_class = self.tables_schemas.get(table)
        vertical_enabled = hasattr(model_class, "__vertical_shards__")
        vertical_data = table_data.get("vertical", {}) if vertical_enabled else {}

        for idx, core_record in enumerate(table_data.get("records", [])):
            full_record = copy.deepcopy(core_record)
            record_id = str(full_record.get("id"))

            if vertical_enabled:
                for shard_label, shard_map in vertical_data.items():
                    if record_id in shard_map:
                        full_record.update(shard_map[record_id])

            migrated_record = migration_function(full_record)

            if vertical_enabled:
                core, vertical_parts = self._split_vertical(
                    model_class, migrated_record
                )
                table_data["records"][idx] = core

                for shard_label, shard_content in vertical_parts.items():
                    vertical_data.setdefault(shard_label, {})
                    vertical_data[shard_label][record_id] = shard_content
            else:
                table_data["records"][idx] = migrated_record

        table_data["version"] = new_version
        self._write_data(data)

    def backup(self, backup_filename: Optional[str] = None) -> None:
        """
        Create a backup copy of the current database file.
        If backup_filename is not provided, a timestamp is appended.
        """
        from shutil import copy

        if backup_filename is None:
            timestamp = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            backup_filename = f"{os.path.splitext(self.filename)[0]}_backup_{timestamp}{os.path.splitext(self.filename)[1]}"
        copy(self.filename, backup_filename)

    def restore(self, backup_filename: str) -> None:
        """Restore the database from a backup file."""
        from shutil import copy

        copy(backup_filename, self.filename)
        self._build_all_indexes()

    def health_check(self) -> bool:
        """A simple health check that verifies the database file can be read and parsed."""
        try:
            self._read_data()
            return True
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    class Transaction:
        def __init__(self, db: "EnhancedJsonDatabase"):
            self.db = db
            self._lock = db.internal_lock

        def commit(self):
            # Persist staged changes
            with self._lock:
                self.db._write_data(self.db._transaction_data)
                self.db._transaction_data = None

        def rollback(self):
            # Restore backup
            with self._lock:
                self.db._transaction_data = None
                self.db._write_data(self._backup)

        def __enter__(self):
            with self._lock:
                self._backup = copy.deepcopy(self.db._read_data())
                # Stage a working copy
                self.db._transaction_data = copy.deepcopy(self._backup)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            """
            On exception: log full details, then rollback.
            Otherwise: commit.
            Return False to re-raise the exception after cleanup.
            """
            import traceback

            if exc_type:
                tb_str = "".join(traceback.format_exception(exc_type, exc_val, exc_tb))
                self.log.error(
                    "Transaction aborted due to exception: %s\nTraceback:\n%s",
                    exc_val,
                    tb_str,
                )
                self.rollback()
            else:
                self.commit()

            return False

    def transaction(self) -> "EnhancedJsonDatabase.Transaction":
        return EnhancedJsonDatabase.Transaction(self)

    def rotate_keys(
        self, new_file_key: str, new_field_key: Optional[str] = None
    ) -> None:
        data = self._read_data()
        for table, model_class in self.tables_schemas.items():
            if hasattr(model_class, "__encrypted_fields__") and self.field_cipher:
                for record in data["tables"][table].get("records", []):
                    self._decrypt_fields(model_class, record)
                if "vertical" in data["tables"][table]:
                    for shard_label, shard in data["tables"][table]["vertical"].items():
                        for _id, shard_data in shard.items():
                            self._decrypt_fields(model_class, shard_data)
        self.file_cipher = AESCipher(new_file_key)
        if new_field_key:
            self.field_cipher = AESCipher(new_field_key)

            for table, model_class in self.tables_schemas.items():
                if hasattr(model_class, "__encrypted_fields__"):
                    for record in data["tables"][table].get("records", []):
                        for field in model_class.__encrypted_fields__:
                            if field in record and record[field] is not None:
                                encoded = str(record[field]).encode("utf-8")
                                record[field] = self.field_cipher.encrypt(encoded).hex()
                    if "vertical" in data["tables"][table]:
                        for shard_label, shard in data["tables"][table][
                            "vertical"
                        ].items():
                            for _id, shard_data in shard.items():
                                for field in model_class.__encrypted_fields__:
                                    if (
                                        field in shard_data
                                        and shard_data[field] is not None
                                    ):
                                        encoded = str(shard_data[field]).encode("utf-8")
                                        shard_data[field] = self.field_cipher.encrypt(
                                            encoded
                                        ).hex()
        self._write_data(data)
