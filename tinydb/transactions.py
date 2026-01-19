"""
This module implements atomic transactions for TinyDB.

Transactions allow multiple database operations to be executed atomically -
either all operations succeed, or all fail together with automatic rollback.
This is essential for maintaining data consistency in scenarios like
transferring money between accounts.

Example usage:

    >>> from tinydb import TinyDB, where
    >>> db = TinyDB('db.json')
    >>> accounts = db.table('accounts')
    >>>
    >>> # Transfer $100 from account 1 to account 2
    >>> with db.transaction() as txn:
    ...     txn.update(accounts, {'balance': lambda doc: doc['balance'] - 100},
    ...                where('id') == 1)
    ...     txn.update(accounts, {'balance': lambda doc: doc['balance'] + 100},
    ...                where('id') == 2)
    ...     txn.commit()
    >>>
    >>> # If any operation fails, all changes are rolled back automatically

The transaction system integrates with:
- Validation: All operations are validated before any writes occur
- Hooks: Before/after hooks are triggered appropriately
- Soft delete: Supports soft_remove, restore, and purge operations
"""

import time
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    TYPE_CHECKING,
)

from .hooks import HookEvent

if TYPE_CHECKING:
    from .database import TinyDB
    from .table import Table


class TransactionError(Exception):
    """
    Exception raised when a transaction fails.

    This exception provides details about which operation failed and why.
    When a transaction fails, all previously executed operations within the
    transaction are automatically rolled back.

    :param message: A descriptive error message
    :param operation: The operation that caused the failure (optional)
    :param original_error: The original exception that caused the failure (optional)
    """

    def __init__(
        self,
        message: str,
        operation: Optional['TransactionOperation'] = None,
        original_error: Optional[Exception] = None
    ):
        self.message = message
        self.operation = operation
        self.original_error = original_error
        super().__init__(message)


class TransactionOperation(ABC):
    """
    Abstract base class for transaction operations.

    Each operation knows how to:
    1. Validate itself before execution
    2. Execute and track what changes were made
    3. Provide information for rollback
    """

    def __init__(self, table: 'Table'):
        self.table = table
        self._executed = False

    @abstractmethod
    def validate(self) -> None:
        """
        Validate the operation before execution.

        :raises ValidationError: If validation fails
        :raises TransactionError: If the operation is invalid
        """
        pass

    @abstractmethod
    def execute(
        self,
        table_data: Dict[int, dict]
    ) -> Tuple[List[int], List[Dict], List[Dict]]:
        """
        Execute the operation on the table data.

        :param table_data: The current table data (mutable)
        :returns: Tuple of (affected_doc_ids, before_docs, after_docs)
                  for hook execution
        """
        pass

    @property
    def table_name(self) -> str:
        return self.table.name


class InsertOperation(TransactionOperation):
    """Insert one or more documents."""

    def __init__(self, table: 'Table', documents: List[Mapping]):
        super().__init__(table)
        self.documents = documents
        self.assigned_ids: List[int] = []

    def validate(self) -> None:
        for doc in self.documents:
            if not isinstance(doc, Mapping):
                raise TransactionError(
                    "Document is not a Mapping",
                    operation=self
                )
            self.table._validate_document(doc)

    def execute(
        self,
        table_data: Dict[int, dict]
    ) -> Tuple[List[int], List[Dict], List[Dict]]:
        from .table import CREATED_AT_KEY, Document

        before_docs: List[Dict] = []
        after_docs: List[Dict] = []
        current_time = time.time()

        for doc in self.documents:
            # Determine the doc_id
            if isinstance(doc, Document):
                doc_id = doc.doc_id
                if doc_id in table_data:
                    raise TransactionError(
                        f"Document with ID {doc_id} already exists",
                        operation=self
                    )
            else:
                # Find the next available ID
                doc_id = max(table_data.keys(), default=0) + 1
                while doc_id in table_data:
                    doc_id += 1

            self.assigned_ids.append(doc_id)

            # Create the document data with timestamp
            doc_data = dict(doc)
            doc_data[CREATED_AT_KEY] = current_time
            table_data[doc_id] = doc_data

            # Track for hooks
            after_docs.append({'doc_id': doc_id, **dict(doc)})

        self._executed = True
        return self.assigned_ids, before_docs, after_docs


class UpdateOperation(TransactionOperation):
    """Update documents matching a condition or by doc_ids."""

    def __init__(
        self,
        table: 'Table',
        fields: Union[Mapping, Callable[[Mapping], None]],
        cond: Optional[Any] = None,
        doc_ids: Optional[List[int]] = None
    ):
        super().__init__(table)
        self.fields = fields
        self.cond = cond
        self.doc_ids = list(doc_ids) if doc_ids else None
        self._is_callable = callable(fields)

    def validate(self) -> None:
        # Require either cond or doc_ids to prevent accidental mass updates
        if self.cond is None and self.doc_ids is None:
            raise TransactionError(
                "Update operation requires either cond or doc_ids to prevent "
                "accidental updates to all documents",
                operation=self
            )
        if not self._is_callable and isinstance(self.fields, Mapping):
            self.table._validate_update_fields(self.fields)

    def execute(
        self,
        table_data: Dict[int, dict]
    ) -> Tuple[List[int], List[Dict], List[Dict]]:
        from .table import UPDATED_AT_KEY, SOFT_DELETE_KEY

        before_docs: List[Dict] = []
        after_docs: List[Dict] = []
        updated_ids: List[int] = []
        current_time = time.time()

        # Determine which documents to update
        # Note: validate() ensures at least one of cond or doc_ids is provided
        if self.doc_ids is not None:
            target_ids = [did for did in self.doc_ids if did in table_data]
        else:
            # cond is guaranteed to be not None due to validate()
            target_ids = [
                did for did, doc in table_data.items()
                if not doc.get(SOFT_DELETE_KEY, False) and self.cond(doc)
            ]

        for doc_id in target_ids:
            if doc_id not in table_data:
                continue

            # Capture before state
            before_docs.append({'doc_id': doc_id, **dict(table_data[doc_id])})

            # Apply update
            if self._is_callable:
                # For callable, apply to a copy, validate, then update
                doc_copy = dict(table_data[doc_id])
                self.fields(doc_copy)
                if self.table._schema is not None:
                    self.table._schema.validate(doc_copy)
                table_data[doc_id].clear()
                table_data[doc_id].update(doc_copy)
            else:
                table_data[doc_id].update(self.fields)

            # Add update timestamp
            table_data[doc_id][UPDATED_AT_KEY] = current_time
            updated_ids.append(doc_id)

            # Capture after state
            after_docs.append({'doc_id': doc_id, **dict(table_data[doc_id])})

        self._executed = True
        return updated_ids, before_docs, after_docs


class DeleteOperation(TransactionOperation):
    """Hard delete documents matching a condition or by doc_ids."""

    def __init__(
        self,
        table: 'Table',
        cond: Optional[Any] = None,
        doc_ids: Optional[List[int]] = None
    ):
        super().__init__(table)
        self.cond = cond
        self.doc_ids = list(doc_ids) if doc_ids else None

    def validate(self) -> None:
        # No validation needed for delete
        pass

    def execute(
        self,
        table_data: Dict[int, dict]
    ) -> Tuple[List[int], List[Dict], List[Dict]]:
        before_docs: List[Dict] = []
        deleted_ids: List[int] = []

        # Determine which documents to delete
        if self.doc_ids is not None:
            target_ids = [did for did in self.doc_ids if did in table_data]
        elif self.cond is not None:
            target_ids = [
                did for did, doc in table_data.items()
                if self.cond(doc)
            ]
        else:
            raise TransactionError(
                "Delete operation requires either cond or doc_ids",
                operation=self
            )

        for doc_id in list(target_ids):  # Use list() to avoid mutation issues
            if doc_id in table_data:
                before_docs.append({'doc_id': doc_id, **dict(table_data[doc_id])})
                del table_data[doc_id]
                deleted_ids.append(doc_id)

        self._executed = True
        return deleted_ids, before_docs, []


class SoftDeleteOperation(TransactionOperation):
    """Soft delete documents (mark as deleted without removing)."""

    def __init__(
        self,
        table: 'Table',
        cond: Optional[Any] = None,
        doc_ids: Optional[List[int]] = None
    ):
        super().__init__(table)
        self.cond = cond
        self.doc_ids = list(doc_ids) if doc_ids else None

    def validate(self) -> None:
        # No validation needed for soft delete
        pass

    def execute(
        self,
        table_data: Dict[int, dict]
    ) -> Tuple[List[int], List[Dict], List[Dict]]:
        from .table import SOFT_DELETE_KEY, DELETED_AT_KEY

        before_docs: List[Dict] = []
        deleted_ids: List[int] = []
        current_time = time.time()

        # Determine which documents to soft delete
        if self.doc_ids is not None:
            target_ids = [did for did in self.doc_ids if did in table_data]
        elif self.cond is not None:
            target_ids = [
                did for did, doc in table_data.items()
                if not doc.get(SOFT_DELETE_KEY, False) and self.cond(doc)
            ]
        else:
            raise TransactionError(
                "Soft delete operation requires either cond or doc_ids",
                operation=self
            )

        for doc_id in target_ids:
            if doc_id in table_data:
                before_docs.append({'doc_id': doc_id, **dict(table_data[doc_id])})
                table_data[doc_id][SOFT_DELETE_KEY] = True
                table_data[doc_id][DELETED_AT_KEY] = current_time
                deleted_ids.append(doc_id)

        self._executed = True
        return deleted_ids, before_docs, []


class RestoreOperation(TransactionOperation):
    """Restore soft-deleted documents."""

    def __init__(
        self,
        table: 'Table',
        cond: Optional[Any] = None,
        doc_ids: Optional[List[int]] = None
    ):
        super().__init__(table)
        self.cond = cond
        self.doc_ids = list(doc_ids) if doc_ids else None

    def validate(self) -> None:
        # No validation needed for restore
        pass

    def execute(
        self,
        table_data: Dict[int, dict]
    ) -> Tuple[List[int], List[Dict], List[Dict]]:
        from .table import SOFT_DELETE_KEY

        after_docs: List[Dict] = []
        restored_ids: List[int] = []

        # Determine which documents to restore
        if self.doc_ids is not None:
            target_ids = [
                did for did in self.doc_ids
                if did in table_data and table_data[did].get(SOFT_DELETE_KEY, False)
            ]
        elif self.cond is not None:
            target_ids = [
                did for did, doc in table_data.items()
                if doc.get(SOFT_DELETE_KEY, False) and self.cond(
                    {k: v for k, v in doc.items() if k != SOFT_DELETE_KEY}
                )
            ]
        else:
            # Restore all soft-deleted
            target_ids = [
                did for did, doc in table_data.items()
                if doc.get(SOFT_DELETE_KEY, False)
            ]

        for doc_id in target_ids:
            if doc_id in table_data and table_data[doc_id].get(SOFT_DELETE_KEY, False):
                del table_data[doc_id][SOFT_DELETE_KEY]
                restored_ids.append(doc_id)
                # Capture for hooks (restore triggers INSERT hooks)
                after_docs.append({'doc_id': doc_id, **dict(table_data[doc_id])})

        self._executed = True
        return restored_ids, [], after_docs


class PurgeOperation(TransactionOperation):
    """Permanently remove soft-deleted documents."""

    def __init__(
        self,
        table: 'Table',
        cond: Optional[Any] = None,
        doc_ids: Optional[List[int]] = None
    ):
        super().__init__(table)
        self.cond = cond
        self.doc_ids = list(doc_ids) if doc_ids else None

    def validate(self) -> None:
        # No validation needed for purge
        pass

    def execute(
        self,
        table_data: Dict[int, dict]
    ) -> Tuple[List[int], List[Dict], List[Dict]]:
        from .table import SOFT_DELETE_KEY

        before_docs: List[Dict] = []
        purged_ids: List[int] = []

        # Determine which documents to purge
        if self.doc_ids is not None:
            target_ids = [
                did for did in self.doc_ids
                if did in table_data and table_data[did].get(SOFT_DELETE_KEY, False)
            ]
        elif self.cond is not None:
            target_ids = [
                did for did, doc in table_data.items()
                if doc.get(SOFT_DELETE_KEY, False) and self.cond(
                    {k: v for k, v in doc.items() if k != SOFT_DELETE_KEY}
                )
            ]
        else:
            # Purge all soft-deleted
            target_ids = [
                did for did, doc in table_data.items()
                if doc.get(SOFT_DELETE_KEY, False)
            ]

        for doc_id in list(target_ids):
            if doc_id in table_data and table_data[doc_id].get(SOFT_DELETE_KEY, False):
                before_docs.append({'doc_id': doc_id, **dict(table_data[doc_id])})
                del table_data[doc_id]
                purged_ids.append(doc_id)

        self._executed = True
        return purged_ids, before_docs, []


class Transaction:
    """
    Represents an atomic database transaction.

    A transaction groups multiple database operations together so they all
    succeed or fail as a unit. This ensures data consistency when performing
    related operations that must not be partially applied.

    Usage:

        >>> with db.transaction() as txn:
        ...     txn.insert(users_table, {'name': 'Alice'})
        ...     txn.update(stats_table, {'user_count': lambda d: d['user_count'] + 1})
        ...     txn.commit()

    If an error occurs during commit, all changes are automatically rolled back.
    If the context exits without calling commit(), the transaction is also rolled back.

    Transaction states:
    - PENDING: Transaction created, operations being added
    - COMMITTED: Transaction successfully committed
    - ROLLED_BACK: Transaction was rolled back (either explicitly or due to error)
    """

    def __init__(self, db: 'TinyDB'):
        """
        Create a new transaction.

        :param db: The TinyDB instance this transaction operates on
        """
        self._db = db
        self._operations: List[TransactionOperation] = []
        self._committed = False
        self._rolled_back = False
        self._tables_involved: Set[str] = set()
        self._results: Dict[str, List[int]] = {}

    @property
    def is_committed(self) -> bool:
        """Whether this transaction has been committed."""
        return self._committed

    @property
    def is_rolled_back(self) -> bool:
        """Whether this transaction has been rolled back."""
        return self._rolled_back

    @property
    def is_pending(self) -> bool:
        """Whether this transaction is still pending (not committed or rolled back)."""
        return not self._committed and not self._rolled_back

    def insert(
        self,
        table: 'Table',
        document: Union[Mapping, List[Mapping]]
    ) -> 'Transaction':
        """
        Queue an insert operation in this transaction.

        :param table: The table to insert into
        :param document: The document(s) to insert
        :returns: self for method chaining
        """
        self._ensure_pending()
        if isinstance(document, list):
            docs = document
        else:
            docs = [document]
        self._operations.append(InsertOperation(table, docs))
        self._tables_involved.add(table.name)
        return self

    def update(
        self,
        table: 'Table',
        fields: Union[Mapping, Callable[[Mapping], None]],
        cond: Optional[Any] = None,
        doc_ids: Optional[List[int]] = None
    ) -> 'Transaction':
        """
        Queue an update operation in this transaction.

        :param table: The table to update
        :param fields: The fields to update or a callable to modify documents
        :param cond: Optional query condition
        :param doc_ids: Optional list of document IDs
        :returns: self for method chaining
        """
        self._ensure_pending()
        self._operations.append(UpdateOperation(table, fields, cond, doc_ids))
        self._tables_involved.add(table.name)
        return self

    def remove(
        self,
        table: 'Table',
        cond: Optional[Any] = None,
        doc_ids: Optional[List[int]] = None
    ) -> 'Transaction':
        """
        Queue a delete operation in this transaction.

        :param table: The table to delete from
        :param cond: Optional query condition
        :param doc_ids: Optional list of document IDs
        :returns: self for method chaining
        """
        self._ensure_pending()
        self._operations.append(DeleteOperation(table, cond, doc_ids))
        self._tables_involved.add(table.name)
        return self

    def soft_remove(
        self,
        table: 'Table',
        cond: Optional[Any] = None,
        doc_ids: Optional[List[int]] = None
    ) -> 'Transaction':
        """
        Queue a soft delete operation in this transaction.

        :param table: The table to soft delete from
        :param cond: Optional query condition
        :param doc_ids: Optional list of document IDs
        :returns: self for method chaining
        """
        self._ensure_pending()
        self._operations.append(SoftDeleteOperation(table, cond, doc_ids))
        self._tables_involved.add(table.name)
        return self

    def restore(
        self,
        table: 'Table',
        cond: Optional[Any] = None,
        doc_ids: Optional[List[int]] = None
    ) -> 'Transaction':
        """
        Queue a restore operation in this transaction.

        :param table: The table to restore documents in
        :param cond: Optional query condition
        :param doc_ids: Optional list of document IDs
        :returns: self for method chaining
        """
        self._ensure_pending()
        self._operations.append(RestoreOperation(table, cond, doc_ids))
        self._tables_involved.add(table.name)
        return self

    def purge(
        self,
        table: 'Table',
        cond: Optional[Any] = None,
        doc_ids: Optional[List[int]] = None
    ) -> 'Transaction':
        """
        Queue a purge operation (permanently delete soft-deleted documents).

        :param table: The table to purge from
        :param cond: Optional query condition
        :param doc_ids: Optional list of document IDs
        :returns: self for method chaining
        """
        self._ensure_pending()
        self._operations.append(PurgeOperation(table, cond, doc_ids))
        self._tables_involved.add(table.name)
        return self

    def _ensure_pending(self) -> None:
        """Ensure the transaction is still pending."""
        if self._committed:
            raise TransactionError("Transaction has already been committed")
        if self._rolled_back:
            raise TransactionError("Transaction has already been rolled back")

    def commit(self) -> Dict[str, List[int]]:
        """
        Commit the transaction, executing all queued operations atomically.

        All operations are first validated, then executed together. If any
        operation fails during execution, all changes are rolled back.

        :returns: A dictionary mapping operation types to lists of affected doc IDs
        :raises TransactionError: If the transaction fails
        :raises ValidationError: If any operation fails validation
        """
        self._ensure_pending()

        if not self._operations:
            self._committed = True
            return {}

        # Phase 1: Validate all operations
        for op in self._operations:
            try:
                op.validate()
            except Exception as e:
                self._rolled_back = True
                if isinstance(e, TransactionError):
                    raise
                raise TransactionError(
                    f"Validation failed: {e}",
                    operation=op,
                    original_error=e
                )

        # Phase 2: Read current database state
        storage = self._db.storage
        tables = storage.read()
        if tables is None:
            tables = {}

        # Keep a backup of original data for rollback
        original_tables = {
            name: {
                doc_id: dict(doc)
                for doc_id, doc in table_data.items()
            }
            for name, table_data in tables.items()
        }

        # Convert to working format (int keys)
        working_tables: Dict[str, Dict[int, dict]] = {}
        for table_name in self._tables_involved:
            if table_name in tables:
                working_tables[table_name] = {
                    int(doc_id): dict(doc)
                    for doc_id, doc in tables[table_name].items()
                }
            else:
                working_tables[table_name] = {}

        # Phase 3: Execute all operations
        hook_events: List[Tuple[HookEvent, str, List[Dict]]] = []
        results: Dict[str, List[int]] = {
            'inserted': [],
            'updated': [],
            'deleted': [],
            'soft_deleted': [],
            'restored': [],
            'purged': []
        }

        try:
            for op in self._operations:
                table_name = op.table_name
                table_data = working_tables[table_name]

                affected_ids, before_docs, after_docs = op.execute(table_data)

                # Track results and hook events based on operation type
                if isinstance(op, InsertOperation):
                    results['inserted'].extend(affected_ids)
                    if before_docs:
                        hook_events.append((HookEvent.BEFORE_INSERT, table_name, before_docs))
                    if after_docs:
                        hook_events.append((HookEvent.AFTER_INSERT, table_name, after_docs))

                elif isinstance(op, UpdateOperation):
                    results['updated'].extend(affected_ids)
                    if before_docs:
                        hook_events.append((HookEvent.BEFORE_UPDATE, table_name, before_docs))
                    if after_docs:
                        hook_events.append((HookEvent.AFTER_UPDATE, table_name, after_docs))

                elif isinstance(op, DeleteOperation):
                    results['deleted'].extend(affected_ids)
                    if before_docs:
                        # BEFORE_DELETE receives the documents that are about to be deleted,
                        # allowing hooks to access the data before it's removed.
                        hook_events.append((HookEvent.BEFORE_DELETE, table_name, before_docs))
                        # AFTER_DELETE receives an empty list because the documents no longer
                        # exist after a hard delete. This is intentional - there's no document
                        # data to provide after deletion. Hooks that need the deleted data
                        # should use BEFORE_DELETE instead.
                        hook_events.append((HookEvent.AFTER_DELETE, table_name, []))

                elif isinstance(op, SoftDeleteOperation):
                    results['soft_deleted'].extend(affected_ids)
                    if before_docs:
                        # BEFORE_DELETE receives the documents before they are marked as deleted.
                        hook_events.append((HookEvent.BEFORE_DELETE, table_name, before_docs))
                        # AFTER_DELETE receives an empty list for consistency with hard delete.
                        # Even though soft-deleted documents still exist in storage, they are
                        # logically "deleted" from the user's perspective. Hooks needing the
                        # deleted document data should use BEFORE_DELETE.
                        hook_events.append((HookEvent.AFTER_DELETE, table_name, []))

                elif isinstance(op, RestoreOperation):
                    results['restored'].extend(affected_ids)
                    if after_docs:
                        hook_events.append((HookEvent.BEFORE_INSERT, table_name, after_docs))
                        hook_events.append((HookEvent.AFTER_INSERT, table_name, after_docs))

                elif isinstance(op, PurgeOperation):
                    results['purged'].extend(affected_ids)
                    if before_docs:
                        # BEFORE_DELETE receives the soft-deleted documents before purging.
                        hook_events.append((HookEvent.BEFORE_DELETE, table_name, before_docs))
                        # AFTER_DELETE receives an empty list because purge permanently removes
                        # documents from storage. Same rationale as DeleteOperation.
                        hook_events.append((HookEvent.AFTER_DELETE, table_name, []))

        except Exception as e:
            # Rollback: restore original tables
            self._rolled_back = True
            if isinstance(e, TransactionError):
                raise
            raise TransactionError(
                f"Execution failed: {e}",
                original_error=e
            )

        # Phase 4: Write updated data back to storage
        try:
            # Merge working tables back into tables
            for table_name, table_data in working_tables.items():
                tables[table_name] = {
                    str(doc_id): doc
                    for doc_id, doc in table_data.items()
                }

            storage.write(tables)

        except Exception as e:
            # Write failed - data wasn't actually changed
            self._rolled_back = True
            raise TransactionError(
                f"Failed to write transaction: {e}",
                original_error=e
            )

        # Phase 5: Clear caches for affected tables
        for table_name in self._tables_involved:
            if table_name in self._db._tables:
                self._db._tables[table_name].clear_cache()
                # Reset next_id since documents may have been added/removed
                self._db._tables[table_name]._next_id = None

        # Phase 6: Run hooks (after successful commit)
        hooks = self._db._hooks
        for event, table_name, docs in hook_events:
            if hooks.has_hooks(event):
                hooks.run(event, table_name, docs)

        self._committed = True
        self._results = results
        return results

    def rollback(self) -> None:
        """
        Explicitly roll back the transaction.

        This discards all queued operations without executing them.
        After rollback, the transaction cannot be committed.
        """
        self._ensure_pending()
        self._rolled_back = True
        self._operations.clear()

    def __enter__(self) -> 'Transaction':
        """Enter the transaction context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the transaction context.

        If an exception occurred or commit was not called, the transaction
        is rolled back.
        """
        if exc_type is not None:
            # An exception occurred - ensure rollback
            if self.is_pending:
                self._rolled_back = True
        elif self.is_pending:
            # No exception but commit wasn't called - rollback
            self._rolled_back = True


__all__ = (
    'Transaction',
    'TransactionError',
    'TransactionOperation',
    'InsertOperation',
    'UpdateOperation',
    'DeleteOperation',
    'SoftDeleteOperation',
    'RestoreOperation',
    'PurgeOperation',
)
