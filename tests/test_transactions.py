"""
Tests for TinyDB transactions.

These tests verify that transactions work correctly with:
- Basic CRUD operations
- Atomic commit/rollback behavior
- Validation integration
- Hook integration
- Soft delete operations
"""

import pytest

from tinydb import TinyDB, where, Transaction, TransactionError
from tinydb.storages import MemoryStorage
from tinydb.hooks import HookEvent
from tinydb.validation import Schema, FieldValidator, ValidationError


@pytest.fixture
def db():
    """Create a fresh in-memory TinyDB instance."""
    return TinyDB(storage=MemoryStorage)


@pytest.fixture
def populated_db():
    """Create a TinyDB instance with initial data."""
    db = TinyDB(storage=MemoryStorage)
    table = db.table('accounts')
    table.insert({'name': 'Alice', 'balance': 1000})
    table.insert({'name': 'Bob', 'balance': 500})
    table.insert({'name': 'Charlie', 'balance': 750})
    return db


@pytest.fixture
def delete_hook_test_setup():
    """
    Fixture for testing AFTER_DELETE hook behavior across different delete operations.

    Sets up a database with a test document and registers BEFORE_DELETE and
    AFTER_DELETE hooks that capture the documents passed to each hook.

    Returns a dict with:
    - db: The TinyDB instance
    - table: The test table with one document (doc_id=1, name='Alice', value=1)
    - before_docs: List that captures documents passed to BEFORE_DELETE
    - after_docs: List that captures documents passed to AFTER_DELETE

    This fixture eliminates code duplication in tests that verify AFTER_DELETE
    receives an empty list for different delete operations (hard delete, soft
    delete, purge).
    """
    db = TinyDB(storage=MemoryStorage)
    table = db.table('test')
    table.insert({'name': 'Alice', 'value': 1})

    before_docs = []
    after_docs = []

    def on_before_delete(table_name, event, documents):
        before_docs.extend(documents)

    def on_after_delete(table_name, event, documents):
        after_docs.extend(documents)

    db.hooks.register(HookEvent.BEFORE_DELETE, on_before_delete)
    db.hooks.register(HookEvent.AFTER_DELETE, on_after_delete)

    return {
        'db': db,
        'table': table,
        'before_docs': before_docs,
        'after_docs': after_docs,
    }


class TestTransactionBasics:
    """Test basic transaction functionality."""

    def test_create_transaction(self, db):
        """Test creating a transaction."""
        txn = db.transaction()
        assert isinstance(txn, Transaction)
        assert txn.is_pending
        assert not txn.is_committed
        assert not txn.is_rolled_back

    def test_transaction_context_manager(self, db):
        """Test using transaction as context manager."""
        with db.transaction() as txn:
            assert txn.is_pending
            txn.commit()
        assert txn.is_committed

    def test_transaction_context_manager_rollback_on_exit(self, db):
        """Test that uncommitted transactions are rolled back on context exit."""
        with db.transaction() as txn:
            txn.insert(db.table('test'), {'value': 1})
            # Don't commit
        assert txn.is_rolled_back
        # Data should not be written
        assert len(db.table('test')) == 0

    def test_transaction_context_manager_rollback_on_exception(self, db):
        """Test that transactions are rolled back on exception."""
        try:
            with db.transaction() as txn:
                txn.insert(db.table('test'), {'value': 1})
                raise ValueError("Test error")
        except ValueError:
            pass
        assert txn.is_rolled_back
        assert len(db.table('test')) == 0

    def test_empty_transaction_commit(self, db):
        """Test committing an empty transaction."""
        with db.transaction() as txn:
            result = txn.commit()
        assert txn.is_committed
        assert result == {}


class TestTransactionInsert:
    """Test transaction insert operations."""

    def test_insert_single_document(self, db):
        """Test inserting a single document in a transaction."""
        table = db.table('test')
        with db.transaction() as txn:
            txn.insert(table, {'name': 'Test'})
            result = txn.commit()

        assert 'inserted' in result
        assert len(result['inserted']) == 1
        assert len(table) == 1
        assert table.all()[0]['name'] == 'Test'

    def test_insert_multiple_documents(self, db):
        """Test inserting multiple documents in a transaction."""
        table = db.table('test')
        with db.transaction() as txn:
            txn.insert(table, [{'name': 'A'}, {'name': 'B'}, {'name': 'C'}])
            result = txn.commit()

        assert len(result['inserted']) == 3
        assert len(table) == 3

    def test_insert_to_multiple_tables(self, db):
        """Test inserting to multiple tables in a transaction."""
        users = db.table('users')
        logs = db.table('logs')

        with db.transaction() as txn:
            txn.insert(users, {'name': 'Alice'})
            txn.insert(logs, {'action': 'user_created', 'user': 'Alice'})
            txn.commit()

        assert len(users) == 1
        assert len(logs) == 1


class TestTransactionUpdate:
    """Test transaction update operations."""

    def test_update_by_condition(self, populated_db):
        """Test updating documents by condition."""
        table = populated_db.table('accounts')
        original_balance = table.search(where('name') == 'Alice')[0]['balance']

        with populated_db.transaction() as txn:
            txn.update(table, {'balance': original_balance + 100}, where('name') == 'Alice')
            txn.commit()

        alice = table.search(where('name') == 'Alice')[0]
        assert alice['balance'] == 1100

    def test_update_by_doc_ids(self, populated_db):
        """Test updating documents by doc_ids."""
        table = populated_db.table('accounts')

        with populated_db.transaction() as txn:
            txn.update(table, {'verified': True}, doc_ids=[1, 2])
            txn.commit()

        doc1 = table.get(doc_id=1)
        doc2 = table.get(doc_id=2)
        doc3 = table.get(doc_id=3)

        assert doc1['verified'] is True
        assert doc2['verified'] is True
        assert 'verified' not in doc3

    def test_update_with_callable(self, populated_db):
        """Test updating documents with a callable."""
        table = populated_db.table('accounts')

        def add_interest(doc):
            doc['balance'] = int(doc['balance'] * 1.05)

        with populated_db.transaction() as txn:
            txn.update(table, add_interest, where('name') == 'Alice')
            txn.commit()

        alice = table.search(where('name') == 'Alice')[0]
        assert alice['balance'] == 1050  # 1000 * 1.05


class TestTransactionDelete:
    """Test transaction delete operations."""

    def test_delete_by_condition(self, populated_db):
        """Test deleting documents by condition."""
        table = populated_db.table('accounts')
        initial_count = len(table)

        with populated_db.transaction() as txn:
            txn.remove(table, where('balance') < 600)
            result = txn.commit()

        assert len(table) == initial_count - 1
        assert len(result['deleted']) == 1
        assert not table.search(where('name') == 'Bob')

    def test_delete_by_doc_ids(self, populated_db):
        """Test deleting documents by doc_ids."""
        table = populated_db.table('accounts')

        with populated_db.transaction() as txn:
            txn.remove(table, doc_ids=[1])
            txn.commit()

        assert table.get(doc_id=1) is None
        assert len(table) == 2


class TestTransactionSoftDelete:
    """Test transaction soft delete operations."""

    def test_soft_remove(self, populated_db):
        """Test soft deleting documents in a transaction."""
        table = populated_db.table('accounts')

        with populated_db.transaction() as txn:
            txn.soft_remove(table, where('name') == 'Bob')
            result = txn.commit()

        assert len(result['soft_deleted']) == 1
        # Document should not appear in normal queries
        assert len(table) == 2
        # But should be visible with include_deleted
        assert len(table.all(include_deleted=True)) == 3

    def test_restore(self, populated_db):
        """Test restoring soft-deleted documents in a transaction."""
        table = populated_db.table('accounts')
        table.soft_remove(where('name') == 'Bob')

        with populated_db.transaction() as txn:
            txn.restore(table, doc_ids=[2])
            result = txn.commit()

        assert len(result['restored']) == 1
        assert len(table) == 3  # All documents visible again

    def test_purge(self, populated_db):
        """Test purging soft-deleted documents in a transaction."""
        table = populated_db.table('accounts')
        table.soft_remove(where('name') == 'Bob')

        with populated_db.transaction() as txn:
            txn.purge(table, doc_ids=[2])
            result = txn.commit()

        assert len(result['purged']) == 1
        assert len(table.all(include_deleted=True)) == 2


class TestTransactionAtomicity:
    """Test transaction atomicity - all or nothing behavior."""

    def test_money_transfer_success(self, populated_db):
        """Test successful money transfer between accounts."""
        accounts = populated_db.table('accounts')

        with populated_db.transaction() as txn:
            # Debit Alice
            txn.update(accounts, {'balance': 900}, where('name') == 'Alice')
            # Credit Bob
            txn.update(accounts, {'balance': 600}, where('name') == 'Bob')
            txn.commit()

        alice = accounts.search(where('name') == 'Alice')[0]
        bob = accounts.search(where('name') == 'Bob')[0]

        assert alice['balance'] == 900
        assert bob['balance'] == 600

    def test_rollback_on_validation_failure(self, db):
        """Test that all operations are rolled back on validation failure."""
        table = db.table('test')
        table.set_schema(Schema({
            'name': FieldValidator(required=True, field_type=str),
            'age': FieldValidator(field_type=int)
        }))

        # First insert a valid document
        table.insert({'name': 'Alice', 'age': 30})

        with pytest.raises(TransactionError) as exc_info:
            with db.transaction() as txn:
                # Valid operation
                txn.insert(table, {'name': 'Bob', 'age': 25})
                # Invalid operation - wrong type for age
                txn.insert(table, {'name': 'Charlie', 'age': 'not a number'})
                txn.commit()

        assert 'Validation failed' in str(exc_info.value)
        # Only the original document should exist
        assert len(table) == 1
        assert table.all()[0]['name'] == 'Alice'

    def test_rollback_preserves_original_data(self, populated_db):
        """Test that rollback preserves original data completely."""
        accounts = populated_db.table('accounts')

        # Capture original state
        original_alice = accounts.search(where('name') == 'Alice')[0]['balance']
        original_bob = accounts.search(where('name') == 'Bob')[0]['balance']

        # Transaction that will be rolled back
        with populated_db.transaction() as txn:
            txn.update(accounts, {'balance': 0}, where('name') == 'Alice')
            txn.update(accounts, {'balance': 0}, where('name') == 'Bob')
            txn.rollback()

        # Verify original data is preserved
        alice = accounts.search(where('name') == 'Alice')[0]
        bob = accounts.search(where('name') == 'Bob')[0]

        assert alice['balance'] == original_alice
        assert bob['balance'] == original_bob

    def test_cannot_commit_after_rollback(self, db):
        """Test that commit fails after rollback."""
        table = db.table('test')

        with pytest.raises(TransactionError):
            txn = db.transaction()
            txn.insert(table, {'value': 1})
            txn.rollback()
            txn.commit()

    def test_cannot_commit_twice(self, db):
        """Test that commit fails after already committed."""
        table = db.table('test')

        with pytest.raises(TransactionError):
            txn = db.transaction()
            txn.insert(table, {'value': 1})
            txn.commit()
            txn.commit()

    def test_cannot_add_operations_after_commit(self, db):
        """Test that adding operations after commit fails."""
        table = db.table('test')

        with pytest.raises(TransactionError):
            txn = db.transaction()
            txn.commit()
            txn.insert(table, {'value': 1})


class TestTransactionValidation:
    """Test transaction validation integration."""

    def test_update_requires_cond_or_doc_ids(self, db):
        """Test that update operation requires either cond or doc_ids.

        This prevents accidental mass updates to all documents.
        """
        table = db.table('test')
        table.insert({'name': 'Alice', 'value': 1})
        table.insert({'name': 'Bob', 'value': 2})

        with pytest.raises(TransactionError) as exc_info:
            with db.transaction() as txn:
                # Attempting to update without cond or doc_ids
                txn.update(table, {'value': 999})
                txn.commit()

        assert 'requires either cond or doc_ids' in str(exc_info.value)

        # Verify no documents were changed
        assert table.get(doc_id=1)['value'] == 1
        assert table.get(doc_id=2)['value'] == 2

    def test_validates_insert_against_schema(self, db):
        """Test that insert operations are validated against schema."""
        table = db.table('users')
        table.set_schema(Schema({
            'email': FieldValidator(required=True, field_type=str)
        }))

        with pytest.raises(TransactionError) as exc_info:
            with db.transaction() as txn:
                txn.insert(table, {'name': 'Alice'})  # Missing email
                txn.commit()

        assert 'Validation failed' in str(exc_info.value)
        assert len(table) == 0

    def test_validates_update_fields(self, db):
        """Test that update operations validate field types."""
        table = db.table('users')
        table.insert({'name': 'Alice', 'age': 30})
        table.set_schema(Schema({
            'name': FieldValidator(field_type=str),
            'age': FieldValidator(field_type=int)
        }))

        with pytest.raises(TransactionError):
            with db.transaction() as txn:
                txn.update(table, {'age': 'thirty'}, doc_ids=[1])
                txn.commit()

        # Original should be unchanged
        assert table.get(doc_id=1)['age'] == 30

    def test_validates_callable_update_result(self, db):
        """Test that callable update results are validated."""
        table = db.table('users')
        table.insert({'name': 'Alice', 'score': 100})
        table.set_schema(Schema({
            'name': FieldValidator(field_type=str),
            'score': FieldValidator(field_type=int)
        }))

        def bad_update(doc):
            doc['score'] = 'not a number'

        with pytest.raises(TransactionError):
            with db.transaction() as txn:
                txn.update(table, bad_update, doc_ids=[1])
                txn.commit()

        # Original should be unchanged
        assert table.get(doc_id=1)['score'] == 100


class TestTransactionHooks:
    """Test transaction hook integration."""

    def test_hooks_triggered_on_commit(self, db):
        """Test that hooks are triggered after successful commit."""
        table = db.table('test')
        hook_calls = []

        def on_insert(table_name, event, documents):
            hook_calls.append(('insert', table_name, len(documents)))

        db.hooks.register(HookEvent.AFTER_INSERT, on_insert)

        with db.transaction() as txn:
            txn.insert(table, [{'a': 1}, {'a': 2}])
            txn.commit()

        assert len(hook_calls) == 1
        assert hook_calls[0] == ('insert', 'test', 2)

    def test_hooks_not_triggered_on_rollback(self, db):
        """Test that hooks are not triggered when transaction is rolled back."""
        table = db.table('test')
        hook_calls = []

        def on_insert(table_name, event, documents):
            hook_calls.append(('insert', table_name, len(documents)))

        db.hooks.register(HookEvent.AFTER_INSERT, on_insert)

        with db.transaction() as txn:
            txn.insert(table, {'a': 1})
            txn.rollback()

        assert len(hook_calls) == 0

    def test_update_hooks_triggered(self, db):
        """Test that update hooks are triggered."""
        table = db.table('test')
        table.insert({'value': 1})

        before_calls = []
        after_calls = []

        def on_before_update(table_name, event, documents):
            before_calls.append(documents)

        def on_after_update(table_name, event, documents):
            after_calls.append(documents)

        db.hooks.register(HookEvent.BEFORE_UPDATE, on_before_update)
        db.hooks.register(HookEvent.AFTER_UPDATE, on_after_update)

        with db.transaction() as txn:
            txn.update(table, {'value': 2}, doc_ids=[1])
            txn.commit()

        assert len(before_calls) == 1
        assert len(after_calls) == 1
        assert before_calls[0][0]['value'] == 1  # Before value
        assert after_calls[0][0]['value'] == 2  # After value

    def test_delete_hooks_triggered(self, db):
        """Test that delete hooks are triggered."""
        table = db.table('test')
        table.insert({'value': 1})

        hook_calls = []

        def on_delete(table_name, event, documents):
            hook_calls.append((event, documents))

        db.hooks.register(HookEvent.BEFORE_DELETE, on_delete)
        db.hooks.register(HookEvent.AFTER_DELETE, on_delete)

        with db.transaction() as txn:
            txn.remove(table, doc_ids=[1])
            txn.commit()

        assert len(hook_calls) == 2  # Before and after

    def test_after_delete_receives_empty_list_for_hard_delete(self, delete_hook_test_setup):
        """Test that AFTER_DELETE hook receives empty list for hard delete.

        After a hard delete, documents no longer exist, so AFTER_DELETE
        should receive an empty list. Hooks needing the deleted data
        should use BEFORE_DELETE instead.
        """
        db = delete_hook_test_setup['db']
        table = delete_hook_test_setup['table']
        before_docs = delete_hook_test_setup['before_docs']
        after_docs = delete_hook_test_setup['after_docs']

        with db.transaction() as txn:
            txn.remove(table, doc_ids=[1])
            txn.commit()

        # BEFORE_DELETE should have received the document
        assert len(before_docs) == 1
        assert before_docs[0]['name'] == 'Alice'

        # AFTER_DELETE should receive empty list (documents no longer exist)
        assert len(after_docs) == 0

    def test_after_delete_receives_empty_list_for_soft_delete(self, delete_hook_test_setup):
        """Test that AFTER_DELETE hook receives empty list for soft delete.

        For consistency with hard delete, AFTER_DELETE receives an empty
        list even for soft deletes. The documents are logically deleted
        from the user's perspective.
        """
        db = delete_hook_test_setup['db']
        table = delete_hook_test_setup['table']
        before_docs = delete_hook_test_setup['before_docs']
        after_docs = delete_hook_test_setup['after_docs']

        with db.transaction() as txn:
            txn.soft_remove(table, doc_ids=[1])
            txn.commit()

        # BEFORE_DELETE should have received the document
        assert len(before_docs) == 1
        assert before_docs[0]['name'] == 'Alice'

        # AFTER_DELETE should receive empty list for consistency
        assert len(after_docs) == 0

    def test_after_delete_receives_empty_list_for_purge(self, delete_hook_test_setup):
        """Test that AFTER_DELETE hook receives empty list for purge.

        Purge permanently removes soft-deleted documents, so AFTER_DELETE
        should receive an empty list.
        """
        db = delete_hook_test_setup['db']
        table = delete_hook_test_setup['table']
        before_docs = delete_hook_test_setup['before_docs']
        after_docs = delete_hook_test_setup['after_docs']

        # Soft delete first (required before purge)
        # This triggers hooks, so we clear the lists afterwards to isolate purge behavior
        table.soft_remove(doc_ids=[1])
        before_docs.clear()
        after_docs.clear()

        with db.transaction() as txn:
            txn.purge(table, doc_ids=[1])
            txn.commit()

        # BEFORE_DELETE should have received the document
        assert len(before_docs) == 1
        assert before_docs[0]['name'] == 'Alice'

        # AFTER_DELETE should receive empty list
        assert len(after_docs) == 0


class TestTransactionChaining:
    """Test method chaining in transactions."""

    def test_method_chaining(self, db):
        """Test that transaction methods can be chained."""
        users = db.table('users')
        logs = db.table('logs')

        result = (
            db.transaction()
            .insert(users, {'name': 'Alice'})
            .insert(users, {'name': 'Bob'})
            .insert(logs, {'action': 'users_created'})
            .commit()
        )

        assert len(users) == 2
        assert len(logs) == 1

    def test_chain_with_different_operations(self, populated_db):
        """Test chaining different operation types."""
        accounts = populated_db.table('accounts')

        (
            populated_db.transaction()
            .insert(accounts, {'name': 'Dave', 'balance': 200})
            .update(accounts, {'active': True}, where('balance') >= 0)  # Update all with condition
            .soft_remove(accounts, where('balance') < 300)
            .commit()
        )

        assert len(accounts) == 3  # 4 total - 1 soft deleted
        all_docs = accounts.all(include_deleted=True)
        for doc in all_docs:
            if doc['name'] != 'Dave' and doc['balance'] < 300:
                # This would be in the deleted set
                pass
            else:
                assert doc.get('active') is True


class TestTransactionMultiTable:
    """Test transactions across multiple tables."""

    def test_multiple_tables_atomicity(self, db):
        """Test that multi-table transactions are atomic."""
        users = db.table('users')
        balances = db.table('balances')

        users.set_schema(Schema({
            'name': FieldValidator(required=True, field_type=str)
        }))

        # First, a successful transaction
        with db.transaction() as txn:
            txn.insert(users, {'name': 'Alice'})
            txn.insert(balances, {'user_id': 1, 'amount': 100})
            txn.commit()

        assert len(users) == 1
        assert len(balances) == 1

        # Now a failing transaction - should rollback both tables
        with pytest.raises(TransactionError):
            with db.transaction() as txn:
                txn.insert(users, {'name': 'Bob'})
                txn.insert(balances, {'user_id': 2, 'amount': 200})
                # This will fail validation
                txn.insert(users, {'invalid': 'document'})  # Missing name
                txn.commit()

        # Both tables should be unchanged
        assert len(users) == 1
        assert len(balances) == 1

    def test_multiple_tables_update(self, db):
        """Test updating multiple tables in one transaction."""
        orders = db.table('orders')
        inventory = db.table('inventory')

        orders.insert({'product': 'Widget', 'quantity': 0})
        inventory.insert({'product': 'Widget', 'stock': 100})

        with db.transaction() as txn:
            txn.update(orders, {'quantity': 10}, where('product') == 'Widget')
            txn.update(inventory, {'stock': 90}, where('product') == 'Widget')
            txn.commit()

        order = orders.all()[0]
        inv = inventory.all()[0]

        assert order['quantity'] == 10
        assert inv['stock'] == 90


class TestTransactionEdgeCases:
    """Test edge cases and error handling."""

    def test_transaction_with_non_existent_doc_ids(self, db):
        """Test updating/deleting non-existent doc_ids."""
        table = db.table('test')
        table.insert({'value': 1})

        with db.transaction() as txn:
            txn.update(table, {'value': 2}, doc_ids=[999])  # Non-existent
            result = txn.commit()

        # No error, but nothing updated
        assert len(result['updated']) == 0
        assert table.get(doc_id=1)['value'] == 1

    def test_transaction_with_empty_result_condition(self, db):
        """Test operations with conditions that match nothing."""
        table = db.table('test')
        table.insert({'value': 1})

        with db.transaction() as txn:
            txn.remove(table, where('value') == 999)  # Won't match anything
            result = txn.commit()

        assert len(result['deleted']) == 0
        assert len(table) == 1

    def test_insert_generates_correct_ids(self, db):
        """Test that transaction inserts generate correct IDs."""
        table = db.table('test')
        table.insert({'a': 1})  # ID 1
        table.insert({'a': 2})  # ID 2
        table.remove(doc_ids=[1])  # Delete ID 1

        with db.transaction() as txn:
            txn.insert(table, [{'b': 1}, {'b': 2}])
            result = txn.commit()

        # IDs should be 3 and 4 (not reusing 1)
        assert 3 in result['inserted']
        assert 4 in result['inserted']
        assert table.get(doc_id=3) is not None
        assert table.get(doc_id=4) is not None

    def test_transaction_cache_clearing(self, db):
        """Test that transaction commit clears query cache."""
        table = db.table('test')
        table.insert({'type': 'a', 'value': 1})
        table.insert({'type': 'a', 'value': 2})

        # Populate cache
        result1 = table.search(where('type') == 'a')
        assert len(result1) == 2

        # Update via transaction
        with db.transaction() as txn:
            txn.insert(table, {'type': 'a', 'value': 3})
            txn.commit()

        # Cache should be cleared, new search should return 3 items
        result2 = table.search(where('type') == 'a')
        assert len(result2) == 3

    def test_transaction_preserves_timestamps(self, db):
        """Test that transaction operations add appropriate timestamps."""
        table = db.table('test')

        with db.transaction() as txn:
            txn.insert(table, {'name': 'Test'})
            txn.commit()

        doc = table.all(include_metadata=True)[0]
        assert '_created_at' in doc

        with db.transaction() as txn:
            txn.update(table, {'name': 'Updated'}, doc_ids=[1])
            txn.commit()

        doc = table.all(include_metadata=True)[0]
        assert '_updated_at' in doc


class TestTransactionResults:
    """Test transaction result reporting."""

    def test_commit_returns_all_affected_ids(self, db):
        """Test that commit returns all affected document IDs."""
        table = db.table('test')

        with db.transaction() as txn:
            txn.insert(table, [{'a': 1}, {'a': 2}])
            txn.insert(table, {'a': 3})
            result = txn.commit()

        assert len(result['inserted']) == 3
        assert 1 in result['inserted']
        assert 2 in result['inserted']
        assert 3 in result['inserted']

    def test_mixed_operations_results(self, populated_db):
        """Test results for mixed operations."""
        accounts = populated_db.table('accounts')

        with populated_db.transaction() as txn:
            txn.insert(accounts, {'name': 'Dave', 'balance': 100})
            txn.update(accounts, {'verified': True}, where('name') == 'Alice')
            txn.remove(accounts, where('name') == 'Charlie')
            result = txn.commit()

        assert len(result['inserted']) == 1
        assert len(result['updated']) == 1
        assert len(result['deleted']) == 1


class TestTransactionIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self, db):
        """Test a complete workflow with validation, hooks, and transactions."""
        users = db.table('users')
        users.set_schema(Schema({
            'name': FieldValidator(required=True, field_type=str),
            'email': FieldValidator(required=True, field_type=str)
        }))

        events = []

        def track_insert(table_name, event, documents):
            events.append(('insert', [d['name'] for d in documents]))

        def track_delete(table_name, event, documents):
            events.append(('delete', [d['name'] for d in documents]))

        db.hooks.register(HookEvent.AFTER_INSERT, track_insert)
        db.hooks.register(HookEvent.AFTER_DELETE, track_delete)

        # Successful transaction
        with db.transaction() as txn:
            txn.insert(users, {'name': 'Alice', 'email': 'alice@test.com'})
            txn.insert(users, {'name': 'Bob', 'email': 'bob@test.com'})
            txn.commit()

        assert len(users) == 2
        assert ('insert', ['Alice']) in events or ('insert', ['Alice', 'Bob']) in events

        # Another successful transaction with soft delete
        with db.transaction() as txn:
            txn.soft_remove(users, where('name') == 'Bob')
            txn.insert(users, {'name': 'Charlie', 'email': 'charlie@test.com'})
            txn.commit()

        assert len(users) == 2  # Alice + Charlie (Bob soft deleted)
        deleted = users.deleted()
        assert len(deleted) == 1
        assert deleted[0]['name'] == 'Bob'

    def test_concurrent_transactions_on_same_db(self, db):
        """Test that multiple transaction objects on same db work correctly."""
        table = db.table('test')
        table.insert({'counter': 0})

        # Create two transactions
        txn1 = db.transaction()
        txn2 = db.transaction()

        # Queue operations on both
        txn1.update(table, {'counter': 1}, doc_ids=[1])
        txn2.update(table, {'counter': 2}, doc_ids=[1])

        # Commit txn1 first
        txn1.commit()
        assert table.get(doc_id=1)['counter'] == 1

        # txn2 should work but overwrite
        txn2.commit()
        assert table.get(doc_id=1)['counter'] == 2
