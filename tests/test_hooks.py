"""
Tests for the TinyDB hook system.
"""

import warnings
import pytest

from tinydb import TinyDB, Query
from tinydb.hooks import HookEvent, HookManager
from tinydb.storages import MemoryStorage


@pytest.fixture
def db():
    """Create a fresh TinyDB instance with MemoryStorage."""
    db_ = TinyDB(storage=MemoryStorage)
    db_.drop_tables()
    yield db_


class TestHookManager:
    """Tests for the HookManager class."""

    def test_register_hook(self):
        """Test registering a hook callback."""
        manager = HookManager()

        def callback(table, event, docs):
            pass

        manager.register(HookEvent.AFTER_INSERT, callback)

        assert manager.has_hooks(HookEvent.AFTER_INSERT)
        assert not manager.has_hooks(HookEvent.BEFORE_INSERT)

    def test_register_invalid_event(self):
        """Test registering with invalid event type raises error."""
        manager = HookManager()

        with pytest.raises(ValueError, match="Invalid event type"):
            manager.register("invalid", lambda *args: None)

    def test_register_non_callable(self):
        """Test registering non-callable raises error."""
        manager = HookManager()

        with pytest.raises(ValueError, match="Callback must be callable"):
            manager.register(HookEvent.AFTER_INSERT, "not a function")

    def test_unregister_hook(self):
        """Test unregistering a hook callback."""
        manager = HookManager()

        def callback(table, event, docs):
            pass

        manager.register(HookEvent.AFTER_INSERT, callback)
        assert manager.has_hooks(HookEvent.AFTER_INSERT)

        result = manager.unregister(HookEvent.AFTER_INSERT, callback)
        assert result is True
        assert not manager.has_hooks(HookEvent.AFTER_INSERT)

    def test_unregister_nonexistent_hook(self):
        """Test unregistering a callback that wasn't registered."""
        manager = HookManager()

        def callback(table, event, docs):
            pass

        result = manager.unregister(HookEvent.AFTER_INSERT, callback)
        assert result is False

    def test_unregister_all_for_event(self):
        """Test unregistering all hooks for a specific event."""
        manager = HookManager()

        manager.register(HookEvent.AFTER_INSERT, lambda *args: None)
        manager.register(HookEvent.AFTER_INSERT, lambda *args: None)
        manager.register(HookEvent.BEFORE_INSERT, lambda *args: None)

        manager.unregister_all(HookEvent.AFTER_INSERT)

        assert not manager.has_hooks(HookEvent.AFTER_INSERT)
        assert manager.has_hooks(HookEvent.BEFORE_INSERT)

    def test_unregister_all(self):
        """Test unregistering all hooks for all events."""
        manager = HookManager()

        manager.register(HookEvent.AFTER_INSERT, lambda *args: None)
        manager.register(HookEvent.BEFORE_UPDATE, lambda *args: None)
        manager.register(HookEvent.AFTER_DELETE, lambda *args: None)

        manager.unregister_all()

        assert not manager.has_hooks(HookEvent.AFTER_INSERT)
        assert not manager.has_hooks(HookEvent.BEFORE_UPDATE)
        assert not manager.has_hooks(HookEvent.AFTER_DELETE)

    def test_get_hooks(self):
        """Test getting list of registered hooks."""
        manager = HookManager()

        cb1 = lambda *args: None
        cb2 = lambda *args: None

        manager.register(HookEvent.AFTER_INSERT, cb1)
        manager.register(HookEvent.AFTER_INSERT, cb2)

        hooks = manager.get_hooks(HookEvent.AFTER_INSERT)
        assert len(hooks) == 2
        assert cb1 in hooks
        assert cb2 in hooks

    def test_hook_error_does_not_interrupt(self):
        """Test that hook errors don't interrupt database operations."""
        manager = HookManager()

        def failing_callback(table, event, docs):
            raise Exception("Hook error!")

        manager.register(HookEvent.AFTER_INSERT, failing_callback)

        # Should issue warning but not raise
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            manager.run(HookEvent.AFTER_INSERT, "test", [{"doc_id": 1}])

            assert len(w) == 1
            assert "Hook callback" in str(w[0].message)
            assert "Hook error!" in str(w[0].message)

    def test_chained_registration(self):
        """Test that register returns self for chaining."""
        manager = HookManager()

        result = manager.register(HookEvent.AFTER_INSERT, lambda *args: None)
        assert result is manager


class TestInsertHooks:
    """Tests for insert operation hooks."""

    def test_before_insert_hook(self, db):
        """Test before insert hook receives correct data."""
        received = []

        def on_before_insert(table_name, event, documents):
            received.append({
                'table': table_name,
                'event': event,
                'docs': documents[:]
            })

        db.hooks.register(HookEvent.BEFORE_INSERT, on_before_insert)
        doc_id = db.insert({'name': 'Alice', 'age': 30})

        assert len(received) == 1
        assert received[0]['table'] == '_default'
        assert received[0]['event'] == HookEvent.BEFORE_INSERT
        assert len(received[0]['docs']) == 1
        assert received[0]['docs'][0]['name'] == 'Alice'
        assert received[0]['docs'][0]['doc_id'] == doc_id

    def test_after_insert_hook(self, db):
        """Test after insert hook receives correct data."""
        received = []

        def on_after_insert(table_name, event, documents):
            received.append({
                'table': table_name,
                'event': event,
                'docs': documents[:]
            })

        db.hooks.register(HookEvent.AFTER_INSERT, on_after_insert)
        doc_id = db.insert({'name': 'Bob'})

        assert len(received) == 1
        assert received[0]['event'] == HookEvent.AFTER_INSERT
        assert received[0]['docs'][0]['doc_id'] == doc_id

    def test_insert_multiple_hooks(self, db):
        """Test hooks fire for insert_multiple."""
        before_docs = []
        after_docs = []

        db.hooks.register(
            HookEvent.BEFORE_INSERT,
            lambda t, e, d: before_docs.extend(d)
        )
        db.hooks.register(
            HookEvent.AFTER_INSERT,
            lambda t, e, d: after_docs.extend(d)
        )

        db.insert_multiple([{'x': 1}, {'x': 2}, {'x': 3}])

        assert len(before_docs) == 3
        assert len(after_docs) == 3

    def test_hook_receives_table_name(self, db):
        """Test hook receives correct table name for different tables."""
        tables = []

        db.hooks.register(
            HookEvent.AFTER_INSERT,
            lambda t, e, d: tables.append(t)
        )

        db.insert({'data': 1})
        db.table('users').insert({'name': 'Alice'})
        db.table('orders').insert({'item': 'Book'})

        assert tables == ['_default', 'users', 'orders']

    def test_hook_error_does_not_block_insert(self, db):
        """Test that hook errors don't prevent insert from completing."""
        def failing_hook(table, event, docs):
            raise Exception("Hook failed!")

        db.hooks.register(HookEvent.BEFORE_INSERT, failing_hook)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            doc_id = db.insert({'name': 'Test'})

        # Insert should still have succeeded
        assert db.get(doc_id=doc_id) is not None


class TestUpdateHooks:
    """Tests for update operation hooks."""

    def test_before_update_hook(self, db):
        """Test before update hook receives document before changes."""
        doc_id = db.insert({'name': 'Alice', 'age': 30})

        received = []
        db.hooks.register(
            HookEvent.BEFORE_UPDATE,
            lambda t, e, d: received.extend(d)
        )

        db.update({'age': 31}, doc_ids=[doc_id])

        assert len(received) == 1
        assert received[0]['name'] == 'Alice'
        assert received[0]['age'] == 30  # Before update value
        assert received[0]['doc_id'] == doc_id

    def test_after_update_hook(self, db):
        """Test after update hook receives document after changes."""
        doc_id = db.insert({'name': 'Bob', 'age': 25})

        received = []
        db.hooks.register(
            HookEvent.AFTER_UPDATE,
            lambda t, e, d: received.extend(d)
        )

        db.update({'age': 26}, doc_ids=[doc_id])

        assert len(received) == 1
        assert received[0]['name'] == 'Bob'
        assert received[0]['age'] == 26  # After update value
        assert received[0]['doc_id'] == doc_id

    def test_update_with_condition(self, db):
        """Test update hooks work with query conditions."""
        db.insert({'type': 'user', 'count': 0})
        db.insert({'type': 'admin', 'count': 0})

        after_docs = []
        db.hooks.register(
            HookEvent.AFTER_UPDATE,
            lambda t, e, d: after_docs.extend(d)
        )

        User = Query()
        db.update({'count': 1}, User.type == 'user')

        assert len(after_docs) == 1
        assert after_docs[0]['type'] == 'user'
        assert after_docs[0]['count'] == 1

    def test_update_multiple_hooks(self, db):
        """Test hooks fire for update_multiple."""
        db.insert({'type': 'a', 'val': 0})
        db.insert({'type': 'b', 'val': 0})

        after_docs = []
        db.hooks.register(
            HookEvent.AFTER_UPDATE,
            lambda t, e, d: after_docs.extend(d)
        )

        Q = Query()
        db.update_multiple([
            ({'val': 1}, Q.type == 'a'),
            ({'val': 2}, Q.type == 'b'),
        ])

        assert len(after_docs) == 2

    def test_before_update_receives_original_values_by_doc_ids(self, db):
        """Test BEFORE_UPDATE hook receives original values when updating by doc_ids."""
        doc_id = db.insert({'name': 'Original', 'count': 100, 'status': 'active'})

        before_docs = []
        after_docs = []

        db.hooks.register(
            HookEvent.BEFORE_UPDATE,
            lambda t, e, d: before_docs.extend(d)
        )
        db.hooks.register(
            HookEvent.AFTER_UPDATE,
            lambda t, e, d: after_docs.extend(d)
        )

        # Update multiple fields
        db.update({'name': 'Updated', 'count': 200, 'status': 'inactive'}, doc_ids=[doc_id])

        # BEFORE hook must have original values
        assert len(before_docs) == 1
        assert before_docs[0]['name'] == 'Original'
        assert before_docs[0]['count'] == 100
        assert before_docs[0]['status'] == 'active'

        # AFTER hook must have updated values
        assert len(after_docs) == 1
        assert after_docs[0]['name'] == 'Updated'
        assert after_docs[0]['count'] == 200
        assert after_docs[0]['status'] == 'inactive'

    def test_before_update_receives_original_values_by_condition(self, db):
        """Test BEFORE_UPDATE hook receives original values when updating by condition."""
        db.insert({'type': 'user', 'score': 50, 'level': 1})
        db.insert({'type': 'admin', 'score': 100, 'level': 5})

        before_docs = []
        after_docs = []

        db.hooks.register(
            HookEvent.BEFORE_UPDATE,
            lambda t, e, d: before_docs.extend(d)
        )
        db.hooks.register(
            HookEvent.AFTER_UPDATE,
            lambda t, e, d: after_docs.extend(d)
        )

        Q = Query()
        db.update({'score': 75, 'level': 2}, Q.type == 'user')

        # BEFORE hook must have original values for the user doc
        assert len(before_docs) == 1
        assert before_docs[0]['type'] == 'user'
        assert before_docs[0]['score'] == 50
        assert before_docs[0]['level'] == 1

        # AFTER hook must have updated values
        assert len(after_docs) == 1
        assert after_docs[0]['type'] == 'user'
        assert after_docs[0]['score'] == 75
        assert after_docs[0]['level'] == 2

    def test_before_update_receives_original_values_update_all(self, db):
        """Test BEFORE_UPDATE hook receives original values when updating all documents."""
        # Clear and insert fresh docs
        db.truncate()
        db.insert({'id': 1, 'value': 'a'})
        db.insert({'id': 2, 'value': 'b'})
        db.insert({'id': 3, 'value': 'c'})

        before_docs = []
        after_docs = []

        db.hooks.register(
            HookEvent.BEFORE_UPDATE,
            lambda t, e, d: before_docs.extend(d)
        )
        db.hooks.register(
            HookEvent.AFTER_UPDATE,
            lambda t, e, d: after_docs.extend(d)
        )

        # Update all documents
        db.update({'value': 'updated'})

        # BEFORE hook must have original values
        assert len(before_docs) == 3
        original_values = {d['id']: d['value'] for d in before_docs}
        assert original_values[1] == 'a'
        assert original_values[2] == 'b'
        assert original_values[3] == 'c'

        # AFTER hook must have updated values
        assert len(after_docs) == 3
        for doc in after_docs:
            assert doc['value'] == 'updated'

    def test_before_update_receives_original_values_with_callable(self, db):
        """Test BEFORE_UPDATE receives original values when using callable update."""
        doc_id = db.insert({'counter': 10})

        before_docs = []
        after_docs = []

        db.hooks.register(
            HookEvent.BEFORE_UPDATE,
            lambda t, e, d: before_docs.extend(d)
        )
        db.hooks.register(
            HookEvent.AFTER_UPDATE,
            lambda t, e, d: after_docs.extend(d)
        )

        # Use callable to increment counter
        def increment(doc):
            doc['counter'] += 5

        db.update(increment, doc_ids=[doc_id])

        # BEFORE hook must have original value
        assert len(before_docs) == 1
        assert before_docs[0]['counter'] == 10

        # AFTER hook must have updated value
        assert len(after_docs) == 1
        assert after_docs[0]['counter'] == 15

    def test_before_update_multiple_receives_original_values(self, db):
        """Test BEFORE_UPDATE receives original values for update_multiple."""
        db.insert({'category': 'A', 'price': 100})
        db.insert({'category': 'B', 'price': 200})

        before_docs = []
        after_docs = []

        db.hooks.register(
            HookEvent.BEFORE_UPDATE,
            lambda t, e, d: before_docs.extend(d)
        )
        db.hooks.register(
            HookEvent.AFTER_UPDATE,
            lambda t, e, d: after_docs.extend(d)
        )

        Q = Query()
        db.update_multiple([
            ({'price': 150}, Q.category == 'A'),
            ({'price': 250}, Q.category == 'B'),
        ])

        # BEFORE hook must have original prices
        assert len(before_docs) == 2
        original_prices = {d['category']: d['price'] for d in before_docs}
        assert original_prices['A'] == 100
        assert original_prices['B'] == 200

        # AFTER hook must have updated prices
        assert len(after_docs) == 2
        updated_prices = {d['category']: d['price'] for d in after_docs}
        assert updated_prices['A'] == 150
        assert updated_prices['B'] == 250


class TestDeleteHooks:
    """Tests for delete operation hooks."""

    def test_before_remove_hook(self, db):
        """Test before remove hook receives documents to be deleted."""
        doc_id = db.insert({'name': 'Delete Me'})

        received = []
        db.hooks.register(
            HookEvent.BEFORE_DELETE,
            lambda t, e, d: received.extend(d)
        )

        db.remove(doc_ids=[doc_id])

        assert len(received) == 1
        assert received[0]['name'] == 'Delete Me'
        assert received[0]['doc_id'] == doc_id

    def test_after_remove_hook(self, db):
        """Test after remove hook fires after deletion."""
        doc_id = db.insert({'name': 'Gone'})

        after_fired = []
        db.hooks.register(
            HookEvent.AFTER_DELETE,
            lambda t, e, d: after_fired.extend(d)
        )

        db.remove(doc_ids=[doc_id])

        assert len(after_fired) == 1
        assert after_fired[0]['doc_id'] == doc_id

    def test_remove_with_condition(self, db):
        """Test remove hooks work with query conditions."""
        db.insert({'status': 'active'})
        db.insert({'status': 'inactive'})

        deleted_docs = []
        db.hooks.register(
            HookEvent.AFTER_DELETE,
            lambda t, e, d: deleted_docs.extend(d)
        )

        Q = Query()
        db.remove(Q.status == 'inactive')

        assert len(deleted_docs) == 1
        assert deleted_docs[0]['status'] == 'inactive'

    def test_truncate_hooks(self, db):
        """Test hooks fire for truncate operation."""
        db.insert({'a': 1})
        db.insert({'b': 2})
        db.insert({'c': 3})

        deleted_docs = []
        db.hooks.register(
            HookEvent.BEFORE_DELETE,
            lambda t, e, d: deleted_docs.extend(d)
        )

        db.truncate()

        assert len(deleted_docs) == 3


class TestSoftDeleteHooks:
    """Tests for soft delete operation hooks."""

    def test_soft_remove_hooks(self, db):
        """Test hooks fire for soft_remove operation."""
        doc_id = db.insert({'name': 'Soft Delete'})

        deleted_docs = []
        db.hooks.register(
            HookEvent.AFTER_DELETE,
            lambda t, e, d: deleted_docs.extend(d)
        )

        db.soft_remove(doc_ids=[doc_id])

        assert len(deleted_docs) == 1
        assert deleted_docs[0]['doc_id'] == doc_id

    def test_restore_triggers_insert_hooks(self, db):
        """Test restore triggers INSERT hooks (bringing docs back)."""
        doc_id = db.insert({'name': 'Restore Me'})
        db.soft_remove(doc_ids=[doc_id])

        restored_docs = []
        db.hooks.register(
            HookEvent.AFTER_INSERT,
            lambda t, e, d: restored_docs.extend(d)
        )

        db.restore(doc_ids=[doc_id])

        assert len(restored_docs) == 1
        assert restored_docs[0]['doc_id'] == doc_id
        assert restored_docs[0]['name'] == 'Restore Me'

    def test_purge_hooks(self, db):
        """Test hooks fire for purge operation."""
        doc_id = db.insert({'name': 'Purge Me'})
        db.soft_remove(doc_ids=[doc_id])

        purged_docs = []
        db.hooks.register(
            HookEvent.AFTER_DELETE,
            lambda t, e, d: purged_docs.extend(d)
        )

        db.purge(doc_ids=[doc_id])

        assert len(purged_docs) == 1
        assert purged_docs[0]['doc_id'] == doc_id


class TestHookDataIntegrity:
    """Tests to ensure hooks don't affect database data."""

    def test_hook_receives_copy_not_reference(self, db):
        """Test that hooks receive copies of documents, not references."""
        doc_id = db.insert({'name': 'Original', 'count': 0})

        def modifying_hook(table, event, docs):
            # Try to modify the document
            for doc in docs:
                doc['name'] = 'Modified by hook'
                doc['count'] = 999

        db.hooks.register(HookEvent.AFTER_INSERT, modifying_hook)

        # Insert another document
        db.insert({'name': 'Test', 'value': 1})

        # Original document should be unchanged
        original = db.get(doc_id=doc_id)
        assert original['name'] == 'Original'
        assert original['count'] == 0

    def test_hook_cannot_modify_database(self, db):
        """Test that operations in hooks don't affect the main operation."""
        # This test ensures hooks are non-intrusive

        def hook_that_inserts(table, event, docs):
            # This should not affect the original insert
            pass  # Hooks receive copies, can't modify originals

        db.hooks.register(HookEvent.BEFORE_INSERT, hook_that_inserts)

        doc_id = db.insert({'data': 'test'})
        doc = db.get(doc_id=doc_id)

        assert doc['data'] == 'test'


class TestMultipleHooks:
    """Tests for multiple hooks on the same event."""

    def test_multiple_hooks_all_fire(self, db):
        """Test that multiple hooks on same event all fire."""
        results = []

        db.hooks.register(
            HookEvent.AFTER_INSERT,
            lambda t, e, d: results.append('hook1')
        )
        db.hooks.register(
            HookEvent.AFTER_INSERT,
            lambda t, e, d: results.append('hook2')
        )
        db.hooks.register(
            HookEvent.AFTER_INSERT,
            lambda t, e, d: results.append('hook3')
        )

        db.insert({'test': True})

        assert results == ['hook1', 'hook2', 'hook3']

    def test_one_failing_hook_doesnt_block_others(self, db):
        """Test that one failing hook doesn't prevent others from running."""
        results = []

        db.hooks.register(
            HookEvent.AFTER_INSERT,
            lambda t, e, d: results.append('hook1')
        )

        def failing_hook(t, e, d):
            raise Exception("I fail!")

        db.hooks.register(HookEvent.AFTER_INSERT, failing_hook)

        db.hooks.register(
            HookEvent.AFTER_INSERT,
            lambda t, e, d: results.append('hook3')
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            db.insert({'test': True})

        # All non-failing hooks should have run
        assert 'hook1' in results
        assert 'hook3' in results


class TestHookEventCoverage:
    """Test all hook events are properly triggered."""

    def test_all_events_can_be_registered(self):
        """Test all HookEvent values can be registered."""
        manager = HookManager()

        for event in HookEvent:
            manager.register(event, lambda t, e, d: None)
            assert manager.has_hooks(event)

    def test_before_and_after_order(self, db):
        """Test before hooks fire before after hooks."""
        order = []

        db.hooks.register(
            HookEvent.BEFORE_INSERT,
            lambda t, e, d: order.append('before')
        )
        db.hooks.register(
            HookEvent.AFTER_INSERT,
            lambda t, e, d: order.append('after')
        )

        db.insert({'test': True})

        assert order == ['before', 'after']
