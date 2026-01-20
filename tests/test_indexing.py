"""
Tests for field-level indexing functionality.
"""

import pytest

from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage
from tinydb.indexing import Index, IndexManager, UnhashableValueError


class TestIndex:
    """Tests for the Index class."""

    def test_add_and_get_eq(self):
        """Test adding documents and equality lookup."""
        index = Index('name')
        index.add(1, {'name': 'Alice'})
        index.add(2, {'name': 'Bob'})
        index.add(3, {'name': 'Alice'})

        assert index.get_eq('Alice') == {1, 3}
        assert index.get_eq('Bob') == {2}
        assert index.get_eq('Charlie') == set()

    def test_add_missing_field(self):
        """Test adding documents without the indexed field."""
        index = Index('name')
        index.add(1, {'name': 'Alice'})
        index.add(2, {'other': 'value'})

        assert index.get_eq('Alice') == {1}
        assert len(index) == 1

    def test_remove(self):
        """Test removing documents from index."""
        index = Index('name')
        index.add(1, {'name': 'Alice'})
        index.add(2, {'name': 'Alice'})
        index.add(3, {'name': 'Bob'})

        index.remove(1)
        assert index.get_eq('Alice') == {2}
        assert index.get_eq('Bob') == {3}
        assert len(index) == 2

    def test_remove_nonexistent(self):
        """Test removing a document that doesn't exist."""
        index = Index('name')
        index.add(1, {'name': 'Alice'})

        # Should not raise
        index.remove(999)
        assert len(index) == 1

    def test_update(self):
        """Test updating a document in the index."""
        index = Index('name')
        index.add(1, {'name': 'Alice'})

        index.update(1, {'name': 'Bob'})
        assert index.get_eq('Alice') == set()
        assert index.get_eq('Bob') == {1}

    def test_get_ne(self):
        """Test not-equal lookup."""
        index = Index('name')
        index.add(1, {'name': 'Alice'})
        index.add(2, {'name': 'Bob'})
        index.add(3, {'name': 'Charlie'})

        assert index.get_ne('Alice') == {2, 3}
        assert index.get_ne('Unknown') == {1, 2, 3}

    def test_get_lt(self):
        """Test less-than lookup."""
        index = Index('age')
        index.add(1, {'age': 25})
        index.add(2, {'age': 30})
        index.add(3, {'age': 35})

        assert index.get_lt(30) == {1}
        assert index.get_lt(35) == {1, 2}
        assert index.get_lt(20) == set()

    def test_get_le(self):
        """Test less-than-or-equal lookup."""
        index = Index('age')
        index.add(1, {'age': 25})
        index.add(2, {'age': 30})
        index.add(3, {'age': 35})

        assert index.get_le(30) == {1, 2}
        assert index.get_le(35) == {1, 2, 3}
        assert index.get_le(20) == set()

    def test_get_gt(self):
        """Test greater-than lookup."""
        index = Index('age')
        index.add(1, {'age': 25})
        index.add(2, {'age': 30})
        index.add(3, {'age': 35})

        assert index.get_gt(30) == {3}
        assert index.get_gt(25) == {2, 3}
        assert index.get_gt(40) == set()

    def test_get_ge(self):
        """Test greater-than-or-equal lookup."""
        index = Index('age')
        index.add(1, {'age': 25})
        index.add(2, {'age': 30})
        index.add(3, {'age': 35})

        assert index.get_ge(30) == {2, 3}
        assert index.get_ge(25) == {1, 2, 3}
        assert index.get_ge(40) == set()

    def test_get_one_of(self):
        """Test one_of lookup."""
        index = Index('status')
        index.add(1, {'status': 'active'})
        index.add(2, {'status': 'pending'})
        index.add(3, {'status': 'inactive'})
        index.add(4, {'status': 'active'})

        assert index.get_one_of(['active', 'pending']) == {1, 2, 4}
        assert index.get_one_of(['inactive']) == {3}
        assert index.get_one_of(['unknown']) == set()

    def test_nested_field(self):
        """Test indexing nested fields."""
        index = Index('address.city')
        index.add(1, {'address': {'city': 'NYC'}})
        index.add(2, {'address': {'city': 'LA'}})
        index.add(3, {'address': {'city': 'NYC'}})

        assert index.get_eq('NYC') == {1, 3}
        assert index.get_eq('LA') == {2}

    def test_nested_field_missing_intermediate(self):
        """Test nested field when intermediate key is missing."""
        index = Index('address.city')
        index.add(1, {'address': {'city': 'NYC'}})
        index.add(2, {'name': 'Bob'})  # Missing 'address'
        index.add(3, {'address': 'string'})  # 'address' is not a dict

        assert index.get_eq('NYC') == {1}
        assert len(index) == 1

    def test_clear(self):
        """Test clearing the index."""
        index = Index('name')
        index.add(1, {'name': 'Alice'})
        index.add(2, {'name': 'Bob'})

        index.clear()
        assert len(index) == 0
        assert index.get_eq('Alice') == set()

    def test_hashable_list_values(self):
        """Test indexing list values."""
        index = Index('tags')
        index.add(1, {'tags': ['a', 'b']})
        index.add(2, {'tags': ['a', 'b']})
        index.add(3, {'tags': ['c']})

        # Lists should be converted to tuples for hashing
        assert index.get_eq(['a', 'b']) == {1, 2}
        assert index.get_eq(['c']) == {3}

    def test_hashable_dict_values(self):
        """Test indexing dict values."""
        index = Index('data')
        index.add(1, {'data': {'x': 1}})
        index.add(2, {'data': {'x': 1}})
        index.add(3, {'data': {'x': 2}})

        assert index.get_eq({'x': 1}) == {1, 2}
        assert index.get_eq({'x': 2}) == {3}

    def test_unhashable_value_raises_error(self):
        """Test that unhashable values raise UnhashableValueError."""
        index = Index('data')

        # Create a custom class without __hash__
        class Unhashable:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                return self.value == other.value

            __hash__ = None  # Explicitly unhashable

        # Should raise UnhashableValueError
        with pytest.raises(UnhashableValueError) as exc_info:
            index.add(1, {'data': Unhashable(42)})

        # Check error details
        assert exc_info.value.field_name == 'data'
        assert exc_info.value.doc_id == 1
        assert exc_info.value.value_path == []  # Direct value, no path
        assert 'unhashable' in str(exc_info.value).lower()

    def test_unhashable_nested_in_list_raises_error(self):
        """Test that unhashable values nested in lists raise error with path."""
        index = Index('items')

        class Unhashable:
            __hash__ = None

        with pytest.raises(UnhashableValueError) as exc_info:
            index.add(1, {'items': [1, 2, Unhashable()]})

        # Path should indicate index 2 in the list
        assert exc_info.value.field_name == 'items'
        assert exc_info.value.value_path == [2]
        assert '[2]' in str(exc_info.value)

    def test_unhashable_nested_in_dict_raises_error(self):
        """Test that unhashable values nested in dicts raise error with path."""
        index = Index('data')

        class Unhashable:
            __hash__ = None

        with pytest.raises(UnhashableValueError) as exc_info:
            index.add(1, {'data': {'key': Unhashable()}})

        # Path should indicate the 'key' in the dict
        assert exc_info.value.field_name == 'data'
        assert exc_info.value.value_path == ['key']
        assert "['key']" in str(exc_info.value)

    def test_unhashable_deeply_nested_in_list_of_dicts(self):
        """Test unhashable in complex structure: list of dicts with nested list."""
        index = Index('data')

        class Unhashable:
            __hash__ = None

        with pytest.raises(UnhashableValueError) as exc_info:
            index.add(1, {
                'data': {
                    'users': [
                        {'name': 'Alice', 'tags': ['a', 'b']},
                        {'name': 'Bob', 'tags': ['c', Unhashable()]},
                    ]
                }
            })

        # Path should be ['users', 1, 'tags', 1]
        assert exc_info.value.field_name == 'data'
        assert exc_info.value.value_path == ['users', 1, 'tags', 1]
        # Verify the formatted path in message
        assert "['users'][1]['tags'][1]" in str(exc_info.value)

    def test_unhashable_in_deeply_nested_dict(self):
        """Test unhashable in deeply nested dict structure."""
        index = Index('config')

        class Unhashable:
            __hash__ = None

        with pytest.raises(UnhashableValueError) as exc_info:
            index.add(5, {
                'config': {
                    'database': {
                        'connection': {
                            'pool': {
                                'handler': Unhashable()
                            }
                        }
                    }
                }
            })

        assert exc_info.value.field_name == 'config'
        assert exc_info.value.value_path == ['database', 'connection', 'pool', 'handler']
        assert "['database']['connection']['pool']['handler']" in str(exc_info.value)

    def test_unhashable_in_mixed_nested_structure(self):
        """Test unhashable in mixed list/dict nested structure."""
        index = Index('payload')

        class Unhashable:
            __hash__ = None

        with pytest.raises(UnhashableValueError) as exc_info:
            index.add(10, {
                'payload': [
                    'item0',
                    {
                        'nested': [
                            {'deep': [0, 1, {'value': Unhashable()}]}
                        ]
                    }
                ]
            })

        assert exc_info.value.field_name == 'payload'
        assert exc_info.value.value_path == [1, 'nested', 0, 'deep', 2, 'value']
        assert "[1]['nested'][0]['deep'][2]['value']" in str(exc_info.value)

    def test_unhashable_at_first_position_in_list(self):
        """Test unhashable at index 0 in list."""
        index = Index('items')

        class Unhashable:
            __hash__ = None

        with pytest.raises(UnhashableValueError) as exc_info:
            index.add(1, {'items': [Unhashable(), 'valid', 'valid']})

        assert exc_info.value.value_path == [0]
        assert '[0]' in str(exc_info.value)

    def test_error_message_includes_type_name(self):
        """Test that error message includes the actual type name."""
        index = Index('data')

        class MyCustomObject:
            __hash__ = None

        with pytest.raises(UnhashableValueError) as exc_info:
            index.add(1, {'data': MyCustomObject()})

        assert 'MyCustomObject' in str(exc_info.value)


class TestIndexManager:
    """Tests for the IndexManager class."""

    def test_create_and_list_indexes(self):
        """Test creating and listing indexes."""
        manager = IndexManager()
        docs = {1: {'name': 'Alice'}, 2: {'name': 'Bob'}}

        manager.create_index('name', docs)
        assert manager.list_indexes() == ['name']
        assert manager.has_index('name')

    def test_create_duplicate_index(self):
        """Test that creating duplicate index raises error."""
        manager = IndexManager()
        docs = {1: {'name': 'Alice'}}

        manager.create_index('name', docs)
        with pytest.raises(ValueError, match="already exists"):
            manager.create_index('name', docs)

    def test_drop_index(self):
        """Test dropping an index."""
        manager = IndexManager()
        docs = {1: {'name': 'Alice'}}

        manager.create_index('name', docs)
        manager.drop_index('name')
        assert not manager.has_index('name')
        assert manager.list_indexes() == []

    def test_drop_nonexistent_index(self):
        """Test dropping an index that doesn't exist."""
        manager = IndexManager()

        with pytest.raises(ValueError, match="No index exists"):
            manager.drop_index('name')

    def test_on_insert(self):
        """Test index update on insert."""
        manager = IndexManager()
        docs = {1: {'name': 'Alice'}}
        manager.create_index('name', docs)

        manager.on_insert(2, {'name': 'Bob'})

        index = manager.get_index('name')
        assert index.get_eq('Alice') == {1}
        assert index.get_eq('Bob') == {2}

    def test_on_update(self):
        """Test index update on document update."""
        manager = IndexManager()
        docs = {1: {'name': 'Alice'}}
        manager.create_index('name', docs)

        manager.on_update(1, {'name': 'Bob'})

        index = manager.get_index('name')
        assert index.get_eq('Alice') == set()
        assert index.get_eq('Bob') == {1}

    def test_on_remove(self):
        """Test index update on document removal."""
        manager = IndexManager()
        docs = {1: {'name': 'Alice'}, 2: {'name': 'Bob'}}
        manager.create_index('name', docs)

        manager.on_remove(1)

        index = manager.get_index('name')
        assert index.get_eq('Alice') == set()
        assert index.get_eq('Bob') == {2}

    def test_try_execute_query_equality(self):
        """Test executing equality query via index."""
        manager = IndexManager()
        docs = {1: {'name': 'Alice'}, 2: {'name': 'Bob'}, 3: {'name': 'Alice'}}
        manager.create_index('name', docs)

        # Simulate query hash for: where('name') == 'Alice'
        query_hash = ('==', ('name',), 'Alice')
        result = manager.try_execute_query(query_hash, set(docs.keys()))

        assert result == {1, 3}

    def test_try_execute_query_comparison(self):
        """Test executing comparison queries via index."""
        manager = IndexManager()
        docs = {1: {'age': 25}, 2: {'age': 30}, 3: {'age': 35}}
        manager.create_index('age', docs)

        # Test < operator
        result = manager.try_execute_query(('<', ('age',), 30), set(docs.keys()))
        assert result == {1}

        # Test <= operator
        result = manager.try_execute_query(('<=', ('age',), 30), set(docs.keys()))
        assert result == {1, 2}

        # Test > operator
        result = manager.try_execute_query(('>', ('age',), 30), set(docs.keys()))
        assert result == {3}

        # Test >= operator
        result = manager.try_execute_query(('>=', ('age',), 30), set(docs.keys()))
        assert result == {2, 3}

    def test_try_execute_query_no_index(self):
        """Test that query without index returns None."""
        manager = IndexManager()
        docs = {1: {'name': 'Alice'}}
        # No index created

        query_hash = ('==', ('name',), 'Alice')
        result = manager.try_execute_query(query_hash, set(docs.keys()))

        assert result is None

    def test_try_execute_query_and(self):
        """Test executing AND query via index."""
        manager = IndexManager()
        docs = {
            1: {'name': 'Alice', 'age': 25},
            2: {'name': 'Bob', 'age': 30},
            3: {'name': 'Alice', 'age': 30}
        }
        manager.create_index('name', docs)
        manager.create_index('age', docs)

        # Simulate: (where('name') == 'Alice') & (where('age') == 30)
        query_hash = ('and', frozenset([
            ('==', ('name',), 'Alice'),
            ('==', ('age',), 30)
        ]))

        result = manager.try_execute_query(query_hash, set(docs.keys()))
        assert result == {3}

    def test_try_execute_query_or(self):
        """Test executing OR query via index."""
        manager = IndexManager()
        docs = {
            1: {'name': 'Alice', 'age': 25},
            2: {'name': 'Bob', 'age': 30},
            3: {'name': 'Charlie', 'age': 35}
        }
        manager.create_index('name', docs)

        # Simulate: (where('name') == 'Alice') | (where('name') == 'Bob')
        query_hash = ('or', frozenset([
            ('==', ('name',), 'Alice'),
            ('==', ('name',), 'Bob')
        ]))

        result = manager.try_execute_query(query_hash, set(docs.keys()))
        assert result == {1, 2}

    def test_try_execute_query_not(self):
        """Test executing NOT query via index."""
        manager = IndexManager()
        docs = {1: {'name': 'Alice'}, 2: {'name': 'Bob'}, 3: {'name': 'Charlie'}}
        manager.create_index('name', docs)

        # Simulate: ~(where('name') == 'Alice')
        query_hash = ('not', ('==', ('name',), 'Alice'))

        result = manager.try_execute_query(query_hash, set(docs.keys()))
        assert result == {2, 3}


class TestTableIndexing:
    """Tests for indexing integrated with TinyDB Table."""

    def setup_method(self):
        """Set up a fresh database for each test."""
        self.db = TinyDB(storage=MemoryStorage)
        self.table = self.db.table('test')

    def teardown_method(self):
        """Clean up after each test."""
        self.db.close()

    def test_create_index(self):
        """Test creating an index on a table."""
        self.table.insert({'name': 'Alice'})
        self.table.insert({'name': 'Bob'})

        self.table.create_index('name')

        assert self.table.has_index('name')
        assert 'name' in self.table.list_indexes()

    def test_create_index_duplicate(self):
        """Test that duplicate index creation raises error."""
        self.table.insert({'name': 'Alice'})
        self.table.create_index('name')

        with pytest.raises(ValueError):
            self.table.create_index('name')

    def test_drop_index(self):
        """Test dropping an index."""
        self.table.insert({'name': 'Alice'})
        self.table.create_index('name')
        self.table.drop_index('name')

        assert not self.table.has_index('name')

    def test_search_with_index(self):
        """Test that search uses index for faster lookup."""
        # Insert many documents
        for i in range(100):
            self.table.insert({'id': i, 'name': f'user_{i}'})

        # Create index
        self.table.create_index('id')

        # Search should use the index
        results = self.table.search(where('id') == 50)
        assert len(results) == 1
        assert results[0]['name'] == 'user_50'

    def test_search_comparison_with_index(self):
        """Test comparison queries with index."""
        for i in range(10):
            self.table.insert({'age': i * 10})

        self.table.create_index('age')

        # Less than
        results = self.table.search(where('age') < 30)
        assert len(results) == 3

        # Greater than or equal
        results = self.table.search(where('age') >= 50)
        assert len(results) == 5

    def test_search_and_query_with_index(self):
        """Test AND queries with indexes."""
        self.table.insert({'name': 'Alice', 'age': 25})
        self.table.insert({'name': 'Bob', 'age': 30})
        self.table.insert({'name': 'Alice', 'age': 35})

        self.table.create_index('name')
        self.table.create_index('age')

        results = self.table.search(
            (where('name') == 'Alice') & (where('age') == 35)
        )
        assert len(results) == 1
        assert results[0]['age'] == 35

    def test_search_or_query_with_index(self):
        """Test OR queries with indexes."""
        self.table.insert({'name': 'Alice'})
        self.table.insert({'name': 'Bob'})
        self.table.insert({'name': 'Charlie'})

        self.table.create_index('name')

        results = self.table.search(
            (where('name') == 'Alice') | (where('name') == 'Bob')
        )
        assert len(results) == 2

    def test_index_maintained_on_insert(self):
        """Test that index is updated on insert."""
        self.table.insert({'name': 'Alice'})
        self.table.create_index('name')

        self.table.insert({'name': 'Bob'})

        results = self.table.search(where('name') == 'Bob')
        assert len(results) == 1

    def test_index_maintained_on_update(self):
        """Test that index is updated on update."""
        doc_id = self.table.insert({'name': 'Alice'})
        self.table.create_index('name')

        self.table.update({'name': 'Bob'}, doc_ids=[doc_id])

        results = self.table.search(where('name') == 'Alice')
        assert len(results) == 0

        results = self.table.search(where('name') == 'Bob')
        assert len(results) == 1

    def test_index_maintained_on_remove(self):
        """Test that index is updated on remove."""
        doc_id = self.table.insert({'name': 'Alice'})
        self.table.create_index('name')

        self.table.remove(doc_ids=[doc_id])

        results = self.table.search(where('name') == 'Alice')
        assert len(results) == 0

    def test_index_maintained_on_insert_multiple(self):
        """Test that index is updated on insert_multiple."""
        self.table.create_index('name')

        self.table.insert_multiple([
            {'name': 'Alice'},
            {'name': 'Bob'},
            {'name': 'Charlie'}
        ])

        results = self.table.search(where('name') == 'Alice')
        assert len(results) == 1

        results = self.table.search(where('name') == 'Bob')
        assert len(results) == 1

    def test_rebuild_indexes(self):
        """Test rebuilding indexes."""
        self.table.insert({'name': 'Alice'})
        self.table.insert({'name': 'Bob'})
        self.table.create_index('name')

        # Rebuild should not lose data
        self.table.rebuild_indexes()

        results = self.table.search(where('name') == 'Alice')
        assert len(results) == 1

    def test_get_with_index(self):
        """Test that get() uses index for faster lookup."""
        for i in range(100):
            self.table.insert({'id': i})

        self.table.create_index('id')

        result = self.table.get(where('id') == 50)
        assert result is not None
        assert result['id'] == 50

    def test_nested_field_index(self):
        """Test indexing nested fields with actual queries."""
        self.table.insert({'user': {'name': 'Alice', 'age': 25}})
        self.table.insert({'user': {'name': 'Bob', 'age': 30}})
        self.table.insert({'user': {'name': 'Alice', 'age': 35}})
        self.table.insert({'other': 'data'})  # Missing nested field

        self.table.create_index('user.name')

        assert self.table.has_index('user.name')

        # Query using bracket notation (this is how TinyDB nested queries work)
        results = self.table.search(where('user')['name'] == 'Alice')
        assert len(results) == 2
        assert all(r['user']['name'] == 'Alice' for r in results)

        results = self.table.search(where('user')['name'] == 'Bob')
        assert len(results) == 1
        assert results[0]['user']['name'] == 'Bob'

        # Query for non-existent name
        results = self.table.search(where('user')['name'] == 'Charlie')
        assert len(results) == 0

    def test_nested_field_index_deep(self):
        """Test indexing deeply nested fields."""
        self.table.insert({'a': {'b': {'c': 'value1'}}})
        self.table.insert({'a': {'b': {'c': 'value2'}}})
        self.table.insert({'a': {'b': {'c': 'value1'}}})

        self.table.create_index('a.b.c')

        results = self.table.search(where('a')['b']['c'] == 'value1')
        assert len(results) == 2

        results = self.table.search(where('a')['b']['c'] == 'value2')
        assert len(results) == 1

    def test_nested_field_index_comparison(self):
        """Test comparison queries on nested indexed fields."""
        self.table.insert({'stats': {'score': 10}})
        self.table.insert({'stats': {'score': 20}})
        self.table.insert({'stats': {'score': 30}})

        self.table.create_index('stats.score')

        results = self.table.search(where('stats')['score'] > 15)
        assert len(results) == 2

        results = self.table.search(where('stats')['score'] <= 20)
        assert len(results) == 2

    def test_nested_field_index_with_get(self):
        """Test get() with nested indexed fields."""
        self.table.insert({'profile': {'email': 'alice@example.com'}})
        self.table.insert({'profile': {'email': 'bob@example.com'}})

        self.table.create_index('profile.email')

        result = self.table.get(where('profile')['email'] == 'alice@example.com')
        assert result is not None
        assert result['profile']['email'] == 'alice@example.com'

        result = self.table.get(where('profile')['email'] == 'unknown@example.com')
        assert result is None

    def test_index_with_soft_delete(self):
        """Test that indexes work correctly with soft delete."""
        doc_id = self.table.insert({'name': 'Alice'})
        self.table.create_index('name')

        # Soft delete
        self.table.soft_remove(doc_ids=[doc_id])

        # Should not find in normal search (soft deleted)
        results = self.table.search(where('name') == 'Alice')
        assert len(results) == 0

        # Restore
        self.table.restore(doc_ids=[doc_id])

        # Should find again
        results = self.table.search(where('name') == 'Alice')
        assert len(results) == 1

    def test_multiple_indexes(self):
        """Test having multiple indexes on different fields."""
        self.table.insert({'name': 'Alice', 'age': 25, 'city': 'NYC'})
        self.table.insert({'name': 'Bob', 'age': 30, 'city': 'LA'})
        self.table.insert({'name': 'Charlie', 'age': 25, 'city': 'NYC'})

        self.table.create_index('name')
        self.table.create_index('age')
        self.table.create_index('city')

        assert len(self.table.list_indexes()) == 3

        # Query on different indexed fields
        results = self.table.search(where('age') == 25)
        assert len(results) == 2

        results = self.table.search(where('city') == 'NYC')
        assert len(results) == 2

    def test_truncate_clears_indexes(self):
        """Test that truncate properly clears index data."""
        self.table.insert({'name': 'Alice'})
        self.table.insert({'name': 'Bob'})
        self.table.create_index('name')

        self.table.truncate()

        # Insert new data
        self.table.insert({'name': 'Charlie'})

        # Index should work correctly after truncate
        results = self.table.search(where('name') == 'Charlie')
        assert len(results) == 1

        results = self.table.search(where('name') == 'Alice')
        assert len(results) == 0

    def test_create_index_with_unhashable_value_raises(self):
        """Test that creating an index with unhashable values raises error."""
        class Unhashable:
            __hash__ = None

        # Insert document with unhashable value
        self.table.insert({'data': Unhashable()})

        # Creating index should raise UnhashableValueError
        with pytest.raises(UnhashableValueError) as exc_info:
            self.table.create_index('data')

        assert exc_info.value.field_name == 'data'
        assert 'unhashable' in str(exc_info.value).lower()

    def test_insert_with_unhashable_after_index_raises(self):
        """Test that inserting unhashable value after index creation raises."""
        class Unhashable:
            __hash__ = None

        # Create index first
        self.table.insert({'data': 'valid'})
        self.table.create_index('data')

        # Inserting document with unhashable value should raise
        with pytest.raises(UnhashableValueError):
            self.table.insert({'data': Unhashable()})

    def test_update_with_unhashable_after_index_raises(self):
        """Test that updating to unhashable value after index creation raises."""
        class Unhashable:
            __hash__ = None

        # Create document and index
        doc_id = self.table.insert({'data': 'valid'})
        self.table.create_index('data')

        # Updating to unhashable value should raise
        with pytest.raises(UnhashableValueError):
            self.table.update({'data': Unhashable()}, doc_ids=[doc_id])
