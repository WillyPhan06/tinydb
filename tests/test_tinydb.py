import re
from collections.abc import Mapping

import pytest

from tinydb import TinyDB, where, Query
from tinydb.middlewares import Middleware, CachingMiddleware
from tinydb.storages import MemoryStorage, JSONStorage
from tinydb.table import Document


def test_drop_tables(db: TinyDB):
    db.drop_tables()

    db.insert({})
    db.drop_tables()

    assert len(db) == 0


def test_all(db: TinyDB):
    db.drop_tables()

    for i in range(10):
        db.insert({})

    assert len(db.all()) == 10


def test_insert(db: TinyDB):
    db.drop_tables()
    db.insert({'int': 1, 'char': 'a'})

    assert db.count(where('int') == 1) == 1

    db.drop_tables()

    db.insert({'int': 1, 'char': 'a'})
    db.insert({'int': 1, 'char': 'b'})
    db.insert({'int': 1, 'char': 'c'})

    assert db.count(where('int') == 1) == 3
    assert db.count(where('char') == 'a') == 1


def test_insert_ids(db: TinyDB):
    db.drop_tables()
    assert db.insert({'int': 1, 'char': 'a'}) == 1
    assert db.insert({'int': 1, 'char': 'a'}) == 2


def test_insert_with_doc_id(db: TinyDB):
    db.drop_tables()
    assert db.insert({'int': 1, 'char': 'a'}) == 1
    assert db.insert(Document({'int': 1, 'char': 'a'}, 12)) == 12
    assert db.insert(Document({'int': 1, 'char': 'a'}, 77)) == 77
    assert db.insert({'int': 1, 'char': 'a'}) == 78


def test_insert_with_duplicate_doc_id(db: TinyDB):
    db.drop_tables()
    assert db.insert({'int': 1, 'char': 'a'}) == 1

    with pytest.raises(ValueError):
        db.insert(Document({'int': 1, 'char': 'a'}, 1))


def test_insert_multiple(db: TinyDB):
    db.drop_tables()
    assert not db.contains(where('int') == 1)

    # Insert multiple from list
    db.insert_multiple([{'int': 1, 'char': 'a'},
                        {'int': 1, 'char': 'b'},
                        {'int': 1, 'char': 'c'}])

    assert db.count(where('int') == 1) == 3
    assert db.count(where('char') == 'a') == 1

    # Insert multiple from generator function
    def generator():
        for j in range(10):
            yield {'int': j}

    db.drop_tables()

    db.insert_multiple(generator())

    for i in range(10):
        assert db.count(where('int') == i) == 1
    assert db.count(where('int').exists()) == 10

    # Insert multiple from inline generator
    db.drop_tables()

    db.insert_multiple({'int': i} for i in range(10))

    for i in range(10):
        assert db.count(where('int') == i) == 1


def test_insert_multiple_with_ids(db: TinyDB):
    db.drop_tables()

    # Insert multiple from list
    assert db.insert_multiple([{'int': 1, 'char': 'a'},
                               {'int': 1, 'char': 'b'},
                               {'int': 1, 'char': 'c'}]) == [1, 2, 3]


def test_insert_multiple_with_doc_ids(db: TinyDB):
    db.drop_tables()

    assert db.insert_multiple([
        Document({'int': 1, 'char': 'a'}, 12),
        Document({'int': 1, 'char': 'b'}, 77)
    ]) == [12, 77]
    assert db.get(doc_id=12) == {'int': 1, 'char': 'a'}
    assert db.get(doc_id=77) == {'int': 1, 'char': 'b'}

    with pytest.raises(ValueError):
        db.insert_multiple([Document({'int': 1, 'char': 'a'}, 12)])


def test_insert_invalid_type_raises_error(db: TinyDB):
    with pytest.raises(ValueError, match='Document is not a Mapping'):
        # object() as an example of a non-mapping-type
        db.insert(object())  # type: ignore


def test_insert_valid_mapping_type(db: TinyDB):
    class CustomDocument(Mapping):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            return self.data[key]

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

    db.drop_tables()
    db.insert(CustomDocument({'int': 1, 'char': 'a'}))
    assert db.count(where('int') == 1) == 1


def test_custom_mapping_type_with_json(tmpdir):
    class CustomDocument(Mapping):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            return self.data[key]

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

    # Insert
    db = TinyDB(str(tmpdir.join('test.db')))
    db.drop_tables()
    db.insert(CustomDocument({'int': 1, 'char': 'a'}))
    assert db.count(where('int') == 1) == 1

    # Insert multiple
    db.insert_multiple([
        CustomDocument({'int': 2, 'char': 'a'}),
        CustomDocument({'int': 3, 'char': 'a'})
    ])
    assert db.count(where('int') == 1) == 1
    assert db.count(where('int') == 2) == 1
    assert db.count(where('int') == 3) == 1

    # Write back
    doc_id = db.get(where('int') == 3).doc_id
    db.update(CustomDocument({'int': 4, 'char': 'a'}), doc_ids=[doc_id])
    assert db.count(where('int') == 3) == 0
    assert db.count(where('int') == 4) == 1


def test_remove(db: TinyDB):
    db.remove(where('char') == 'b')

    assert len(db) == 2
    assert db.count(where('int') == 1) == 2


def test_remove_all_fails(db: TinyDB):
    with pytest.raises(RuntimeError):
        db.remove()


def test_remove_multiple(db: TinyDB):
    db.remove(where('int') == 1)

    assert len(db) == 0


def test_remove_ids(db: TinyDB):
    db.remove(doc_ids=[1, 2])

    assert len(db) == 1


def test_remove_returns_ids(db: TinyDB):
    assert db.remove(where('char') == 'b') == [2]


def test_update(db: TinyDB):
    assert len(db) == 3

    db.update({'int': 2}, where('char') == 'a')

    assert db.count(where('int') == 2) == 1
    assert db.count(where('int') == 1) == 2


def test_update_all(db: TinyDB):
    assert db.count(where('int') == 1) == 3

    db.update({'newField': True})

    assert db.count(where('newField') == True) == 3  # noqa


def test_update_returns_ids(db: TinyDB):
    db.drop_tables()
    assert db.insert({'int': 1, 'char': 'a'}) == 1
    assert db.insert({'int': 1, 'char': 'a'}) == 2

    assert db.update({'char': 'b'}, where('int') == 1) == [1, 2]


def test_update_transform(db: TinyDB):
    def increment(field):
        def transform(el):
            el[field] += 1

        return transform

    def delete(field):
        def transform(el):
            del el[field]

        return transform

    assert db.count(where('int') == 1) == 3

    db.update(increment('int'), where('char') == 'a')
    db.update(delete('char'), where('char') == 'a')

    assert db.count(where('int') == 2) == 1
    assert db.count(where('char') == 'a') == 0
    assert db.count(where('int') == 1) == 2


def test_update_ids(db: TinyDB):
    db.update({'int': 2}, doc_ids=[1, 2])

    assert db.count(where('int') == 2) == 2


def test_update_multiple(db: TinyDB):
    assert len(db) == 3

    db.update_multiple([
        ({'int': 2}, where('char') == 'a'),
        ({'int': 4}, where('char') == 'b'),
    ])

    assert db.count(where('int') == 1) == 1
    assert db.count(where('int') == 2) == 1
    assert db.count(where('int') == 4) == 1


def test_update_multiple_operation(db: TinyDB):
    def increment(field):
        def transform(el):
            el[field] += 1

        return transform

    assert db.count(where('int') == 1) == 3

    db.update_multiple([
        (increment('int'), where('char') == 'a'),
        (increment('int'), where('char') == 'b')
    ])

    assert db.count(where('int') == 2) == 2


def test_upsert(db: TinyDB):
    assert len(db) == 3

    # Document existing
    db.upsert({'int': 5}, where('char') == 'a')
    assert db.count(where('int') == 5) == 1

    # Document missing
    assert db.upsert({'int': 9, 'char': 'x'}, where('char') == 'x') == [4]
    assert db.count(where('int') == 9) == 1


def test_upsert_by_id(db: TinyDB):
    assert len(db) == 3

    # Single document existing
    extant_doc = Document({'char': 'v'}, doc_id=1)
    assert db.upsert(extant_doc) == [1]
    doc = db.get(where('char') == 'v')
    assert isinstance(doc, Document)
    assert doc is not None
    assert doc.doc_id == 1
    assert len(db) == 3

    # Single document missing
    missing_doc = Document({'int': 5, 'char': 'w'}, doc_id=5)
    assert db.upsert(missing_doc) == [5]
    doc = db.get(where('char') == 'w')
    assert isinstance(doc, Document)
    assert doc is not None
    assert doc.doc_id == 5
    assert len(db) == 4

    # Missing doc_id and condition
    with pytest.raises(ValueError, match=r"(?=.*\bdoc_id\b)(?=.*\bquery\b)"):
        db.upsert({'no_Document': 'no_query'})

    # Make sure we didn't break anything
    assert db.insert({'check': '_next_id'}) == 6


def test_search(db: TinyDB):
    assert not db._query_cache
    assert len(db.search(where('int') == 1)) == 3

    assert len(db._query_cache) == 1
    assert len(db.search(where('int') == 1)) == 3  # Query result from cache


def test_search_path(db: TinyDB):
    assert not db._query_cache
    assert len(db.search(where('int').exists())) == 3
    assert len(db._query_cache) == 1

    assert len(db.search(where('asd').exists())) == 0
    assert len(db.search(where('int').exists())) == 3  # Query result from cache


def test_search_no_results_cache(db: TinyDB):
    assert len(db.search(where('missing').exists())) == 0
    assert len(db.search(where('missing').exists())) == 0


def test_get(db: TinyDB):
    item = db.get(where('char') == 'b')
    assert isinstance(item, Document)
    assert item is not None
    assert item['char'] == 'b'


def test_get_ids(db: TinyDB):
    el = db.all()[0]
    assert db.get(doc_id=el.doc_id) == el
    assert db.get(doc_id=float('NaN')) is None  # type: ignore


def test_get_multiple_ids(db: TinyDB):
    el = db.all()
    assert db.get(doc_ids=[x.doc_id for x in el]) == el


def test_get_invalid(db: TinyDB):
    with pytest.raises(RuntimeError):
        db.get()


def test_count(db: TinyDB):
    assert db.count(where('int') == 1) == 3
    assert db.count(where('char') == 'd') == 0


def test_contains(db: TinyDB):
    assert db.contains(where('int') == 1)
    assert not db.contains(where('int') == 0)


def test_contains_ids(db: TinyDB):
    assert db.contains(doc_id=1)
    assert db.contains(doc_id=2)
    assert not db.contains(doc_id=88)


def test_contains_invalid(db: TinyDB):
    with pytest.raises(RuntimeError):
        db.contains()


def test_get_idempotent(db: TinyDB):
    u = db.get(where('int') == 1)
    z = db.get(where('int') == 1)
    assert u == z


def test_multiple_dbs():
    """
    Regression test for issue #3
    """
    db1 = TinyDB(storage=MemoryStorage)
    db2 = TinyDB(storage=MemoryStorage)

    db1.insert({'int': 1, 'char': 'a'})
    db1.insert({'int': 1, 'char': 'b'})
    db1.insert({'int': 1, 'value': 5.0})

    db2.insert({'color': 'blue', 'animal': 'turtle'})

    assert len(db1) == 3
    assert len(db2) == 1


def test_storage_closed_once():
    class Storage:
        def __init__(self):
            self.closed = False

        def read(self):
            return {}

        def write(self, data):
            pass

        def close(self):
            assert not self.closed
            self.closed = True

    with TinyDB(storage=Storage) as db:
        db.close()

    del db
    # If db.close() is called during cleanup, the assertion will fail and throw
    # and exception


def test_unique_ids(tmpdir):
    """
    :type tmpdir: py._path.local.LocalPath
    """
    path = str(tmpdir.join('db.json'))

    # Verify ids are unique when reopening the DB and inserting
    with TinyDB(path) as _db:
        _db.insert({'x': 1})

    with TinyDB(path) as _db:
        _db.insert({'x': 1})

    with TinyDB(path) as _db:
        data = _db.all()

        assert data[0].doc_id != data[1].doc_id

    # Verify ids stay unique when inserting/removing
    with TinyDB(path) as _db:
        _db.drop_tables()
        _db.insert_multiple({'x': i} for i in range(5))
        _db.remove(where('x') == 2)

        assert len(_db) == 4

        ids = [e.doc_id for e in _db.all()]
        assert len(ids) == len(set(ids))


def test_lastid_after_open(tmpdir):
    """
    Regression test for issue #34

    :type tmpdir: py._path.local.LocalPath
    """

    NUM = 100
    path = str(tmpdir.join('db.json'))

    with TinyDB(path) as _db:
        _db.insert_multiple({'i': i} for i in range(NUM))

    with TinyDB(path) as _db:
        assert _db._get_next_id() - 1 == NUM


def test_doc_ids_json(tmpdir):
    """
    Regression test for issue #45
    """

    path = str(tmpdir.join('db.json'))

    with TinyDB(path) as _db:
        _db.drop_tables()
        assert _db.insert({'int': 1, 'char': 'a'}) == 1
        assert _db.insert({'int': 1, 'char': 'a'}) == 2

        _db.drop_tables()
        assert _db.insert_multiple([{'int': 1, 'char': 'a'},
                                    {'int': 1, 'char': 'b'},
                                    {'int': 1, 'char': 'c'}]) == [1, 2, 3]

        assert _db.contains(doc_id=1)
        assert _db.contains(doc_id=2)
        assert not _db.contains(doc_id=88)

        _db.update({'int': 2}, doc_ids=[1, 2])
        assert _db.count(where('int') == 2) == 2

        el = _db.all()[0]
        assert _db.get(doc_id=el.doc_id) == el
        assert _db.get(doc_id=float('NaN')) is None

        _db.remove(doc_ids=[1, 2])
        assert len(_db) == 1


def test_insert_string(tmpdir):
    path = str(tmpdir.join('db.json'))

    with TinyDB(path) as _db:
        data = [{'int': 1}, {'int': 2}]
        _db.insert_multiple(data)

        with pytest.raises(ValueError):
            _db.insert([1, 2, 3])  # Fails

        with pytest.raises(ValueError):
            _db.insert({'bark'})  # Fails

        assert data == _db.all()

        _db.insert({'int': 3})  # Does not fail


def test_insert_invalid_dict(tmpdir):
    path = str(tmpdir.join('db.json'))

    with TinyDB(path) as _db:
        data = [{'int': 1}, {'int': 2}]
        _db.insert_multiple(data)

        with pytest.raises(TypeError):
            _db.insert({'int': _db})  # Fails

        assert data == _db.all()

        _db.insert({'int': 3})  # Does not fail


def test_gc(tmpdir):
    # See https://github.com/msiemens/tinydb/issues/92
    path = str(tmpdir.join('db.json'))
    db = TinyDB(path)
    table = db.table('foo')
    table.insert({'something': 'else'})
    table.insert({'int': 13})
    assert len(table.search(where('int') == 13)) == 1
    assert table.all() == [{'something': 'else'}, {'int': 13}]
    db.close()


def test_drop_table():
    db = TinyDB(storage=MemoryStorage)
    default_table_name = db.table(db.default_table_name).name

    assert [] == list(db.tables())
    db.drop_table(default_table_name)

    db.insert({'a': 1})
    assert [default_table_name] == list(db.tables())

    db.drop_table(default_table_name)
    assert [] == list(db.tables())

    table_name = 'some-other-table'
    db = TinyDB(storage=MemoryStorage)
    db.table(table_name).insert({'a': 1})
    assert {table_name} == db.tables()

    db.drop_table(table_name)
    assert set() == db.tables()
    assert table_name not in db._tables

    db.drop_table('non-existent-table-name')
    assert set() == db.tables()


def test_empty_write(tmpdir):
    path = str(tmpdir.join('db.json'))

    class ReadOnlyMiddleware(Middleware):
        def write(self, data):
            raise AssertionError('No write for unchanged db')

    TinyDB(path).close()
    TinyDB(path, storage=ReadOnlyMiddleware(JSONStorage)).close()


def test_query_cache():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple([
        {'name': 'foo', 'value': 42},
        {'name': 'bar', 'value': -1337}
    ])

    query = where('value') > 0

    results = db.search(query)
    assert len(results) == 1

    # Modify the db instance to not return any results when
    # bypassing the query cache
    db._tables[db.table(db.default_table_name).name]._read_table = lambda: {}

    # Make sure we got an independent copy of the result list
    results.extend([1])
    assert db.search(query) == [{'name': 'foo', 'value': 42}]


def test_tinydb_is_iterable(db: TinyDB):
    assert [r for r in db] == db.all()


def test_repr(tmpdir):
    path = str(tmpdir.join('db.json'))

    db = TinyDB(path)
    db.insert({'a': 1})

    assert re.match(
        r"<TinyDB "
        r"tables=\[u?\'_default\'\], "
        r"tables_count=1, "
        r"default_table_documents_count=1, "
        r"all_tables_documents_count=\[\'_default=1\'\]>",
        repr(db))


def test_delete(tmpdir):
    path = str(tmpdir.join('db.json'))

    db = TinyDB(path, ensure_ascii=False)
    q = Query()
    db.insert({'network': {'id': '114', 'name': 'ok', 'rpc': 'dac',
                           'ticker': 'mkay'}})
    assert db.search(q.network.id == '114') == [
        {'network': {'id': '114', 'name': 'ok', 'rpc': 'dac',
                     'ticker': 'mkay'}}
    ]
    db.remove(q.network.id == '114')
    assert db.search(q.network.id == '114') == []


def test_insert_multiple_with_single_dict(db: TinyDB):
    with pytest.raises(ValueError):
        d = {'first': 'John', 'last': 'smith'}
        db.insert_multiple(d)  # type: ignore
        db.close()


def test_access_storage():
    assert isinstance(TinyDB(storage=MemoryStorage).storage,
                      MemoryStorage)
    assert isinstance(TinyDB(storage=CachingMiddleware(MemoryStorage)).storage,
                      CachingMiddleware)


def test_empty_db_len():
    db = TinyDB(storage=MemoryStorage)
    assert len(db) == 0


def test_insert_on_existing_db(tmpdir):
    path = str(tmpdir.join('db.json'))

    db = TinyDB(path, ensure_ascii=False)
    db.insert({'foo': 'bar'})

    assert len(db) == 1

    db.close()

    db = TinyDB(path, ensure_ascii=False)
    db.insert({'foo': 'bar'})
    db.insert({'foo': 'bar'})

    assert len(db) == 3


def test_storage_access():
    db = TinyDB(storage=MemoryStorage)

    assert isinstance(db.storage, MemoryStorage)


def test_lambda_query():
    db = TinyDB(storage=MemoryStorage)
    db.insert({'foo': 'bar'})

    query = lambda doc: doc.get('foo') == 'bar'
    query.is_cacheable = lambda: False
    assert db.search(query) == [{'foo': 'bar'}]
    assert not db._query_cache


def test_search_with_limit():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(10))

    # Test limit
    results = db.search(where('int').exists(), limit=5)
    assert len(results) == 5

    # Test limit larger than result set
    results = db.search(where('int').exists(), limit=100)
    assert len(results) == 10

    # Test limit of 0
    results = db.search(where('int').exists(), limit=0)
    assert len(results) == 0

    # Test no limit (default behavior)
    results = db.search(where('int').exists())
    assert len(results) == 10


def test_search_with_skip():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(10))

    # Test skip
    results = db.search(where('int').exists(), skip=3)
    assert len(results) == 7

    # Test skip larger than result set
    results = db.search(where('int').exists(), skip=100)
    assert len(results) == 0

    # Test skip of 0 (default behavior)
    results = db.search(where('int').exists(), skip=0)
    assert len(results) == 10


def test_search_with_limit_and_skip():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(10))

    # Test combined limit and skip
    results = db.search(where('int').exists(), limit=3, skip=2)
    assert len(results) == 3

    # Test skip + limit exceeding result set
    results = db.search(where('int').exists(), limit=10, skip=8)
    assert len(results) == 2

    # Test skip beyond result set with limit
    results = db.search(where('int').exists(), limit=5, skip=15)
    assert len(results) == 0


def test_search_pagination_with_cache():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(10))

    query = where('int').exists()

    # First search populates cache
    assert not db._query_cache
    results1 = db.search(query, limit=3)
    assert len(results1) == 3
    assert len(db._query_cache) == 1

    # Cache stores full results, not paginated
    cached = db._query_cache.get(query)
    assert len(cached) == 10

    # Second search uses cache but applies different pagination
    results2 = db.search(query, limit=5, skip=2)
    assert len(results2) == 5


def test_all_with_limit():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(10))

    # Test limit
    results = db.all(limit=5)
    assert len(results) == 5

    # Test limit larger than result set
    results = db.all(limit=100)
    assert len(results) == 10

    # Test limit of 0
    results = db.all(limit=0)
    assert len(results) == 0

    # Test no limit (default behavior)
    results = db.all()
    assert len(results) == 10


def test_all_with_skip():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(10))

    # Test skip
    results = db.all(skip=3)
    assert len(results) == 7

    # Test skip larger than result set
    results = db.all(skip=100)
    assert len(results) == 0

    # Test skip of 0 (default behavior)
    results = db.all(skip=0)
    assert len(results) == 10


def test_all_with_limit_and_skip():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(10))

    # Test combined limit and skip
    results = db.all(limit=3, skip=2)
    assert len(results) == 3

    # Test skip + limit exceeding result set
    results = db.all(limit=10, skip=8)
    assert len(results) == 2

    # Test skip beyond result set with limit
    results = db.all(limit=5, skip=15)
    assert len(results) == 0


def test_pagination_preserves_document_order():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(10))

    # Get all documents
    all_docs = db.all()

    # Verify pagination preserves order
    page1 = db.all(limit=3)
    page2 = db.all(limit=3, skip=3)
    page3 = db.all(limit=3, skip=6)
    page4 = db.all(limit=3, skip=9)

    assert page1 == all_docs[0:3]
    assert page2 == all_docs[3:6]
    assert page3 == all_docs[6:9]
    assert page4 == all_docs[9:12]  # Only 1 document


def test_table_pagination(db: TinyDB):
    """Test pagination works on table level too."""
    table = db.table('test_table')
    table.insert_multiple({'int': i} for i in range(10))

    # Test search with pagination
    results = table.search(where('int').exists(), limit=5)
    assert len(results) == 5

    # Test all with pagination
    results = table.all(limit=5, skip=2)
    assert len(results) == 5


def test_search_negative_limit_raises_error():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(5))

    with pytest.raises(ValueError, match='limit must be a non-negative integer'):
        db.search(where('int').exists(), limit=-1)

    with pytest.raises(ValueError, match='limit must be a non-negative integer'):
        db.search(where('int').exists(), limit=-10)


def test_search_negative_skip_raises_error():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(5))

    with pytest.raises(ValueError, match='skip must be a non-negative integer'):
        db.search(where('int').exists(), skip=-1)

    with pytest.raises(ValueError, match='skip must be a non-negative integer'):
        db.search(where('int').exists(), skip=-10)


def test_all_negative_limit_raises_error():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(5))

    with pytest.raises(ValueError, match='limit must be a non-negative integer'):
        db.all(limit=-1)

    with pytest.raises(ValueError, match='limit must be a non-negative integer'):
        db.all(limit=-10)


def test_all_negative_skip_raises_error():
    db = TinyDB(storage=MemoryStorage)
    db.insert_multiple({'int': i} for i in range(5))

    with pytest.raises(ValueError, match='skip must be a non-negative integer'):
        db.all(skip=-1)

    with pytest.raises(ValueError, match='skip must be a non-negative integer'):
        db.all(skip=-10)


# =============================================================================
# search_iter() Tests
# =============================================================================
# These tests verify the iterator-based search functionality which provides
# memory-efficient processing of large result sets.


class TestSearchIterBasic:
    """Basic functionality tests for search_iter."""

    def test_returns_iterator_not_list(self):
        """Verify search_iter returns a generator/iterator, not a list."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(10))

        result = db.search_iter(where('int').exists())

        # Should be an iterator with __iter__ and __next__
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')
        # Should NOT be a list
        assert not isinstance(result, list)

    def test_returns_document_instances(self):
        """Verify each yielded item is a Document instance with correct data."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i, 'type': 'user'} for i in range(10))

        for doc in db.search_iter(where('type') == 'user'):
            assert isinstance(doc, Document)
            assert 'int' in doc
            assert doc['type'] == 'user'
            assert hasattr(doc, 'doc_id')

    def test_filters_documents_correctly(self):
        """Verify only matching documents are yielded."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'value': i, 'even': i % 2 == 0} for i in range(20))

        results = list(db.search_iter(where('even') == True))

        assert len(results) == 10
        assert all(doc['even'] is True for doc in results)

    def test_works_on_table_level(self):
        """Verify search_iter works correctly on table instances."""
        db = TinyDB(storage=MemoryStorage)
        table = db.table('test_table')
        table.insert_multiple({'int': i} for i in range(10))

        results = list(table.search_iter(where('int').exists()))
        assert len(results) == 10
        assert [doc['int'] for doc in results] == list(range(10))

    def test_no_cache_population(self):
        """Verify search_iter does not populate the query cache."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(10))

        query = where('int').exists()
        assert not db._query_cache

        # Consume the entire iterator
        list(db.search_iter(query))

        # Cache should remain empty
        assert not db._query_cache


class TestSearchIterLazyEvaluation:
    """Tests demonstrating the memory efficiency of search_iter through lazy evaluation."""

    def test_early_termination_does_not_process_all_documents(self):
        """
        Demonstrate that stopping iteration early means we don't need to
        materialize all matching documents into memory.

        This is the key benefit of search_iter over search() - with a large
        dataset, we can process just what we need without loading everything.
        """
        db = TinyDB(storage=MemoryStorage)
        # Insert a large number of documents
        total_docs = 10000
        db.insert_multiple({'index': i, 'data': f'value_{i}'} for i in range(total_docs))

        # Track how many documents we actually consume
        consumed_count = 0
        desired_count = 5

        iterator = db.search_iter(where('index').exists())
        for doc in iterator:
            consumed_count += 1
            if consumed_count >= desired_count:
                break

        # We only consumed the small amount we needed
        assert consumed_count == desired_count

    def test_limit_stops_iteration_early(self):
        """
        Verify that using limit parameter stops yielding after the limit,
        demonstrating we don't need to process all matching documents.
        """
        db = TinyDB(storage=MemoryStorage)
        total_docs = 5000
        db.insert_multiple({'index': i} for i in range(total_docs))

        # With limit, we should only get that many results
        small_limit = 10
        results = list(db.search_iter(where('index').exists(), limit=small_limit))

        assert len(results) == small_limit
        # Verify we got the first 10 documents (in insertion order)
        for i, doc in enumerate(results):
            assert doc['index'] == i

    def test_iterator_can_be_partially_consumed(self):
        """
        Verify that an iterator can be partially consumed, stopped,
        and the consumed portion is correct.
        """
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'index': i} for i in range(1000))

        iterator = db.search_iter(where('index').exists())

        # Consume first 3 items
        first_three = [next(iterator) for _ in range(3)]
        assert len(first_three) == 3
        assert [doc['index'] for doc in first_three] == [0, 1, 2]

        # Consume next 2 items
        next_two = [next(iterator) for _ in range(2)]
        assert len(next_two) == 2
        assert [doc['index'] for doc in next_two] == [3, 4]

        # We've only consumed 5 total, iterator still has more
        sixth = next(iterator)
        assert sixth['index'] == 5


class TestSearchIterPagination:
    """Tests for skip and limit pagination parameters."""

    def test_limit_basic(self):
        """Test basic limit functionality with exact document verification."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(100))

        results = list(db.search_iter(where('int').exists(), limit=10))

        assert len(results) == 10
        # Verify exact documents: should be first 10 documents (0-9)
        assert [doc['int'] for doc in results] == list(range(10))
        # Verify doc_ids are correct
        assert [doc.doc_id for doc in results] == list(range(1, 11))

    def test_limit_zero_returns_empty(self):
        """Test that limit=0 returns no results."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(100))

        results = list(db.search_iter(where('int').exists(), limit=0))

        assert results == []
        assert len(results) == 0

    def test_limit_larger_than_results(self):
        """Test limit larger than total matching documents returns all documents."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(10))

        results = list(db.search_iter(where('int').exists(), limit=100))

        assert len(results) == 10
        # Verify all 10 documents are returned with correct values
        assert [doc['int'] for doc in results] == list(range(10))
        assert [doc.doc_id for doc in results] == list(range(1, 11))

    def test_skip_basic(self):
        """Test basic skip functionality with exact document verification."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(100))

        results = list(db.search_iter(where('int').exists(), skip=90))

        assert len(results) == 10
        # Verify exact documents: should be documents 90-99
        assert [doc['int'] for doc in results] == list(range(90, 100))
        # Verify doc_ids are correct (doc_id starts at 1)
        assert [doc.doc_id for doc in results] == list(range(91, 101))

    def test_skip_zero_returns_all(self):
        """Test that skip=0 returns all results with correct values."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(10))

        results = list(db.search_iter(where('int').exists(), skip=0))

        assert len(results) == 10
        # Verify all documents returned with correct values
        assert [doc['int'] for doc in results] == list(range(10))
        assert [doc.doc_id for doc in results] == list(range(1, 11))

    def test_combined_limit_and_skip(self):
        """Test combined limit and skip parameters with exact document verification."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(100))

        results = list(db.search_iter(where('int').exists(), limit=10, skip=20))

        assert len(results) == 10
        # Verify exact documents: should be documents 20-29
        assert [doc['int'] for doc in results] == list(range(20, 30))
        # Verify doc_ids are correct
        assert [doc.doc_id for doc in results] == list(range(21, 31))

    def test_skip_plus_limit_exceeds_results(self):
        """Test when skip + limit exceeds total results returns remaining documents."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(10))

        results = list(db.search_iter(where('int').exists(), limit=10, skip=8))

        assert len(results) == 2
        # Verify exact documents: should be documents 8 and 9
        assert [doc['int'] for doc in results] == [8, 9]
        assert [doc.doc_id for doc in results] == [9, 10]

    def test_pagination_with_filtered_query(self):
        """Test pagination with a query that only matches some documents."""
        db = TinyDB(storage=MemoryStorage)
        # Insert 50 documents, only even 'int' values will match
        db.insert_multiple({'int': i, 'even': i % 2 == 0} for i in range(50))

        # Query for even documents (25 total: 0, 2, 4, ..., 48)
        results = list(db.search_iter(where('even') == True, limit=5, skip=10))

        assert len(results) == 5
        # Should get documents with int values: 20, 22, 24, 26, 28
        # (skipping first 10 even numbers: 0,2,4,6,8,10,12,14,16,18)
        assert [doc['int'] for doc in results] == [20, 22, 24, 26, 28]
        assert all(doc['even'] is True for doc in results)


class TestSearchIterEdgeCases:
    """Edge case tests for search_iter."""

    def test_skip_larger_than_total_documents(self):
        """Test that skip larger than total documents returns empty iterator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(10))

        results = list(db.search_iter(where('int').exists(), skip=100))
        assert results == []
        assert len(results) == 0

    def test_skip_equals_total_documents(self):
        """Test that skip equal to total documents returns empty iterator."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(10))

        results = list(db.search_iter(where('int').exists(), skip=10))
        assert results == []
        assert len(results) == 0

    def test_skip_larger_than_matching_documents(self):
        """Test skip larger than matching (not total) documents."""
        db = TinyDB(storage=MemoryStorage)
        # Insert 100 docs, but only 10 will match
        db.insert_multiple({'int': i, 'match': i < 10} for i in range(100))

        results = list(db.search_iter(where('match') == True, skip=20))
        assert results == []
        assert len(results) == 0

    def test_skip_with_limit_beyond_results(self):
        """Test skip + limit combination where skip exceeds results."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(10))

        results = list(db.search_iter(where('int').exists(), limit=5, skip=15))
        assert results == []
        assert len(results) == 0

    def test_empty_database(self):
        """Test search_iter on empty database."""
        db = TinyDB(storage=MemoryStorage)

        results = list(db.search_iter(where('int').exists()))
        assert results == []

    def test_no_matching_documents(self):
        """Test search_iter when no documents match the query."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(10))

        results = list(db.search_iter(where('nonexistent').exists()))
        assert results == []

    def test_negative_limit_raises_error(self):
        """Test that negative limit raises ValueError."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(5))

        with pytest.raises(ValueError, match='limit must be a non-negative integer'):
            list(db.search_iter(where('int').exists(), limit=-1))

        with pytest.raises(ValueError, match='limit must be a non-negative integer'):
            list(db.search_iter(where('int').exists(), limit=-10))

    def test_negative_skip_raises_error(self):
        """Test that negative skip raises ValueError."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(5))

        with pytest.raises(ValueError, match='skip must be a non-negative integer'):
            list(db.search_iter(where('int').exists(), skip=-1))

        with pytest.raises(ValueError, match='skip must be a non-negative integer'):
            list(db.search_iter(where('int').exists(), skip=-10))


class TestSearchIterConsistencyWithSearch:
    """Tests ensuring search_iter produces identical results to search."""

    def test_basic_results_match(self):
        """Verify basic results match between search and search_iter."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i, 'type': 'user' if i % 2 == 0 else 'admin'}
                          for i in range(50))

        query = where('type') == 'user'

        search_results = db.search(query)
        iter_results = list(db.search_iter(query))

        assert search_results == iter_results

    def test_pagination_consistency_multiple_pages(self):
        """
        Test that paginating through results with search_iter produces
        the same results as search for multiple consecutive pages.
        """
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'index': i, 'category': i % 3} for i in range(100))

        query = where('category') == 0  # 34 matching documents (0, 3, 6, ..., 99)
        page_size = 10

        # Get all pages using both methods and compare
        page = 0
        while True:
            skip = page * page_size
            search_page = db.search(query, limit=page_size, skip=skip)
            iter_page = list(db.search_iter(query, limit=page_size, skip=skip))

            assert search_page == iter_page, f"Mismatch on page {page}"

            if len(search_page) < page_size:
                # Last page reached
                break
            page += 1

        # Verify we actually tested multiple pages
        assert page >= 3, "Should have tested at least 3 pages"

    def test_various_limit_values_match(self):
        """Test various limit values produce matching results."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(50))

        query = where('int').exists()

        for limit in [1, 5, 10, 25, 50, 100]:
            search_results = db.search(query, limit=limit)
            iter_results = list(db.search_iter(query, limit=limit))
            assert search_results == iter_results, f"Mismatch for limit={limit}"

    def test_various_skip_values_match(self):
        """Test various skip values produce matching results."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(50))

        query = where('int').exists()

        for skip in [0, 1, 10, 25, 49, 50, 100]:
            search_results = db.search(query, skip=skip)
            iter_results = list(db.search_iter(query, skip=skip))
            assert search_results == iter_results, f"Mismatch for skip={skip}"

    def test_combined_pagination_values_match(self):
        """Test various combinations of limit and skip produce matching results."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(100))

        query = where('int').exists()

        test_cases = [
            (5, 0), (5, 10), (5, 95), (5, 100),
            (10, 0), (10, 50), (10, 90), (10, 100),
            (20, 0), (20, 40), (20, 80), (20, 100),
            (1, 0), (1, 50), (1, 99), (1, 100),
            (100, 0), (100, 50), (100, 100),
        ]

        for limit, skip in test_cases:
            search_results = db.search(query, limit=limit, skip=skip)
            iter_results = list(db.search_iter(query, limit=limit, skip=skip))
            assert search_results == iter_results, \
                f"Mismatch for limit={limit}, skip={skip}"

    def test_document_order_preserved(self):
        """Verify document order is preserved between search and search_iter."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'index': i} for i in range(100))

        query = where('index').exists()

        search_order = [doc['index'] for doc in db.search(query)]
        iter_order = [doc['index'] for doc in db.search_iter(query)]

        assert search_order == iter_order

    def test_repeated_calls_produce_same_results(self):
        """Verify multiple calls with same parameters produce identical results."""
        db = TinyDB(storage=MemoryStorage)
        db.insert_multiple({'int': i} for i in range(50))

        query = where('int') >= 25

        # Call search_iter multiple times with same parameters
        results1 = list(db.search_iter(query, limit=10, skip=5))
        results2 = list(db.search_iter(query, limit=10, skip=5))
        results3 = list(db.search_iter(query, limit=10, skip=5))

        assert results1 == results2 == results3

        # Also verify against search
        search_results = db.search(query, limit=10, skip=5)
        assert results1 == search_results
