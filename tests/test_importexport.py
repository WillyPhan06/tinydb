"""
Tests for import/export functionality (CSV and JSONL formats).
"""

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

from tinydb import TinyDB, where
from tinydb.storages import MemoryStorage
from tinydb.table import Document
from tinydb.importexport import (
    export_csv,
    import_csv,
    export_jsonl,
    import_jsonl,
)


@pytest.fixture
def db():
    """Create a fresh in-memory database for each test."""
    db_ = TinyDB(storage=MemoryStorage)
    yield db_
    db_.close()


@pytest.fixture
def tmp_file(tmp_path):
    """Factory fixture that creates temporary file paths."""
    def _tmp_file(extension):
        return tmp_path / f'test_export.{extension}'
    return _tmp_file


class TestExportCSV:
    """Tests for CSV export functionality."""

    def test_export_csv_basic(self, db, tmp_file):
        """Test basic CSV export with simple data and exact field values."""
        table = db.table('users')
        id1 = table.insert({'name': 'Alice', 'age': 30})
        id2 = table.insert({'name': 'Bob', 'age': 25})

        csv_path = tmp_file('csv')
        count = table.export_csv(csv_path)

        assert count == 2
        assert csv_path.exists()

        # Parse CSV and verify exact field-value pairs
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2

        # Create a mapping by _id for easier assertion
        rows_by_id = {int(row['_id']): row for row in rows}

        # Assert exact values for Alice
        assert rows_by_id[id1]['_id'] == str(id1)
        assert rows_by_id[id1]['name'] == 'Alice'
        assert rows_by_id[id1]['age'] == '30'

        # Assert exact values for Bob
        assert rows_by_id[id2]['_id'] == str(id2)
        assert rows_by_id[id2]['name'] == 'Bob'
        assert rows_by_id[id2]['age'] == '25'

    def test_export_csv_empty_table(self, db, tmp_file):
        """Test CSV export of an empty table."""
        table = db.table('empty')
        csv_path = tmp_file('csv')

        count = table.export_csv(csv_path)

        assert count == 0
        assert csv_path.exists()

    def test_export_csv_complex_types(self, db, tmp_file):
        """Test CSV export with nested dicts and lists, asserting exact values."""
        table = db.table('complex')
        original_tags = ['a', 'b', 'c']
        original_metadata = {'key': 'value', 'nested': {'deep': True}}
        doc_id = table.insert({
            'name': 'Test',
            'tags': original_tags,
            'metadata': original_metadata,
        })

        csv_path = tmp_file('csv')
        count = table.export_csv(csv_path)

        assert count == 1

        # Parse CSV and verify exact field-value pairs
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        # Assert exact values
        assert row['_id'] == str(doc_id)
        assert row['name'] == 'Test'

        # Complex types should be JSON serialized - parse them back
        assert json.loads(row['tags']) == original_tags
        assert json.loads(row['metadata']) == original_metadata

    def test_export_csv_preserves_ids(self, db, tmp_file):
        """Test that document IDs are preserved in CSV export."""
        table = db.table('users')
        id1 = table.insert({'name': 'Alice'})
        id2 = table.insert({'name': 'Bob'})

        csv_path = tmp_file('csv')
        table.export_csv(csv_path)

        with open(csv_path, 'r') as f:
            content = f.read()
            assert str(id1) in content
            assert str(id2) in content

    def test_export_csv_excludes_soft_deleted_by_default(self, db, tmp_file):
        """Test that soft-deleted docs are excluded by default."""
        table = db.table('users')
        table.insert({'name': 'Alice'})
        id2 = table.insert({'name': 'Bob'})
        table.soft_remove(doc_ids=[id2])

        csv_path = tmp_file('csv')
        count = table.export_csv(csv_path)

        assert count == 1

        with open(csv_path, 'r') as f:
            content = f.read()
            assert 'Alice' in content
            assert 'Bob' not in content

    def test_export_csv_include_deleted(self, db, tmp_file):
        """Test CSV export including soft-deleted documents."""
        table = db.table('users')
        table.insert({'name': 'Alice'})
        id2 = table.insert({'name': 'Bob'})
        table.soft_remove(doc_ids=[id2])

        csv_path = tmp_file('csv')
        count = table.export_csv(csv_path, include_deleted=True)

        assert count == 2

        with open(csv_path, 'r') as f:
            content = f.read()
            assert 'Alice' in content
            assert 'Bob' in content

    def test_export_csv_with_none_values(self, db, tmp_file):
        """Test CSV export handles None values correctly."""
        table = db.table('users')
        table.insert({'name': 'Alice', 'email': None})

        csv_path = tmp_file('csv')
        table.export_csv(csv_path)

        with open(csv_path, 'r') as f:
            content = f.read()
            assert 'Alice' in content

    def test_export_csv_with_boolean_values(self, db, tmp_file):
        """Test CSV export handles boolean values correctly with exact assertions."""
        table = db.table('users')
        doc_id = table.insert({'name': 'Alice', 'active': True, 'admin': False})

        csv_path = tmp_file('csv')
        table.export_csv(csv_path)

        # Parse CSV and verify exact boolean values
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]

        assert row['_id'] == str(doc_id)
        assert row['name'] == 'Alice'
        # Booleans are JSON serialized
        assert json.loads(row['active']) is True
        assert json.loads(row['admin']) is False


class TestImportCSV:
    """Tests for CSV import functionality."""

    def test_import_csv_basic(self, db, tmp_file):
        """Test basic CSV import."""
        csv_path = tmp_file('csv')
        with open(csv_path, 'w') as f:
            f.write('_id,name,age\n')
            f.write('1,Alice,30\n')
            f.write('2,Bob,25\n')

        table = db.table('users')
        doc_ids = table.import_csv(csv_path)

        assert len(doc_ids) == 2
        assert len(table) == 2

        alice = table.get(doc_id=1)
        assert alice['name'] == 'Alice'
        assert alice['age'] == 30

    def test_import_csv_preserves_ids(self, db, tmp_file):
        """Test that document IDs are preserved during import."""
        csv_path = tmp_file('csv')
        with open(csv_path, 'w') as f:
            f.write('_id,name\n')
            f.write('100,Alice\n')
            f.write('200,Bob\n')

        table = db.table('users')
        doc_ids = table.import_csv(csv_path)

        assert 100 in doc_ids
        assert 200 in doc_ids
        assert table.get(doc_id=100)['name'] == 'Alice'
        assert table.get(doc_id=200)['name'] == 'Bob'

    def test_import_csv_without_ids(self, db, tmp_file):
        """Test CSV import without _id column generates new IDs."""
        csv_path = tmp_file('csv')
        with open(csv_path, 'w') as f:
            f.write('name,age\n')
            f.write('Alice,30\n')
            f.write('Bob,25\n')

        table = db.table('users')
        doc_ids = table.import_csv(csv_path)

        assert len(doc_ids) == 2
        assert all(isinstance(id_, int) for id_ in doc_ids)

    def test_import_csv_complex_types(self, db, tmp_file):
        """Test CSV import with JSON-serialized complex types."""
        csv_path = tmp_file('csv')
        with open(csv_path, 'w') as f:
            f.write('_id,name,tags,metadata\n')
            f.write('1,Test,"[""a"", ""b""]","{""key"": ""value""}"\n')

        table = db.table('complex')
        doc_ids = table.import_csv(csv_path)

        assert len(doc_ids) == 1
        doc = table.get(doc_id=1)
        assert doc['name'] == 'Test'
        assert doc['tags'] == ['a', 'b']
        assert doc['metadata'] == {'key': 'value'}

    def test_import_csv_numeric_conversion(self, db, tmp_file):
        """Test that numeric strings are converted to numbers."""
        csv_path = tmp_file('csv')
        with open(csv_path, 'w') as f:
            f.write('_id,int_val,float_val,str_val\n')
            f.write('1,42,3.14,hello\n')

        table = db.table('types')
        table.import_csv(csv_path)

        doc = table.get(doc_id=1)
        assert doc['int_val'] == 42
        assert isinstance(doc['int_val'], int)
        assert doc['float_val'] == 3.14
        assert isinstance(doc['float_val'], float)
        assert doc['str_val'] == 'hello'
        assert isinstance(doc['str_val'], str)

    def test_import_csv_empty_file(self, db, tmp_file):
        """Test importing an empty CSV file."""
        csv_path = tmp_file('csv')
        with open(csv_path, 'w') as f:
            pass  # Empty file

        table = db.table('users')
        doc_ids = table.import_csv(csv_path)

        assert len(doc_ids) == 0
        assert len(table) == 0

    def test_import_csv_boolean_values(self, db, tmp_file):
        """Test CSV import with boolean values."""
        csv_path = tmp_file('csv')
        with open(csv_path, 'w') as f:
            f.write('_id,name,active\n')
            f.write('1,Alice,true\n')
            f.write('2,Bob,false\n')

        table = db.table('users')
        table.import_csv(csv_path)

        alice = table.get(doc_id=1)
        bob = table.get(doc_id=2)
        assert alice['active'] is True
        assert bob['active'] is False


class TestExportJSONL:
    """Tests for JSONL export functionality."""

    def test_export_jsonl_basic(self, db, tmp_file):
        """Test basic JSONL export."""
        table = db.table('users')
        table.insert({'name': 'Alice', 'age': 30})
        table.insert({'name': 'Bob', 'age': 25})

        jsonl_path = tmp_file('jsonl')
        count = table.export_jsonl(jsonl_path)

        assert count == 2
        assert jsonl_path.exists()

        # Verify file contents
        with open(jsonl_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2

            doc1 = json.loads(lines[0])
            doc2 = json.loads(lines[1])

            assert '_id' in doc1
            assert '_id' in doc2
            names = {doc1['name'], doc2['name']}
            assert names == {'Alice', 'Bob'}

    def test_export_jsonl_empty_table(self, db, tmp_file):
        """Test JSONL export of an empty table."""
        table = db.table('empty')
        jsonl_path = tmp_file('jsonl')

        count = table.export_jsonl(jsonl_path)

        assert count == 0
        assert jsonl_path.exists()
        with open(jsonl_path, 'r') as f:
            assert f.read() == ''

    def test_export_jsonl_complex_types(self, db, tmp_file):
        """Test JSONL export with nested dicts and lists."""
        table = db.table('complex')
        table.insert({
            'name': 'Test',
            'tags': ['a', 'b', 'c'],
            'metadata': {'key': 'value', 'nested': {'deep': True}},
        })

        jsonl_path = tmp_file('jsonl')
        count = table.export_jsonl(jsonl_path)

        assert count == 1

        with open(jsonl_path, 'r') as f:
            doc = json.loads(f.readline())
            assert doc['tags'] == ['a', 'b', 'c']
            assert doc['metadata'] == {'key': 'value', 'nested': {'deep': True}}

    def test_export_jsonl_preserves_ids(self, db, tmp_file):
        """Test that document IDs are preserved in JSONL export."""
        table = db.table('users')
        id1 = table.insert({'name': 'Alice'})
        id2 = table.insert({'name': 'Bob'})

        jsonl_path = tmp_file('jsonl')
        table.export_jsonl(jsonl_path)

        with open(jsonl_path, 'r') as f:
            docs = [json.loads(line) for line in f]
            ids = {doc['_id'] for doc in docs}
            assert ids == {id1, id2}

    def test_export_jsonl_excludes_soft_deleted_by_default(self, db, tmp_file):
        """Test that soft-deleted docs are excluded by default."""
        table = db.table('users')
        table.insert({'name': 'Alice'})
        id2 = table.insert({'name': 'Bob'})
        table.soft_remove(doc_ids=[id2])

        jsonl_path = tmp_file('jsonl')
        count = table.export_jsonl(jsonl_path)

        assert count == 1

        with open(jsonl_path, 'r') as f:
            docs = [json.loads(line) for line in f]
            assert len(docs) == 1
            assert docs[0]['name'] == 'Alice'

    def test_export_jsonl_include_deleted(self, db, tmp_file):
        """Test JSONL export including soft-deleted documents."""
        table = db.table('users')
        table.insert({'name': 'Alice'})
        id2 = table.insert({'name': 'Bob'})
        table.soft_remove(doc_ids=[id2])

        jsonl_path = tmp_file('jsonl')
        count = table.export_jsonl(jsonl_path, include_deleted=True)

        assert count == 2

        with open(jsonl_path, 'r') as f:
            docs = [json.loads(line) for line in f]
            names = {doc['name'] for doc in docs}
            assert names == {'Alice', 'Bob'}


class TestImportJSONL:
    """Tests for JSONL import functionality."""

    def test_import_jsonl_basic(self, db, tmp_file):
        """Test basic JSONL import."""
        jsonl_path = tmp_file('jsonl')
        with open(jsonl_path, 'w') as f:
            f.write('{"_id": 1, "name": "Alice", "age": 30}\n')
            f.write('{"_id": 2, "name": "Bob", "age": 25}\n')

        table = db.table('users')
        doc_ids = table.import_jsonl(jsonl_path)

        assert len(doc_ids) == 2
        assert len(table) == 2

        alice = table.get(doc_id=1)
        assert alice['name'] == 'Alice'
        assert alice['age'] == 30

    def test_import_jsonl_preserves_ids(self, db, tmp_file):
        """Test that document IDs are preserved during import."""
        jsonl_path = tmp_file('jsonl')
        with open(jsonl_path, 'w') as f:
            f.write('{"_id": 100, "name": "Alice"}\n')
            f.write('{"_id": 200, "name": "Bob"}\n')

        table = db.table('users')
        doc_ids = table.import_jsonl(jsonl_path)

        assert 100 in doc_ids
        assert 200 in doc_ids
        assert table.get(doc_id=100)['name'] == 'Alice'
        assert table.get(doc_id=200)['name'] == 'Bob'

    def test_import_jsonl_without_ids(self, db, tmp_file):
        """Test JSONL import without _id field generates new IDs."""
        jsonl_path = tmp_file('jsonl')
        with open(jsonl_path, 'w') as f:
            f.write('{"name": "Alice", "age": 30}\n')
            f.write('{"name": "Bob", "age": 25}\n')

        table = db.table('users')
        doc_ids = table.import_jsonl(jsonl_path)

        assert len(doc_ids) == 2
        assert all(isinstance(id_, int) for id_ in doc_ids)

    def test_import_jsonl_complex_types(self, db, tmp_file):
        """Test JSONL import with nested data structures."""
        jsonl_path = tmp_file('jsonl')
        with open(jsonl_path, 'w') as f:
            data = {
                '_id': 1,
                'name': 'Test',
                'tags': ['a', 'b', 'c'],
                'metadata': {'key': 'value', 'nested': {'deep': True}},
            }
            f.write(json.dumps(data) + '\n')

        table = db.table('complex')
        doc_ids = table.import_jsonl(jsonl_path)

        assert len(doc_ids) == 1
        doc = table.get(doc_id=1)
        assert doc['name'] == 'Test'
        assert doc['tags'] == ['a', 'b', 'c']
        assert doc['metadata'] == {'key': 'value', 'nested': {'deep': True}}

    def test_import_jsonl_empty_file(self, db, tmp_file):
        """Test importing an empty JSONL file."""
        jsonl_path = tmp_file('jsonl')
        with open(jsonl_path, 'w') as f:
            pass  # Empty file

        table = db.table('users')
        doc_ids = table.import_jsonl(jsonl_path)

        assert len(doc_ids) == 0
        assert len(table) == 0

    def test_import_jsonl_with_blank_lines(self, db, tmp_file):
        """Test JSONL import skips blank lines."""
        jsonl_path = tmp_file('jsonl')
        with open(jsonl_path, 'w') as f:
            f.write('{"_id": 1, "name": "Alice"}\n')
            f.write('\n')
            f.write('{"_id": 2, "name": "Bob"}\n')
            f.write('   \n')

        table = db.table('users')
        doc_ids = table.import_jsonl(jsonl_path)

        assert len(doc_ids) == 2


class TestRoundTrip:
    """Tests for export then import round-trips."""

    def test_csv_roundtrip_basic(self, db, tmp_file):
        """Test CSV export then import preserves data."""
        table1 = db.table('source')
        table1.insert({'name': 'Alice', 'age': 30})
        table1.insert({'name': 'Bob', 'age': 25})

        csv_path = tmp_file('csv')
        table1.export_csv(csv_path)

        # Import into a new table
        table2 = db.table('dest')
        table2.import_csv(csv_path)

        assert len(table2) == 2
        docs = table2.all()
        names = {doc['name'] for doc in docs}
        assert names == {'Alice', 'Bob'}

    def test_csv_roundtrip_preserves_ids(self, db, tmp_file):
        """Test CSV round-trip preserves document IDs."""
        table1 = db.table('source')
        id1 = table1.insert({'name': 'Alice'})
        id2 = table1.insert({'name': 'Bob'})

        csv_path = tmp_file('csv')
        table1.export_csv(csv_path)

        table2 = db.table('dest')
        table2.import_csv(csv_path)

        assert table2.get(doc_id=id1)['name'] == 'Alice'
        assert table2.get(doc_id=id2)['name'] == 'Bob'

    def test_csv_roundtrip_complex_types(self, db, tmp_file):
        """Test CSV round-trip preserves complex data types."""
        table1 = db.table('source')
        original = {
            'name': 'Test',
            'tags': ['a', 'b', 'c'],
            'metadata': {'key': 'value'},
            'active': True,
            'count': 42,
            'ratio': 3.14,
        }
        doc_id = table1.insert(original)

        csv_path = tmp_file('csv')
        table1.export_csv(csv_path)

        table2 = db.table('dest')
        table2.import_csv(csv_path)

        imported = table2.get(doc_id=doc_id)
        assert imported['name'] == original['name']
        assert imported['tags'] == original['tags']
        assert imported['metadata'] == original['metadata']
        assert imported['active'] == original['active']
        assert imported['count'] == original['count']
        assert imported['ratio'] == original['ratio']

    def test_jsonl_roundtrip_basic(self, db, tmp_file):
        """Test JSONL export then import preserves data."""
        table1 = db.table('source')
        table1.insert({'name': 'Alice', 'age': 30})
        table1.insert({'name': 'Bob', 'age': 25})

        jsonl_path = tmp_file('jsonl')
        table1.export_jsonl(jsonl_path)

        table2 = db.table('dest')
        table2.import_jsonl(jsonl_path)

        assert len(table2) == 2
        docs = table2.all()
        names = {doc['name'] for doc in docs}
        assert names == {'Alice', 'Bob'}

    def test_jsonl_roundtrip_preserves_ids(self, db, tmp_file):
        """Test JSONL round-trip preserves document IDs."""
        table1 = db.table('source')
        id1 = table1.insert({'name': 'Alice'})
        id2 = table1.insert({'name': 'Bob'})

        jsonl_path = tmp_file('jsonl')
        table1.export_jsonl(jsonl_path)

        table2 = db.table('dest')
        table2.import_jsonl(jsonl_path)

        assert table2.get(doc_id=id1)['name'] == 'Alice'
        assert table2.get(doc_id=id2)['name'] == 'Bob'

    def test_jsonl_roundtrip_complex_types(self, db, tmp_file):
        """Test JSONL round-trip preserves complex data types."""
        table1 = db.table('source')
        original = {
            'name': 'Test',
            'tags': ['a', 'b', 'c'],
            'metadata': {'key': 'value', 'nested': {'deep': True}},
            'active': True,
            'count': 42,
            'ratio': 3.14,
            'nothing': None,
        }
        doc_id = table1.insert(original)

        jsonl_path = tmp_file('jsonl')
        table1.export_jsonl(jsonl_path)

        table2 = db.table('dest')
        table2.import_jsonl(jsonl_path)

        imported = table2.get(doc_id=doc_id)
        assert imported['name'] == original['name']
        assert imported['tags'] == original['tags']
        assert imported['metadata'] == original['metadata']
        assert imported['active'] == original['active']
        assert imported['count'] == original['count']
        assert imported['ratio'] == original['ratio']
        assert imported['nothing'] == original['nothing']


class TestStandaloneFunctions:
    """Tests for standalone import/export functions."""

    def test_export_csv_function(self, db, tmp_file):
        """Test standalone export_csv function."""
        table = db.table('users')
        table.insert({'name': 'Alice'})

        csv_path = tmp_file('csv')
        count = export_csv(table, csv_path)

        assert count == 1
        assert csv_path.exists()

    def test_import_csv_function(self, db, tmp_file):
        """Test standalone import_csv function."""
        csv_path = tmp_file('csv')
        with open(csv_path, 'w') as f:
            f.write('_id,name\n')
            f.write('1,Alice\n')

        table = db.table('users')
        doc_ids = import_csv(table, csv_path)

        assert len(doc_ids) == 1

    def test_export_jsonl_function(self, db, tmp_file):
        """Test standalone export_jsonl function."""
        table = db.table('users')
        table.insert({'name': 'Alice'})

        jsonl_path = tmp_file('jsonl')
        count = export_jsonl(table, jsonl_path)

        assert count == 1
        assert jsonl_path.exists()

    def test_import_jsonl_function(self, db, tmp_file):
        """Test standalone import_jsonl function."""
        jsonl_path = tmp_file('jsonl')
        with open(jsonl_path, 'w') as f:
            f.write('{"_id": 1, "name": "Alice"}\n')

        table = db.table('users')
        doc_ids = import_jsonl(table, jsonl_path)

        assert len(doc_ids) == 1


class TestTableMethods:
    """Tests for Table method wrappers."""

    def test_table_export_csv_method(self, db, tmp_file):
        """Test Table.export_csv method."""
        table = db.table('users')
        table.insert({'name': 'Alice'})

        csv_path = tmp_file('csv')
        count = table.export_csv(csv_path)

        assert count == 1

    def test_table_import_csv_method(self, db, tmp_file):
        """Test Table.import_csv method."""
        csv_path = tmp_file('csv')
        with open(csv_path, 'w') as f:
            f.write('_id,name\n')
            f.write('1,Alice\n')

        table = db.table('users')
        doc_ids = table.import_csv(csv_path)

        assert len(doc_ids) == 1

    def test_table_export_jsonl_method(self, db, tmp_file):
        """Test Table.export_jsonl method."""
        table = db.table('users')
        table.insert({'name': 'Alice'})

        jsonl_path = tmp_file('jsonl')
        count = table.export_jsonl(jsonl_path)

        assert count == 1

    def test_table_import_jsonl_method(self, db, tmp_file):
        """Test Table.import_jsonl method."""
        jsonl_path = tmp_file('jsonl')
        with open(jsonl_path, 'w') as f:
            f.write('{"_id": 1, "name": "Alice"}\n')

        table = db.table('users')
        doc_ids = table.import_jsonl(jsonl_path)

        assert len(doc_ids) == 1


class TestPathTypes:
    """Tests for different path types (str vs Path)."""

    def test_export_csv_with_path_object(self, db, tmp_file):
        """Test export_csv with pathlib.Path."""
        table = db.table('users')
        table.insert({'name': 'Alice'})

        csv_path = Path(tmp_file('csv'))
        count = table.export_csv(csv_path)

        assert count == 1

    def test_export_csv_with_str_path(self, db, tmp_file):
        """Test export_csv with string path."""
        table = db.table('users')
        table.insert({'name': 'Alice'})

        csv_path = str(tmp_file('csv'))
        count = table.export_csv(csv_path)

        assert count == 1

    def test_import_jsonl_with_path_object(self, db, tmp_file):
        """Test import_jsonl with pathlib.Path."""
        jsonl_path = Path(tmp_file('jsonl'))
        with open(jsonl_path, 'w') as f:
            f.write('{"_id": 1, "name": "Alice"}\n')

        table = db.table('users')
        doc_ids = table.import_jsonl(jsonl_path)

        assert len(doc_ids) == 1


class TestEncoding:
    """Tests for encoding parameter with Unicode characters."""

    def test_csv_utf8_byte_level_encoding(self, db, tmp_file):
        """Test that CSV export actually writes UTF-8 encoded bytes."""
        table = db.table('users')

        # Insert document with known UTF-8 characters
        # These have specific UTF-8 byte sequences we can verify
        doc = {
            'name': 'Björk',           # ö = 0xC3 0xB6 in UTF-8
            'city': '北京',             # 北 = 0xE5 0x8C 0x97, 京 = 0xE4 0xBA 0xAC
            'greeting': 'Привет',       # Russian characters
            'emoji_like': '©®™',        # Special symbols
        }
        table.insert(doc)

        csv_path = tmp_file('csv')
        table.export_csv(csv_path, encoding='utf-8')

        # Read file as raw bytes
        with open(csv_path, 'rb') as f:
            raw_bytes = f.read()

        # Verify the file can be decoded as UTF-8 without errors
        decoded_content = raw_bytes.decode('utf-8')

        # Verify specific UTF-8 byte sequences are present
        # 'ö' in UTF-8 is bytes 0xC3 0xB6
        assert b'\xc3\xb6' in raw_bytes, "UTF-8 encoding for 'ö' not found"

        # '北' in UTF-8 is bytes 0xE5 0x8C 0x97
        assert b'\xe5\x8c\x97' in raw_bytes, "UTF-8 encoding for '北' not found"

        # '京' in UTF-8 is bytes 0xE4 0xBA 0xAC
        assert b'\xe4\xba\xac' in raw_bytes, "UTF-8 encoding for '京' not found"

        # Verify the decoded content contains our original strings
        assert 'Björk' in decoded_content
        assert '北京' in decoded_content
        assert 'Привет' in decoded_content
        assert '©®™' in decoded_content

        # Verify it's NOT Latin-1 encoded (which would be different bytes)
        # In Latin-1, 'ö' would be single byte 0xF6, not 0xC3 0xB6
        assert b'\xf6' not in raw_bytes or b'\xc3\xb6' in raw_bytes, \
            "File may be Latin-1 encoded instead of UTF-8"

    def test_jsonl_utf8_byte_level_encoding(self, db, tmp_file):
        """Test that JSONL export actually writes UTF-8 encoded bytes."""
        table = db.table('users')

        # Insert document with known UTF-8 characters
        doc = {
            'name': 'Björk',           # ö = 0xC3 0xB6 in UTF-8
            'city': '北京',             # 北 = 0xE5 0x8C 0x97, 京 = 0xE4 0xBA 0xAC
            'greeting': 'Привет',       # Russian characters
            'emoji_like': '©®™',        # Special symbols
        }
        table.insert(doc)

        jsonl_path = tmp_file('jsonl')
        table.export_jsonl(jsonl_path, encoding='utf-8')

        # Read file as raw bytes
        with open(jsonl_path, 'rb') as f:
            raw_bytes = f.read()

        # Verify the file can be decoded as UTF-8 without errors
        decoded_content = raw_bytes.decode('utf-8')

        # Verify specific UTF-8 byte sequences are present
        # 'ö' in UTF-8 is bytes 0xC3 0xB6
        assert b'\xc3\xb6' in raw_bytes, "UTF-8 encoding for 'ö' not found"

        # '北' in UTF-8 is bytes 0xE5 0x8C 0x97
        assert b'\xe5\x8c\x97' in raw_bytes, "UTF-8 encoding for '北' not found"

        # '京' in UTF-8 is bytes 0xE4 0xBA 0xAC
        assert b'\xe4\xba\xac' in raw_bytes, "UTF-8 encoding for '京' not found"

        # Verify the decoded content contains our original strings
        assert 'Björk' in decoded_content
        assert '北京' in decoded_content
        assert 'Привет' in decoded_content
        assert '©®™' in decoded_content

        # Verify it's valid JSON when decoded
        json.loads(decoded_content.strip())

    def test_csv_utf8_import_from_bytes(self, db, tmp_file):
        """Test that CSV import correctly reads UTF-8 encoded bytes."""
        csv_path = tmp_file('csv')

        # Write a CSV file with known UTF-8 bytes directly
        utf8_content = '_id,name,city\n1,Björk,北京\n2,Café,Zürich\n'
        utf8_bytes = utf8_content.encode('utf-8')

        # Verify our test data has the expected UTF-8 byte sequences
        assert b'\xc3\xb6' in utf8_bytes  # ö
        assert b'\xe5\x8c\x97' in utf8_bytes  # 北

        # Write raw bytes to file
        with open(csv_path, 'wb') as f:
            f.write(utf8_bytes)

        # Import using UTF-8 encoding
        table = db.table('users')
        doc_ids = table.import_csv(csv_path, encoding='utf-8')

        assert len(doc_ids) == 2

        # Verify the imported data matches exactly
        doc1 = table.get(doc_id=1)
        assert doc1['name'] == 'Björk'
        assert doc1['city'] == '北京'

        doc2 = table.get(doc_id=2)
        assert doc2['name'] == 'Café'
        assert doc2['city'] == 'Zürich'

    def test_jsonl_utf8_import_from_bytes(self, db, tmp_file):
        """Test that JSONL import correctly reads UTF-8 encoded bytes."""
        jsonl_path = tmp_file('jsonl')

        # Write a JSONL file with known UTF-8 bytes directly
        utf8_content = '{"_id": 1, "name": "Björk", "city": "北京"}\n{"_id": 2, "name": "Café", "city": "Zürich"}\n'
        utf8_bytes = utf8_content.encode('utf-8')

        # Verify our test data has the expected UTF-8 byte sequences
        assert b'\xc3\xb6' in utf8_bytes  # ö
        assert b'\xe5\x8c\x97' in utf8_bytes  # 北

        # Write raw bytes to file
        with open(jsonl_path, 'wb') as f:
            f.write(utf8_bytes)

        # Import using UTF-8 encoding
        table = db.table('users')
        doc_ids = table.import_jsonl(jsonl_path, encoding='utf-8')

        assert len(doc_ids) == 2

        # Verify the imported data matches exactly
        doc1 = table.get(doc_id=1)
        assert doc1['name'] == 'Björk'
        assert doc1['city'] == '北京'

        doc2 = table.get(doc_id=2)
        assert doc2['name'] == 'Café'
        assert doc2['city'] == 'Zürich'

    def test_csv_full_byte_roundtrip(self, db, tmp_file):
        """Test complete byte-level roundtrip: insert -> export -> verify bytes -> import -> verify data."""
        table1 = db.table('source')

        # Original data with various UTF-8 characters
        original_docs = [
            {'name': 'Müller', 'desc': 'German umlaut ü'},
            {'name': '田中', 'desc': 'Japanese kanji'},
            {'name': 'Ωmega', 'desc': 'Greek omega Ω'},
            {'name': '한글', 'desc': 'Korean hangul'},
        ]

        doc_ids = [table1.insert(doc.copy()) for doc in original_docs]

        csv_path = tmp_file('csv')
        table1.export_csv(csv_path, encoding='utf-8')

        # Read and verify bytes
        with open(csv_path, 'rb') as f:
            raw_bytes = f.read()

        # Verify UTF-8 byte sequences
        assert b'\xc3\xbc' in raw_bytes, "ü not UTF-8 encoded"  # ü = 0xC3 0xBC
        assert b'\xe7\x94\xb0' in raw_bytes, "田 not UTF-8 encoded"  # 田
        assert b'\xce\xa9' in raw_bytes, "Ω not UTF-8 encoded"  # Ω = 0xCE 0xA9
        assert b'\xed\x95\x9c' in raw_bytes, "한 not UTF-8 encoded"  # 한

        # Verify no decoding errors
        decoded = raw_bytes.decode('utf-8')
        assert 'Müller' in decoded
        assert '田中' in decoded
        assert 'Ωmega' in decoded
        assert '한글' in decoded

        # Import and verify data integrity
        table2 = db.table('dest')
        table2.import_csv(csv_path, encoding='utf-8')

        for doc_id, original in zip(doc_ids, original_docs):
            imported = table2.get(doc_id=doc_id)
            assert imported['name'] == original['name'], f"Name mismatch for doc {doc_id}"
            assert imported['desc'] == original['desc'], f"Desc mismatch for doc {doc_id}"

    def test_jsonl_full_byte_roundtrip(self, db, tmp_file):
        """Test complete byte-level roundtrip: insert -> export -> verify bytes -> import -> verify data."""
        table1 = db.table('source')

        # Original data with various UTF-8 characters
        original_docs = [
            {'name': 'Müller', 'desc': 'German umlaut ü'},
            {'name': '田中', 'desc': 'Japanese kanji'},
            {'name': 'Ωmega', 'desc': 'Greek omega Ω'},
            {'name': '한글', 'desc': 'Korean hangul'},
        ]

        doc_ids = [table1.insert(doc.copy()) for doc in original_docs]

        jsonl_path = tmp_file('jsonl')
        table1.export_jsonl(jsonl_path, encoding='utf-8')

        # Read and verify bytes
        with open(jsonl_path, 'rb') as f:
            raw_bytes = f.read()

        # Verify UTF-8 byte sequences
        assert b'\xc3\xbc' in raw_bytes, "ü not UTF-8 encoded"  # ü = 0xC3 0xBC
        assert b'\xe7\x94\xb0' in raw_bytes, "田 not UTF-8 encoded"  # 田
        assert b'\xce\xa9' in raw_bytes, "Ω not UTF-8 encoded"  # Ω = 0xCE 0xA9
        assert b'\xed\x95\x9c' in raw_bytes, "한 not UTF-8 encoded"  # 한

        # Verify no decoding errors
        decoded = raw_bytes.decode('utf-8')
        assert 'Müller' in decoded
        assert '田中' in decoded
        assert 'Ωmega' in decoded
        assert '한글' in decoded

        # Verify each line is valid JSON
        for line in decoded.strip().split('\n'):
            json.loads(line)

        # Import and verify data integrity
        table2 = db.table('dest')
        table2.import_jsonl(jsonl_path, encoding='utf-8')

        for doc_id, original in zip(doc_ids, original_docs):
            imported = table2.get(doc_id=doc_id)
            assert imported['name'] == original['name'], f"Name mismatch for doc {doc_id}"
            assert imported['desc'] == original['desc'], f"Desc mismatch for doc {doc_id}"

    def test_csv_invalid_utf8_decode_fails(self, db, tmp_file):
        """Test that importing non-UTF-8 file with UTF-8 encoding raises an error."""
        csv_path = tmp_file('csv')

        # Write Latin-1 encoded content (which is NOT valid UTF-8 for certain characters)
        # In Latin-1, ö is single byte 0xF6, which is invalid UTF-8
        latin1_content = '_id,name\n1,Björk\n'
        latin1_bytes = latin1_content.encode('latin-1')

        # Verify Latin-1 encoding uses single byte for ö
        assert b'\xf6' in latin1_bytes  # Latin-1 ö is 0xF6

        with open(csv_path, 'wb') as f:
            f.write(latin1_bytes)

        table = db.table('users')

        # Importing with UTF-8 should fail because 0xF6 is invalid UTF-8
        with pytest.raises(UnicodeDecodeError):
            table.import_csv(csv_path, encoding='utf-8')

    def test_jsonl_invalid_utf8_decode_fails(self, db, tmp_file):
        """Test that importing non-UTF-8 file with UTF-8 encoding raises an error."""
        jsonl_path = tmp_file('jsonl')

        # Write Latin-1 encoded content (which is NOT valid UTF-8 for certain characters)
        latin1_content = '{"_id": 1, "name": "Björk"}\n'
        latin1_bytes = latin1_content.encode('latin-1')

        # Verify Latin-1 encoding uses single byte for ö
        assert b'\xf6' in latin1_bytes  # Latin-1 ö is 0xF6

        with open(jsonl_path, 'wb') as f:
            f.write(latin1_bytes)

        table = db.table('users')

        # Importing with UTF-8 should fail because 0xF6 is invalid UTF-8
        with pytest.raises(UnicodeDecodeError):
            table.import_jsonl(jsonl_path, encoding='utf-8')

    def test_csv_utf8_encoding_roundtrip(self, db, tmp_file):
        """Test CSV export/import with UTF-8 characters preserves data exactly."""
        table = db.table('users')

        # Insert documents with various UTF-8 characters
        original_docs = [
            {'name': 'Alice', 'city': 'New York', 'greeting': 'Hello'},
            {'name': 'Bj\u00f6rk', 'city': 'Reykjav\u00edk', 'greeting': 'Hall\u00f3'},
            {'name': '\u5c0f\u660e', 'city': '\u5317\u4eac', 'greeting': '\u4f60\u597d'},
            {'name': '\u0410\u043b\u0435\u043a\u0441\u0435\u0439', 'city': '\u041c\u043e\u0441\u043a\u0432\u0430', 'greeting': '\u041f\u0440\u0438\u0432\u0435\u0442'},
            {'name': 'Caf\u00e9', 'city': 'Z\u00fcrich', 'greeting': 'Gr\u00fcezi'},
        ]

        doc_ids = []
        for doc in original_docs:
            doc_ids.append(table.insert(doc.copy()))

        # Export to CSV with UTF-8 encoding
        csv_path = tmp_file('csv')
        count = table.export_csv(csv_path, encoding='utf-8')
        assert count == 5

        # Verify the exported file contains correct UTF-8 data
        with open(csv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            exported_rows = {int(row['_id']): row for row in reader}

        # Assert exact values for each document
        for i, (doc_id, original) in enumerate(zip(doc_ids, original_docs)):
            row = exported_rows[doc_id]
            assert row['name'] == original['name'], f"Name mismatch for doc {doc_id}"
            assert row['city'] == original['city'], f"City mismatch for doc {doc_id}"
            assert row['greeting'] == original['greeting'], f"Greeting mismatch for doc {doc_id}"

        # Import into a new table
        table2 = db.table('users_imported')
        imported_ids = table2.import_csv(csv_path, encoding='utf-8')
        assert len(imported_ids) == 5

        # Verify imported data matches original exactly
        for doc_id, original in zip(doc_ids, original_docs):
            imported = table2.get(doc_id=doc_id)
            assert imported['name'] == original['name'], f"Import name mismatch for doc {doc_id}"
            assert imported['city'] == original['city'], f"Import city mismatch for doc {doc_id}"
            assert imported['greeting'] == original['greeting'], f"Import greeting mismatch for doc {doc_id}"

    def test_jsonl_utf8_encoding_roundtrip(self, db, tmp_file):
        """Test JSONL export/import with UTF-8 characters preserves data exactly."""
        table = db.table('users')

        # Insert documents with various UTF-8 characters
        original_docs = [
            {'name': 'Alice', 'city': 'New York', 'greeting': 'Hello'},
            {'name': 'Bj\u00f6rk', 'city': 'Reykjav\u00edk', 'greeting': 'Hall\u00f3'},
            {'name': '\u5c0f\u660e', 'city': '\u5317\u4eac', 'greeting': '\u4f60\u597d'},
            {'name': '\u0410\u043b\u0435\u043a\u0441\u0435\u0439', 'city': '\u041c\u043e\u0441\u043a\u0432\u0430', 'greeting': '\u041f\u0440\u0438\u0432\u0435\u0442'},
            {'name': 'Caf\u00e9', 'city': 'Z\u00fcrich', 'greeting': 'Gr\u00fcezi'},
        ]

        doc_ids = []
        for doc in original_docs:
            doc_ids.append(table.insert(doc.copy()))

        # Export to JSONL with UTF-8 encoding
        jsonl_path = tmp_file('jsonl')
        count = table.export_jsonl(jsonl_path, encoding='utf-8')
        assert count == 5

        # Verify the exported file contains correct UTF-8 data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            exported_docs = {}
            for line in f:
                doc = json.loads(line)
                exported_docs[doc['_id']] = doc

        # Assert exact values for each document
        for doc_id, original in zip(doc_ids, original_docs):
            exported = exported_docs[doc_id]
            assert exported['name'] == original['name'], f"Name mismatch for doc {doc_id}"
            assert exported['city'] == original['city'], f"City mismatch for doc {doc_id}"
            assert exported['greeting'] == original['greeting'], f"Greeting mismatch for doc {doc_id}"

        # Import into a new table
        table2 = db.table('users_imported')
        imported_ids = table2.import_jsonl(jsonl_path, encoding='utf-8')
        assert len(imported_ids) == 5

        # Verify imported data matches original exactly
        for doc_id, original in zip(doc_ids, original_docs):
            imported = table2.get(doc_id=doc_id)
            assert imported['name'] == original['name'], f"Import name mismatch for doc {doc_id}"
            assert imported['city'] == original['city'], f"Import city mismatch for doc {doc_id}"
            assert imported['greeting'] == original['greeting'], f"Import greeting mismatch for doc {doc_id}"

    def test_csv_special_characters(self, db, tmp_file):
        """Test CSV with special characters like emojis and symbols."""
        table = db.table('messages')

        original_docs = [
            {'user': 'Alice', 'message': 'Hello! How are you?'},
            {'user': 'Bob', 'message': 'Great! See you tomorrow.'},
            {'user': 'Charlie', 'message': 'Price: $100.00 (50% off)'},
            {'user': 'Diana', 'message': 'Email: test@example.com'},
            {'user': 'Eve', 'message': 'Path: C:\\Users\\test'},
        ]

        doc_ids = []
        for doc in original_docs:
            doc_ids.append(table.insert(doc.copy()))

        csv_path = tmp_file('csv')
        table.export_csv(csv_path, encoding='utf-8')

        # Import and verify exact match
        table2 = db.table('messages_imported')
        table2.import_csv(csv_path, encoding='utf-8')

        for doc_id, original in zip(doc_ids, original_docs):
            imported = table2.get(doc_id=doc_id)
            assert imported['user'] == original['user']
            assert imported['message'] == original['message']

    def test_jsonl_special_characters(self, db, tmp_file):
        """Test JSONL with special characters like emojis and symbols."""
        table = db.table('messages')

        original_docs = [
            {'user': 'Alice', 'message': 'Hello! How are you?'},
            {'user': 'Bob', 'message': 'Great! See you tomorrow.'},
            {'user': 'Charlie', 'message': 'Price: $100.00 (50% off)'},
            {'user': 'Diana', 'message': 'Email: test@example.com'},
            {'user': 'Eve', 'message': 'Path: C:\\Users\\test'},
        ]

        doc_ids = []
        for doc in original_docs:
            doc_ids.append(table.insert(doc.copy()))

        jsonl_path = tmp_file('jsonl')
        table.export_jsonl(jsonl_path, encoding='utf-8')

        # Import and verify exact match
        table2 = db.table('messages_imported')
        table2.import_jsonl(jsonl_path, encoding='utf-8')

        for doc_id, original in zip(doc_ids, original_docs):
            imported = table2.get(doc_id=doc_id)
            assert imported['user'] == original['user']
            assert imported['message'] == original['message']


class TestExactDataIntegrity:
    """Tests for exact data integrity during import/export operations."""

    def test_csv_all_data_types_exact_values(self, db, tmp_file):
        """Test CSV round-trip preserves exact values for all data types.

        Note: CSV format has a known limitation where empty strings and None
        values are indistinguishable (both become empty CSV cells). After
        import, both are converted to None. Use JSONL format if you need
        to preserve the distinction between empty strings and None.
        """
        table = db.table('test')

        original = {
            'string_field': 'hello world',
            'int_field': 42,
            'float_field': 3.14159,
            'bool_true': True,
            'bool_false': False,
            'none_field': None,
            'list_field': [1, 2, 3, 'four', 5.0],
            'dict_field': {'nested': 'value', 'count': 10},
            'empty_list': [],
            'empty_dict': {},
        }

        doc_id = table.insert(original.copy())

        csv_path = tmp_file('csv')
        table.export_csv(csv_path)

        # Import into new table
        table2 = db.table('test_imported')
        table2.import_csv(csv_path)

        imported = table2.get(doc_id=doc_id)

        # Assert each field exactly
        assert imported['string_field'] == original['string_field']
        assert imported['int_field'] == original['int_field']
        assert isinstance(imported['int_field'], int)
        assert imported['float_field'] == original['float_field']
        assert isinstance(imported['float_field'], float)
        assert imported['bool_true'] is True
        assert imported['bool_false'] is False
        assert imported['none_field'] is None
        assert imported['list_field'] == original['list_field']
        assert imported['dict_field'] == original['dict_field']
        assert imported['empty_list'] == original['empty_list']
        assert imported['empty_dict'] == original['empty_dict']

    def test_csv_empty_string_becomes_none(self, db, tmp_file):
        """Test that CSV format converts empty strings to None (known limitation)."""
        table = db.table('test')
        doc_id = table.insert({'name': 'test', 'empty': ''})

        csv_path = tmp_file('csv')
        table.export_csv(csv_path)

        table2 = db.table('test_imported')
        table2.import_csv(csv_path)

        imported = table2.get(doc_id=doc_id)
        # CSV limitation: empty strings become None after round-trip
        assert imported['empty'] is None

    def test_jsonl_all_data_types_exact_values(self, db, tmp_file):
        """Test JSONL round-trip preserves exact values for all data types."""
        table = db.table('test')

        original = {
            'string_field': 'hello world',
            'int_field': 42,
            'float_field': 3.14159,
            'bool_true': True,
            'bool_false': False,
            'none_field': None,
            'list_field': [1, 2, 3, 'four', 5.0],
            'dict_field': {'nested': 'value', 'count': 10},
            'empty_string': '',
            'empty_list': [],
            'empty_dict': {},
            'nested_structure': {
                'level1': {
                    'level2': {
                        'level3': ['deep', 'value']
                    }
                }
            },
        }

        doc_id = table.insert(original.copy())

        jsonl_path = tmp_file('jsonl')
        table.export_jsonl(jsonl_path)

        # Import into new table
        table2 = db.table('test_imported')
        table2.import_jsonl(jsonl_path)

        imported = table2.get(doc_id=doc_id)

        # Assert each field exactly
        assert imported['string_field'] == original['string_field']
        assert imported['int_field'] == original['int_field']
        assert isinstance(imported['int_field'], int)
        assert imported['float_field'] == original['float_field']
        assert isinstance(imported['float_field'], float)
        assert imported['bool_true'] is True
        assert imported['bool_false'] is False
        assert imported['none_field'] is None
        assert imported['list_field'] == original['list_field']
        assert imported['dict_field'] == original['dict_field']
        assert imported['empty_string'] == ''
        assert imported['empty_list'] == original['empty_list']
        assert imported['empty_dict'] == original['empty_dict']
        assert imported['nested_structure'] == original['nested_structure']

    def test_multiple_documents_exact_field_mapping(self, db, tmp_file):
        """Test that field values are correctly mapped to their documents."""
        table = db.table('test')

        docs = [
            {'name': 'Doc1', 'value': 100, 'type': 'A'},
            {'name': 'Doc2', 'value': 200, 'type': 'B'},
            {'name': 'Doc3', 'value': 300, 'type': 'A'},
            {'name': 'Doc4', 'value': 400, 'type': 'C'},
        ]

        doc_ids = [table.insert(doc.copy()) for doc in docs]

        # Test CSV
        csv_path = tmp_file('csv')
        table.export_csv(csv_path)

        table_csv = db.table('test_csv')
        table_csv.import_csv(csv_path)

        for doc_id, original in zip(doc_ids, docs):
            imported = table_csv.get(doc_id=doc_id)
            assert imported['name'] == original['name'], f"CSV: name mismatch for {doc_id}"
            assert imported['value'] == original['value'], f"CSV: value mismatch for {doc_id}"
            assert imported['type'] == original['type'], f"CSV: type mismatch for {doc_id}"

        # Test JSONL
        jsonl_path = tmp_file('jsonl')
        table.export_jsonl(jsonl_path)

        table_jsonl = db.table('test_jsonl')
        table_jsonl.import_jsonl(jsonl_path)

        for doc_id, original in zip(doc_ids, docs):
            imported = table_jsonl.get(doc_id=doc_id)
            assert imported['name'] == original['name'], f"JSONL: name mismatch for {doc_id}"
            assert imported['value'] == original['value'], f"JSONL: value mismatch for {doc_id}"
            assert imported['type'] == original['type'], f"JSONL: type mismatch for {doc_id}"
