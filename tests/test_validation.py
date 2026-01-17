"""
Tests for the TinyDB validation module.
"""

import pytest

from tinydb import TinyDB, where, ValidationError, FieldValidator, Schema
from tinydb.storages import MemoryStorage


@pytest.fixture
def db():
    """Create a fresh in-memory database for each test."""
    db_ = TinyDB(storage=MemoryStorage)
    yield db_
    db_.close()


class TestFieldValidator:
    """Tests for the FieldValidator class."""

    def test_required_field_missing(self):
        """Test that required fields raise error when missing."""
        validator = FieldValidator(required=True)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate('name', None, is_present=False)
        assert exc_info.value.field == 'name'
        assert 'required but missing' in exc_info.value.message

    def test_required_field_none(self):
        """Test that required fields raise error when value is None."""
        validator = FieldValidator(required=True)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate('name', None, is_present=True)
        assert exc_info.value.field == 'name'
        assert 'required but value is None' in exc_info.value.message

    def test_required_field_present(self):
        """Test that required fields pass when present with value."""
        validator = FieldValidator(required=True)
        # Should not raise
        validator.validate('name', 'John', is_present=True)

    def test_optional_field_missing(self):
        """Test that optional fields pass when missing."""
        validator = FieldValidator(required=False)
        # Should not raise
        validator.validate('age', None, is_present=False)

    def test_type_string(self):
        """Test type validation for strings."""
        validator = FieldValidator(field_type=str)
        validator.validate('name', 'John', is_present=True)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate('name', 123, is_present=True)
        assert "expected type 'str'" in exc_info.value.message
        assert "got 'int'" in exc_info.value.message

    def test_type_int(self):
        """Test type validation for integers."""
        validator = FieldValidator(field_type=int)
        validator.validate('age', 25, is_present=True)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate('age', '25', is_present=True)
        assert "expected type 'int'" in exc_info.value.message
        assert "got 'str'" in exc_info.value.message

    def test_type_int_rejects_bool(self):
        """Test that int type rejects boolean values."""
        validator = FieldValidator(field_type=int)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate('count', True, is_present=True)
        assert "expected type 'int'" in exc_info.value.message
        assert "got 'bool'" in exc_info.value.message

    def test_type_float(self):
        """Test type validation for floats."""
        validator = FieldValidator(field_type=float)
        validator.validate('price', 19.99, is_present=True)
        # Integers should also be accepted for float type
        validator.validate('price', 20, is_present=True)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate('price', '19.99', is_present=True)
        assert "expected type 'float'" in exc_info.value.message

    def test_type_float_rejects_bool(self):
        """Test that float type rejects boolean values."""
        validator = FieldValidator(field_type=float)
        with pytest.raises(ValidationError) as exc_info:
            validator.validate('value', False, is_present=True)
        assert "expected type 'float'" in exc_info.value.message

    def test_type_bool(self):
        """Test type validation for booleans."""
        validator = FieldValidator(field_type=bool)
        validator.validate('active', True, is_present=True)
        validator.validate('active', False, is_present=True)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate('active', 1, is_present=True)
        assert "expected type 'bool'" in exc_info.value.message

    def test_type_list(self):
        """Test type validation for lists."""
        validator = FieldValidator(field_type=list)
        validator.validate('tags', ['a', 'b'], is_present=True)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate('tags', 'a,b', is_present=True)
        assert "expected type 'list'" in exc_info.value.message

    def test_type_dict(self):
        """Test type validation for dicts."""
        validator = FieldValidator(field_type=dict)
        validator.validate('metadata', {'key': 'value'}, is_present=True)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate('metadata', [('key', 'value')], is_present=True)
        assert "expected type 'dict'" in exc_info.value.message

    def test_type_as_string(self):
        """Test that type can be specified as a string."""
        validator = FieldValidator(field_type='string')
        validator.validate('name', 'John', is_present=True)

        validator = FieldValidator(field_type='integer')
        validator.validate('age', 25, is_present=True)

        validator = FieldValidator(field_type='INT')  # Case insensitive
        validator.validate('count', 10, is_present=True)

    def test_invalid_type_string(self):
        """Test that invalid type string raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FieldValidator(field_type='invalid_type')
        assert "Unknown type 'invalid_type'" in str(exc_info.value)

    def test_custom_validator(self):
        """Test custom validation function."""
        validator = FieldValidator(
            field_type=int,
            validator=lambda x: x > 0,
            validator_message="must be positive"
        )
        validator.validate('count', 5, is_present=True)

        with pytest.raises(ValidationError) as exc_info:
            validator.validate('count', -1, is_present=True)
        assert exc_info.value.message == "must be positive"

    def test_combined_required_and_type(self):
        """Test combining required and type validation."""
        validator = FieldValidator(required=True, field_type=str)

        # Missing field
        with pytest.raises(ValidationError) as exc_info:
            validator.validate('name', None, is_present=False)
        assert 'required but missing' in exc_info.value.message

        # Wrong type
        with pytest.raises(ValidationError) as exc_info:
            validator.validate('name', 123, is_present=True)
        assert "expected type 'str'" in exc_info.value.message

        # Valid
        validator.validate('name', 'John', is_present=True)

    def test_optional_none_skips_type_check(self):
        """Test that None value skips type check for optional fields."""
        validator = FieldValidator(field_type=int)
        # Should not raise - None is allowed for optional fields
        validator.validate('age', None, is_present=True)


class TestSchema:
    """Tests for the Schema class."""

    def test_create_schema_with_dict(self):
        """Test creating a schema with a dictionary."""
        schema = Schema({
            'name': FieldValidator(required=True, field_type=str),
            'age': FieldValidator(field_type=int),
        })
        assert len(schema) == 2
        assert 'name' in schema
        assert 'age' in schema

    def test_add_and_remove_field(self):
        """Test adding and removing fields from schema."""
        schema = Schema()
        schema.add_field('name', FieldValidator(required=True))
        assert 'name' in schema

        schema.remove_field('name')
        assert 'name' not in schema

    def test_validate_document_success(self):
        """Test validating a valid document."""
        schema = Schema({
            'name': FieldValidator(required=True, field_type=str),
            'age': FieldValidator(field_type=int),
        })
        # Should not raise
        schema.validate({'name': 'John', 'age': 30})
        schema.validate({'name': 'Jane'})  # age is optional

    def test_validate_document_missing_required(self):
        """Test that missing required field raises error."""
        schema = Schema({
            'name': FieldValidator(required=True, field_type=str),
        })
        with pytest.raises(ValidationError) as exc_info:
            schema.validate({'age': 30})
        assert exc_info.value.field == 'name'

    def test_validate_document_wrong_type(self):
        """Test that wrong type raises error."""
        schema = Schema({
            'age': FieldValidator(field_type=int),
        })
        with pytest.raises(ValidationError) as exc_info:
            schema.validate({'age': 'thirty'})
        assert exc_info.value.field == 'age'

    def test_validate_update_fields(self):
        """Test validating update fields."""
        schema = Schema({
            'name': FieldValidator(required=True, field_type=str),
            'age': FieldValidator(field_type=int),
        })
        # Should not raise - only validates fields being updated
        schema.validate_update({'age': 25})

        # Should raise for wrong type
        with pytest.raises(ValidationError):
            schema.validate_update({'age': 'twenty-five'})

    def test_validate_update_setting_required_to_none(self):
        """Test that setting a required field to None raises error."""
        schema = Schema({
            'name': FieldValidator(required=True, field_type=str),
        })
        with pytest.raises(ValidationError) as exc_info:
            schema.validate_update({'name': None})
        assert 'required but value is None' in exc_info.value.message

    def test_schema_repr(self):
        """Test schema string representation."""
        schema = Schema({
            'name': FieldValidator(),
            'age': FieldValidator(),
        })
        repr_str = repr(schema)
        assert 'Schema' in repr_str
        assert 'age' in repr_str
        assert 'name' in repr_str


class TestTableValidation:
    """Tests for validation integrated into Table operations."""

    def test_set_schema(self, db):
        """Test setting a schema on a table."""
        table = db.table('users')
        schema = Schema({
            'name': FieldValidator(required=True, field_type=str),
        })
        table.set_schema(schema)
        assert table.schema is schema

    def test_clear_schema(self, db):
        """Test clearing a schema from a table."""
        table = db.table('users')
        table.set_schema(Schema({'name': FieldValidator()}))
        table.clear_validation()
        assert table.schema is None

    def test_add_validation(self, db):
        """Test adding validation rules using convenience method."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)
        table.add_validation('age', field_type=int)

        assert table.schema is not None
        assert 'name' in table.schema
        assert 'age' in table.schema

    def test_remove_validation(self, db):
        """Test removing validation rules."""
        table = db.table('users')
        table.add_validation('name', required=True)
        table.add_validation('age', field_type=int)
        table.remove_validation('age')

        assert 'name' in table.schema
        assert 'age' not in table.schema

    def test_insert_valid_document(self, db):
        """Test inserting a valid document with schema."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)
        table.add_validation('age', field_type=int)

        doc_id = table.insert({'name': 'John', 'age': 30})
        assert doc_id == 1
        assert table.get(doc_id=doc_id)['name'] == 'John'

    def test_insert_invalid_document_missing_required(self, db):
        """Test that inserting document missing required field raises error."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)

        with pytest.raises(ValidationError) as exc_info:
            table.insert({'age': 30})
        assert exc_info.value.field == 'name'
        # Ensure nothing was inserted
        assert len(table) == 0

    def test_insert_invalid_document_wrong_type(self, db):
        """Test that inserting document with wrong type raises error."""
        table = db.table('users')
        table.add_validation('age', field_type=int)

        with pytest.raises(ValidationError) as exc_info:
            table.insert({'name': 'John', 'age': 'thirty'})
        assert exc_info.value.field == 'age'
        assert len(table) == 0

    def test_insert_multiple_valid_documents(self, db):
        """Test inserting multiple valid documents."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)

        doc_ids = table.insert_multiple([
            {'name': 'John'},
            {'name': 'Jane'},
            {'name': 'Bob'},
        ])
        assert len(doc_ids) == 3
        assert len(table) == 3

    def test_insert_multiple_one_invalid(self, db):
        """Test that insert_multiple fails fast on invalid document."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)

        with pytest.raises(ValidationError):
            table.insert_multiple([
                {'name': 'John'},
                {'name': 123},  # Invalid
                {'name': 'Bob'},
            ])
        # None should be inserted due to fail-fast validation
        assert len(table) == 0

    def test_update_valid_fields(self, db):
        """Test updating with valid fields."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)
        table.add_validation('age', field_type=int)

        table.insert({'name': 'John', 'age': 30})
        table.update({'age': 31}, where('name') == 'John')

        doc = table.get(where('name') == 'John')
        assert doc['age'] == 31

    def test_update_invalid_type(self, db):
        """Test that updating with wrong type raises error."""
        table = db.table('users')
        table.add_validation('age', field_type=int)

        table.insert({'name': 'John', 'age': 30})

        with pytest.raises(ValidationError) as exc_info:
            table.update({'age': 'thirty-one'}, where('name') == 'John')
        assert exc_info.value.field == 'age'

        # Original value should be unchanged
        doc = table.get(where('name') == 'John')
        assert doc['age'] == 30

    def test_update_required_to_none(self, db):
        """Test that updating required field to None raises error."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)

        table.insert({'name': 'John'})

        with pytest.raises(ValidationError):
            table.update({'name': None}, where('name') == 'John')

    def test_update_with_callable_valid(self, db):
        """Test that callable updates work when result is valid."""
        table = db.table('users')
        table.add_validation('count', field_type=int)

        table.insert({'count': 5})
        # Callable updates should work when result is valid
        table.update(lambda doc: doc.update({'count': doc['count'] + 1}))

        doc = table.get(doc_id=1)
        assert doc['count'] == 6

    def test_update_with_callable_invalid_type(self, db):
        """Test that callable updates fail when result has wrong type."""
        table = db.table('users')
        table.add_validation('count', field_type=int)

        table.insert({'count': 5})

        # Callable that changes type should fail validation
        with pytest.raises(ValidationError) as exc_info:
            table.update(lambda doc: doc.update({'count': 'not a number'}))
        assert exc_info.value.field == 'count'

        # Original value should be unchanged
        doc = table.get(doc_id=1)
        assert doc['count'] == 5

    def test_update_with_callable_removes_required(self, db):
        """Test that callable updates fail when removing required field."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)

        table.insert({'name': 'John', 'age': 30})

        # Callable that removes required field should fail
        with pytest.raises(ValidationError) as exc_info:
            table.update(lambda doc: doc.pop('name'))
        assert exc_info.value.field == 'name'

        # Original value should be unchanged
        doc = table.get(doc_id=1)
        assert doc['name'] == 'John'

    def test_update_with_callable_sets_required_to_none(self, db):
        """Test that callable updates fail when setting required field to None."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)

        table.insert({'name': 'John'})

        # Callable that sets required field to None should fail
        with pytest.raises(ValidationError) as exc_info:
            table.update(lambda doc: doc.update({'name': None}))
        assert exc_info.value.field == 'name'

        # Original value should be unchanged
        doc = table.get(doc_id=1)
        assert doc['name'] == 'John'

    def test_update_multiple_with_callable_valid(self, db):
        """Test update_multiple with callable that produces valid results."""
        table = db.table('users')
        table.add_validation('count', field_type=int)

        table.insert({'name': 'John', 'count': 5})
        table.insert({'name': 'Jane', 'count': 10})

        # Callable updates with valid results
        table.update_multiple([
            (lambda doc: doc.update({'count': doc['count'] + 1}), where('name') == 'John'),
            (lambda doc: doc.update({'count': doc['count'] * 2}), where('name') == 'Jane'),
        ])

        assert table.get(where('name') == 'John')['count'] == 6
        assert table.get(where('name') == 'Jane')['count'] == 20

    def test_update_multiple_with_callable_invalid(self, db):
        """Test that update_multiple with callable fails on invalid result."""
        table = db.table('users')
        table.add_validation('count', field_type=int)

        table.insert({'name': 'John', 'count': 5})

        # Callable that produces invalid result should fail
        with pytest.raises(ValidationError):
            table.update_multiple([
                (lambda doc: doc.update({'count': 'invalid'}), where('name') == 'John'),
            ])

        # Original value should be unchanged
        doc = table.get(where('name') == 'John')
        assert doc['count'] == 5

    def test_update_multiple_valid(self, db):
        """Test update_multiple with valid fields."""
        table = db.table('users')
        table.add_validation('status', field_type=str)

        table.insert({'name': 'John', 'type': 'admin'})
        table.insert({'name': 'Jane', 'type': 'user'})

        table.update_multiple([
            ({'status': 'active'}, where('type') == 'admin'),
            ({'status': 'pending'}, where('type') == 'user'),
        ])

        assert table.get(where('name') == 'John')['status'] == 'active'
        assert table.get(where('name') == 'Jane')['status'] == 'pending'

    def test_update_multiple_invalid(self, db):
        """Test that update_multiple fails on invalid fields."""
        table = db.table('users')
        table.add_validation('status', field_type=str)

        table.insert({'name': 'John', 'type': 'admin'})

        with pytest.raises(ValidationError):
            table.update_multiple([
                ({'status': 'active'}, where('type') == 'admin'),
                ({'status': 123}, where('type') == 'user'),  # Invalid
            ])

    def test_upsert_insert_path(self, db):
        """Test upsert when inserting new document."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)

        # Should fail - missing required field
        with pytest.raises(ValidationError):
            table.upsert({'age': 30}, where('name') == 'John')

        # Should succeed with valid document
        table.upsert({'name': 'John', 'age': 30}, where('name') == 'John')
        assert len(table) == 1

    def test_upsert_update_path(self, db):
        """Test upsert when updating existing document."""
        table = db.table('users')
        table.add_validation('age', field_type=int)

        table.insert({'name': 'John', 'age': 30})

        # Should fail - wrong type
        with pytest.raises(ValidationError):
            table.upsert({'name': 'John', 'age': 'thirty-one'}, where('name') == 'John')

        # Original should be unchanged
        assert table.get(where('name') == 'John')['age'] == 30

    def test_no_schema_allows_anything(self, db):
        """Test that tables without schema allow any documents."""
        table = db.table('users')
        # No schema set - anything goes
        table.insert({'name': 123, 'age': 'thirty', 'extra': [1, 2, 3]})
        assert len(table) == 1

    def test_extra_fields_allowed(self, db):
        """Test that fields not in schema are allowed."""
        table = db.table('users')
        table.add_validation('name', required=True, field_type=str)

        # Extra 'email' field not in schema should be allowed
        doc_id = table.insert({'name': 'John', 'email': 'john@example.com'})
        doc = table.get(doc_id=doc_id)
        assert doc['email'] == 'john@example.com'

    def test_custom_validator_integration(self, db):
        """Test custom validator function in table."""
        table = db.table('users')
        table.add_validation(
            'age',
            field_type=int,
            validator=lambda x: 0 <= x <= 150,
            validator_message="age must be between 0 and 150"
        )

        # Valid age
        table.insert({'name': 'John', 'age': 30})

        # Invalid age
        with pytest.raises(ValidationError) as exc_info:
            table.insert({'name': 'Jane', 'age': 200})
        assert 'age must be between 0 and 150' in exc_info.value.message


class TestValidationError:
    """Tests for the ValidationError exception."""

    def test_error_message_format(self):
        """Test error message formatting."""
        error = ValidationError('name', 'field is required')
        assert str(error) == "Validation failed for field 'name': field is required"

    def test_error_attributes(self):
        """Test error attributes."""
        error = ValidationError('age', 'wrong type', value='thirty')
        assert error.field == 'age'
        assert error.message == 'wrong type'
        assert error.value == 'thirty'
