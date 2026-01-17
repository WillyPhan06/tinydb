"""
This module implements field validation for TinyDB documents.

It provides validation rules for enforcing document schemas, including
required field checks and type validation.
"""

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Set,
    Type,
    Union,
)


class ValidationError(Exception):
    """
    Exception raised when document validation fails.

    This exception provides detailed information about which field(s) failed
    validation and why.

    :param field: The name of the field that failed validation
    :param message: A descriptive error message
    :param value: The value that failed validation (optional)
    """

    def __init__(
        self,
        field: str,
        message: str,
        value: Any = None
    ):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"Validation failed for field '{field}': {message}")


class FieldValidator:
    """
    Defines validation rules for a single field in a document.

    A FieldValidator can check:
    - Whether a field is required (must be present and not None)
    - Whether a field value matches an expected type

    Example usage:

    >>> from tinydb.validation import FieldValidator
    >>> # Create a validator for a required string field
    >>> name_validator = FieldValidator(required=True, field_type=str)
    >>> # Create a validator for an optional integer field
    >>> age_validator = FieldValidator(field_type=int)
    >>> # Create a validator with custom validation
    >>> positive_validator = FieldValidator(
    ...     field_type=int,
    ...     validator=lambda x: x > 0,
    ...     validator_message="must be positive"
    ... )
    """

    # Mapping of type names to actual Python types for convenience
    TYPE_MAP: Dict[str, Type] = {
        'str': str,
        'string': str,
        'int': int,
        'integer': int,
        'float': float,
        'bool': bool,
        'boolean': bool,
        'list': list,
        'dict': dict,
    }

    def __init__(
        self,
        required: bool = False,
        field_type: Optional[Union[Type, str]] = None,
        validator: Optional[Callable[[Any], bool]] = None,
        validator_message: Optional[str] = None
    ):
        """
        Create a new field validator.

        :param required: If True, the field must be present and not None
        :param field_type: The expected type of the field value. Can be a
                          Python type (str, int, float, bool, list, dict) or
                          a string name ('str', 'string', 'int', 'integer',
                          'float', 'bool', 'boolean', 'list', 'dict')
        :param validator: Optional custom validation function that takes the
                         field value and returns True if valid, False otherwise
        :param validator_message: Custom error message for the validator
        """
        self.required = required
        self._validator = validator
        self._validator_message = validator_message or "failed custom validation"

        # Resolve field_type from string to actual type if needed
        if field_type is None:
            self._field_type = None
        elif isinstance(field_type, str):
            type_lower = field_type.lower()
            if type_lower not in self.TYPE_MAP:
                valid_types = ', '.join(sorted(self.TYPE_MAP.keys()))
                raise ValueError(
                    f"Unknown type '{field_type}'. "
                    f"Valid types are: {valid_types}"
                )
            self._field_type = self.TYPE_MAP[type_lower]
        else:
            self._field_type = field_type

    @property
    def field_type(self) -> Optional[Type]:
        """Get the expected field type."""
        return self._field_type

    def validate(self, field_name: str, value: Any, is_present: bool) -> None:
        """
        Validate a field value against this validator's rules.

        :param field_name: The name of the field being validated
        :param value: The value to validate
        :param is_present: Whether the field is present in the document
        :raises ValidationError: If validation fails
        """
        # Check required constraint
        if self.required:
            if not is_present:
                raise ValidationError(
                    field_name,
                    "field is required but missing"
                )
            if value is None:
                raise ValidationError(
                    field_name,
                    "field is required but value is None",
                    value
                )

        # If field is not present and not required, skip other validations
        if not is_present:
            return

        # If value is None and field is not required, skip type/custom checks
        if value is None:
            return

        # Check type constraint
        if self._field_type is not None:
            # Special handling for int vs float:
            # Python's bool is a subclass of int, so we need to handle that
            if self._field_type == int and isinstance(value, bool):
                raise ValidationError(
                    field_name,
                    f"expected type 'int', got 'bool'",
                    value
                )
            # For float, we accept both int and float (common numeric coercion)
            if self._field_type == float:
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValidationError(
                        field_name,
                        f"expected type 'float', got '{type(value).__name__}'",
                        value
                    )
            elif not isinstance(value, self._field_type):
                raise ValidationError(
                    field_name,
                    f"expected type '{self._field_type.__name__}', "
                    f"got '{type(value).__name__}'",
                    value
                )

        # Check custom validator
        if self._validator is not None:
            if not self._validator(value):
                raise ValidationError(
                    field_name,
                    self._validator_message,
                    value
                )


class Schema:
    """
    Defines a validation schema for documents in a table.

    A Schema contains validators for multiple fields and can validate
    entire documents against those rules.

    Example usage:

    >>> from tinydb.validation import Schema, FieldValidator
    >>> # Create a schema for user documents
    >>> user_schema = Schema({
    ...     'name': FieldValidator(required=True, field_type=str),
    ...     'age': FieldValidator(field_type=int),
    ...     'email': FieldValidator(required=True, field_type=str),
    ... })
    >>> # Validate a document
    >>> user_schema.validate({'name': 'John', 'email': 'john@example.com'})
    """

    def __init__(self, fields: Optional[Dict[str, FieldValidator]] = None):
        """
        Create a new schema.

        :param fields: A dictionary mapping field names to FieldValidator
                      instances
        """
        self._fields: Dict[str, FieldValidator] = fields or {}

    def add_field(self, name: str, validator: FieldValidator) -> 'Schema':
        """
        Add a field validator to the schema.

        :param name: The field name
        :param validator: The FieldValidator for this field
        :returns: self (for method chaining)
        """
        self._fields[name] = validator
        return self

    def remove_field(self, name: str) -> 'Schema':
        """
        Remove a field validator from the schema.

        :param name: The field name to remove
        :returns: self (for method chaining)
        """
        self._fields.pop(name, None)
        return self

    def get_field(self, name: str) -> Optional[FieldValidator]:
        """
        Get a field validator by name.

        :param name: The field name
        :returns: The FieldValidator or None if not found
        """
        return self._fields.get(name)

    @property
    def field_names(self) -> Set[str]:
        """Get the set of field names in this schema."""
        return set(self._fields.keys())

    def validate(self, document: Mapping) -> None:
        """
        Validate a document against this schema.

        :param document: The document to validate
        :raises ValidationError: If any field fails validation
        """
        for field_name, validator in self._fields.items():
            is_present = field_name in document
            value = document.get(field_name)
            validator.validate(field_name, value, is_present)

    def validate_update(self, fields: Mapping) -> None:
        """
        Validate fields being used to update a document.

        This method validates only the fields being updated.

        :param fields: The fields being updated (key-value pairs)
        :raises ValidationError: If any field fails validation
        """
        # For updates, we only validate fields that are being changed
        for field_name, value in fields.items():
            validator = self._fields.get(field_name)
            if validator is not None:
                # The field is present since it's in the update fields
                validator.validate(field_name, value, is_present=True)

        # Check if any required fields are being set to None
        for field_name, validator in self._fields.items():
            if validator.required and field_name in fields:
                if fields[field_name] is None:
                    raise ValidationError(
                        field_name,
                        "field is required but value is None",
                        None
                    )

    def __len__(self) -> int:
        """Get the number of fields in this schema."""
        return len(self._fields)

    def __contains__(self, field_name: str) -> bool:
        """Check if a field is in this schema."""
        return field_name in self._fields

    def __repr__(self) -> str:
        fields_repr = ', '.join(sorted(self._fields.keys()))
        return f"<Schema fields=[{fields_repr}]>"
