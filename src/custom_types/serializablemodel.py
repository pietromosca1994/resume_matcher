from dataclasses import asdict, is_dataclass
from typing import Dict, Any, get_origin, get_args
from enum import Enum
import pandas as pd
import json
import logging 

class DictSerializable:
    # def to_dict(self) -> Dict:
    #     def convert(obj):
    #         if isinstance(obj, Enum):
    #             return obj.value
    #         elif is_dataclass(obj):
    #             return {k: convert(v) for k, v in asdict(obj).items()}
    #         elif isinstance(obj, list):
    #             return [convert(v) for v in obj]
    #         elif isinstance(obj, dict):
    #             return {k: convert(v) for k, v in obj.items()}
    #         elif isinstance(obj, pd.DataFrame):
    #             return obj.to_dict(orient="records")
    #         elif isinstance(obj, pd.Series):
    #             return obj.tolist()
    #         else:
    #             return obj

    #     return {k: convert(v) for k, v in asdict(self).items() if v is not None}

    def to_dict(self) -> Dict[str, Any]:
        def convert(value: Any) -> Any:
            """Recursively converts values into serializable Python types."""
            if value is None:
                return None

            # Handle enums
            if isinstance(value, Enum):
                return value.value

            # Handle nested SerializableModel-like objects
            if hasattr(value, "to_dict") and callable(value.to_dict):
                try:
                    return value.to_dict()
                except Exception:
                    pass  # Fallback to raw value if something goes wrong

            # Handle lists and tuples
            if isinstance(value, (list, tuple)):
                return [convert(v) for v in value]

            # Handle dicts
            if isinstance(value, dict):
                return {convert(k): convert(v) for k, v in value.items()}

            # Handle pandas objects
            if isinstance(value, pd.DataFrame):
                return value.to_dict(orient="records")

            if isinstance(value, pd.Series):
                return value.tolist()

            # Handle strings — including potential JSON-encoded strings
            if isinstance(value, str):
                stripped = value.strip()
                # Try to decode only if it looks like JSON
                if (stripped.startswith("{") and stripped.endswith("}")) or \
                (stripped.startswith("[") and stripped.endswith("]")):
                    try:
                        parsed = json.loads(stripped)
                        # Recursively convert if parsed successfully
                        return convert(parsed)
                    except (json.JSONDecodeError, TypeError):
                        pass  # Fall back to raw string if not valid JSON
                return value  # Always return string as-is

            # Everything else (numbers, bools, etc.)
            return value

        # Handle SQLAlchemy models
        if hasattr(self.__class__, "__mapper__"):
            return {
                col.key: convert(getattr(self, col.key))
                for col in self.__mapper__.columns
                if getattr(self, col.key) is not None
            }

        # Handle dataclasses
        if is_dataclass(self):
            return {
                k: convert(v)
                for k, v in asdict(self).items()
                if v is not None
            }

        # Handle general Python objects
        if hasattr(self, "__dict__"):
            return {
                k: convert(v)
                for k, v in self.__dict__.items()
                if v is not None
            }

        # Fallback for unhandled cases
        return {}

    @classmethod
    def from_dict(cls, dict_obj: Dict[str, Any]):
        """
        Create an instance from a dictionary.
        Works for SQLAlchemy, dataclasses, and general models.
        """

        # Collect type hints (if available)
        field_types = getattr(cls, "__annotations__", {})

        kwargs = {}

        for key, value in dict_obj.items():
            expected_type = field_types.get(key, None)

            # Handle nulls / missing values early
            if value is None:
                kwargs[key] = None
                continue

            # Enum fields
            if isinstance(expected_type, type) and issubclass(expected_type, Enum):
                try:
                    kwargs[key] = expected_type(value)
                except Exception:
                    kwargs[key] = value
                continue

            # Nested SerializableModel or dataclass with from_dict
            if hasattr(expected_type, "from_dict") and isinstance(value, dict):
                kwargs[key] = expected_type.from_dict(value)
                continue

            # Dicts (possibly of dataclasses)
            if get_origin(expected_type) is dict and isinstance(value, dict):
                sub_type = get_args(expected_type)[1] if len(get_args(expected_type)) > 1 else None
                if hasattr(sub_type, "from_dict"):
                    kwargs[key] = {k: sub_type.from_dict(v) for k, v in value.items()}
                else:
                    kwargs[key] = value
                continue

            # Lists (possibly of enums or dataclasses)
            if get_origin(expected_type) is list and isinstance(value, list):
                sub_type = get_args(expected_type)[0] if get_args(expected_type) else None
                if isinstance(sub_type, type) and issubclass(sub_type, Enum):
                    kwargs[key] = [sub_type(v) for v in value]
                elif hasattr(sub_type, "from_dict"):
                    kwargs[key] = [sub_type.from_dict(v) for v in value]
                else:
                    kwargs[key] = value
                continue

            # Pandas
            if expected_type == pd.DataFrame and isinstance(value, list):
                kwargs[key] = pd.DataFrame(value)
                continue

            if expected_type == pd.Series and isinstance(value, list):
                kwargs[key] = pd.Series(value)
                continue

            # JSON-like strings (decode if possible)
            if isinstance(value, str):
                stripped = value.strip()
                if (stripped.startswith("{") and stripped.endswith("}")) or \
                (stripped.startswith("[") and stripped.endswith("]")):
                    try:
                        parsed = json.loads(stripped)
                        kwargs[key] = parsed
                        continue
                    except json.JSONDecodeError:
                        pass
                # Normal string, keep as-is
                kwargs[key] = value
                continue

            # Fallback for primitives and unmapped SQLAlchemy fields
            kwargs[key] = value

        return cls(**kwargs)

class JsonSerializable:
    def to_json(self, path: str):
        try:
            with open(path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logging.info(f'Saved {self.__class__.__name__} to {path}')
        except Exception as e:
            logging.error(f'Failed to save {self.__class__.__name__} to {path}: {e}')

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        logging.info(f'Loaded {cls.__name__} from {path}')
        return cls.from_dict(config_dict)
    
class SerializableModel(DictSerializable, JsonSerializable):
    """Base class for dataclasses with dict and JSON serialization support."""
    pass