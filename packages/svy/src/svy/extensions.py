# src/svy/extensions.py
import warnings

from .core import Sample


# This dictionary holds the registered accessors
# Key: name (e.g., 'sae'), Value: The Class
_sample_accessors = {}


def register_sample_accessor(name):
    """
    Decorator to register a custom accessor for the Sample class.

    Usage:
    @register_sample_accessor("sae")
    class SAEAccessor: ...
    """

    def decorator(accessor_class):
        if hasattr(Sample, name):
            warnings.warn(f"Accessor '{name}' is already defined on Sample. Overwriting.")

        _sample_accessors[name] = accessor_class

        # This is the magic:We create a property on the Sample class dynamically
        def accessor_getter(self):
            # Cache the accessor instance on the object so we don't recreate it every time
            cache_name = f"_accessor_{name}"
            if not hasattr(self, cache_name):
                setattr(self, cache_name, accessor_class(self))
            return getattr(self, cache_name)

        # Attach the property to the Sample class
        setattr(Sample, name, property(accessor_getter))

        return accessor_class

    return decorator
