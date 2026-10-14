"""refusalscope: a research harness for stock-versus-ablated language model pairs.

The harness measures whether removing the refusal direction from a model damages structured
tool calling and whether it raises compliance with instructions injected into tool results.
The core uses only the standard library and numpy.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
