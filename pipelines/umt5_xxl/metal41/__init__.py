"""Shared umT5-XXL download and int8 pack."""


PIPELINE = "umt5_xxl"


from umt5_xxl.metal41.commands import main

__all__ = ["main"]
