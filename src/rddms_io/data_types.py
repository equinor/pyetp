import typing

from energistics.etp.v12.datatypes.object import Edge, Resource


class LinkedObjects(typing.NamedTuple):
    start_uri: str
    source_resources: list[Resource]
    source_edges: list[Edge]
    target_resources: list[Resource]
    target_edges: list[Edge]
