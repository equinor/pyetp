import datetime
import typing
from dataclasses import dataclass

from energistics.etp.v12.datatypes import DataValue
from energistics.etp.v12.datatypes.object import Dataspace, Edge, Resource


class LinkedObjects(typing.NamedTuple):
    start_uri: str
    source_resources: list[Resource]
    source_edges: list[Edge]
    target_resources: list[Resource]
    target_edges: list[Edge]
