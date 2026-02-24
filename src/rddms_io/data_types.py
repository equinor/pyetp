import typing

from energistics.etp.v12.datatypes.object import Edge, Resource


class LinkedObjects(typing.NamedTuple):
    """
    Container for results after calling
    [`RDDMSClient.list_linked_objects`][rddms_io.client.RDDMSClient.list_linked_objects].
    Objects in RESQML are structured as graphs. Objects can point to other
    objects. If object A has a reference to object B, we say that A is a
    _source_ to B, but B is also a _target_ to A.

    Attributes
    ----------
    start_uri
        The uri of the object that we are looking for links to.
    self_resource
        The [`Resource`][energistics.etp.v12.datatypes.object.Resource] of the
        object identified by `start_uri`.
    source_resources
        A list of
        [`Resource`][energistics.etp.v12.datatypes.object.Resource]-objects
        that act as sources to the object referenced by `start_uri`.
    source_edges
        A list of [`Edge`][energistics.etp.v12.datatypes.object.Edge]-objects
        that describes how the sources links to the object referenced by
        `start_uri`.
    target_resources
        A list of
        [`Resource`][energistics.etp.v12.datatypes.object.Resource]-objects
        that act as targets to the object referenced by `start_uri`.
    target_edges
        A list of [`Edge`][energistics.etp.v12.datatypes.object.Edge]-objects
        that describes how the targets links to the object referenced by
        `start_uri`.
    """

    start_uri: str
    self_resource: Resource
    source_resources: list[Resource]
    source_edges: list[Edge]
    target_resources: list[Resource]
    target_edges: list[Edge]
