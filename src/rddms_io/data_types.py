import typing

import numpy.typing as npt

import resqml_objects.v201 as ro
from energistics.etp.v12.datatypes.object import Edge, Resource
from pyetp import utils_arrays


class RDDMSModel(typing.NamedTuple):
    """
    Container for results after calling
    [`RDDMSClient.download_models`][rddms_io.client.RDDMSClient.download_models].

    Attributes
    ----------
    obj
        The main object in the model, i.e., the object that is referenced by a
        passed in uri in the argument `ml_uris` in
        [`RDDMSClient.download_models`][rddms_io.client.RDDMSClient.download_models].
    arrays
        A dictionary with arrays referenced by `obj` (if any). The keys are
        found in the field `path_in_hdf_file` of any
        [`Hdf5Dataset`][resqml_objects.v201.generated.Hdf5Dataset]-objects
        occuring in `obj`.
    linked_models
        A list of `RDDMSModel`-objects where the `RDDMSModel.obj`-field is an
        object referenced by the current `obj`. These linked models might also
        contain arrays and linked models themselves.
    """

    obj: ro.AbstractCitedDataObject
    arrays: dict[str, list[npt.NDArray[utils_arrays.LogicalArrayDTypes]]]
    linked_models: list["RDDMSModel"]


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
