import asyncio
import datetime
import typing
from collections.abc import Sequence

import numpy.typing as npt
from pydantic import SecretStr

import resqml_objects.v201 as ro
from energistics.etp.v12.datatypes.data_array_types import (
    DataArrayMetadata,
)
from energistics.etp.v12.datatypes.object import (
    Dataspace,
    Resource,
)
from energistics.uris import DataObjectURI, DataspaceURI
from pyetp.utils_arrays import LogicalArrayDTypes
from rddms_io.client import rddms_connect
from rddms_io.data_types import LinkedObjects, RDDMSModel


class RDDMSClientSync:
    """
    Synchronized version of the [`RDDMSClient`][rddms_io.client.RDDMSClient].
    The purpose of this client is to serve the same high-level endpoints
    towards the RDDMS server as `RDDMSClient`, but without the need to use
    `async` and `await`. Only the methods (and parameters) that can be wrapped
    in a single call are included in `RDDMSClientSync`. The client works by
    passing in the same parameters as
    [`rddms_connect`][rddms_io.client.rddms_connect] to the constructor, and
    then calling the methods without using `await`. The parameters to
    `RDDMSClientSync` are the same as to `rddms_connect`.

    Parameters
    ----------
    uri
        The uri to the RDDMS server. This should be the uri to a websockets
        endpoint to an ETP server.
    data_partition_id
        The data partition id used when connecting to the OSDU open-etp-server
        in multi-partition mode. Default is `None`.
    authorization
        Bearer token used for authenticating to the RDDMS server. This token
        should be on the form `"Bearer 1234..."`. Default is `None`.
    etp_timeout
        The timeout in seconds for when to stop waiting for a message from the
        server. Setting it to `None` will persist the connection indefinetly.
        Default is `None`.
    max_message_size
        The maximum number of bytes for a single websockets message. Default is
        `2**20` corresponding to `1` MiB.



    Notes
    -----
    The `authorization`-token (if using, e.g.,
    [`msal`](https://learn.microsoft.com/en-us/entra/msal/python)` will have an
    expiration time. If this expiration time is met, the client needs to be
    updated with a fresh token.

    Whenever you call one of the methods of this client, it will set up a new
    ETP session, call the relevant method from the asynchronous client, and
    then tear down the connection. If you find yourself repeatedly calling
    multiple methods in a succesive fashion, consider switching to the
    asynchronous client as this will be much faster.

    See Also
    --------
    [`RDDMSClient`][rddms_io.client.RDDMSClient]:
        The asynchronous driver class which `RDDMSClientSync` wraps.

    [`rddms_connect`][rddms_io.client.rddms_connect]:
        The connection class to set up the asynchronous `RDDMSClient`.
    """

    def __init__(
        self,
        uri: str,
        data_partition_id: str | None = None,
        authorization: str | SecretStr | None = None,
        etp_timeout: float | None = None,
        max_message_size: float = 2**20,
    ) -> None:
        if isinstance(authorization, SecretStr):
            authorization = authorization
        else:
            authorization = SecretStr(authorization)

        self.connection_args = dict(
            uri=uri,
            data_partition_id=data_partition_id,
            authorization=authorization,
            etp_timeout=etp_timeout,
            max_message_size=max_message_size,
        )

    def create_dataspace(
        self,
        dataspace_uri: str | DataspaceURI,
        legal_tags: list[str] = [],
        other_relevant_data_countries: list[str] = [],
        owners: list[str] = [],
        viewers: list[str] = [],
        ignore_if_exists: bool = False,
    ) -> None:
        """
        Method creating a new dataspace on the ETP server. This function is
        limited to creating a single dataspace with optional access-control
        list (ACL) information.

        Parameters
        ----------
        dataspace_uri
            The ETP dataspace uri, or path, to create. If it is a dataspace
            path (on the form `'foo/bar'`) it will be converted to the
            dataspace uri `"eml:///dataspace('foo/bar')"`.
        legal_tags
            List of legal tag strings for the ACL. The default is an empty
            list.
        other_relevant_data_countries: list[str]
            List of data countries for the ACL. The default is an empty list.
        owners
            List of owners ACL. The default is an empty list.
        viewers
            List of viewers ACL. The default is an empty list.
        ignore_if_exists
            When `True` the method silently ignores any `ETPError` with error
            code `5` (`EINVALID_ARGUMENT`). This error occurs if the dataspace
            already exists on the server. Otherwise, all errors are raised.
            Default is `False`.

        See Also
        --------
        [`RDDMSClient.create_dataspace`][rddms_io.client.RDDMSClient.create_dataspace]:
            The asynchronous counterpart.
        """

        async def create_dataspace() -> None:
            async with rddms_connect(**self.connection_args) as rddms_client:
                return await rddms_client.create_dataspace(
                    dataspace_uri=dataspace_uri,
                    legal_tags=legal_tags,
                    other_relevant_data_countries=other_relevant_data_countries,
                    owners=owners,
                    viewers=viewers,
                    ignore_if_exists=ignore_if_exists,
                )

        return asyncio.run(create_dataspace())

    def delete_dataspace(self, dataspace_uri: DataspaceURI | str) -> None:
        """
        Method deleting a dataspace.

        Parameters
        ----------
        dataspace_uri
            The ETP dataspace uri, or path, for the dataspace to delete. If it
            is a dataspace path (on the form `'foo/bar'`) it will be converted
            to the dataspace uri `"eml:///dataspace('foo/bar')"`.

        See Also
        --------
        [`RDDMSClient.delete_dataspace`][rddms_io.client.RDDMSClient.delete_dataspace]:
            The asynchronous version of this method.
        """

        async def delete_dataspace() -> None:
            async with rddms_connect(**self.connection_args) as rddms_client:
                return await rddms_client.delete_dataspace(dataspace_uri=dataspace_uri)

        return asyncio.run(delete_dataspace())

    def list_dataspaces(
        self, store_last_write_filter: datetime.datetime | int | None = None
    ) -> list[Dataspace]:
        """
        Method used to list all dataspaces on the ETP-server.

        Parameters
        ----------
        store_last_write_filter
            A parameter that can be used to limit the results to only include
            dataspaces that were written to after the time specified in the
            filter. The default is `None`, meaning all dataspaces will be
            included.

        Returns
        -------
        list[Dataspace]
            A list of ETP `Dataspace`-data objects. See section 23.43.10 of the
            ETP v1.2 standards documentation for an accurate description of the
            different fields.

        See Also
        --------
        [`RDDMSClient.list_dataspaces`][rddms_io.client.RDDMSClient.list_dataspaces]:
            The asynchronous version of this method.
        """

        async def list_dataspaces() -> None:
            async with rddms_connect(**self.connection_args) as rddms_client:
                return await rddms_client.list_dataspaces(
                    store_last_write_filter=store_last_write_filter
                )

        return asyncio.run(list_dataspaces())

    def list_objects_under_dataspace(
        self,
        dataspace_uri: DataspaceURI | str,
        data_object_types: list[str | typing.Type[ro.AbstractCitedDataObject]] = [],
        count_objects: bool = True,
        store_last_write_filter: int | None = None,
    ) -> list[Resource]:
        """
        This method will list all objects under a given dataspace.

        Parameters
        ----------
        dataspace_uri
            The uri of the dataspace to list objects.
        data_object_types
            Object types to look for. This can either be a list of strings,
            e.g., `["eml20.*", "resqml20.obj_Grid2dRepresentation"]` to query
            all Energistic Common version 2.0-objects and
            `obj_Grid2dRepresentation`-objects from RESQML v2.0.1, or it can be
            a list of objects from `resqml_objects.v201`, e.g.,
            `[ro.obj_Grid2dRepresentation, ro.obj_LocalDepth3dCrs]`. Default is
            `[]`, i.e., an empty list which means that all data object types
            will be returned.
        count_objects
            Toggle if the number of target and source objects should be
            counted. Default is `True`.
        store_last_write_filter
            Filter to only include objects that are written after the provided
            datetime or timestamp. Default is `None`, meaning no filter is
            applied. Note that the timestamp should be in microsecond
            resolution.

        Returns
        -------
        list[Resource]
            A list of
            [`Resource`][energistics.etp.v12.datatypes.object.Resource]-objects.

        See Also
        --------
        [`RDDMSClient.list_objects_under_dataspace`][rddms_io.client.RDDMSClient.list_objects_under_dataspace]:
            The asynchronous version of this method.
        """

        async def list_objects_under_dataspace() -> list[Resource]:
            async with rddms_connect(**self.connection_args) as rddms_client:
                return await rddms_client.list_objects_under_dataspace(
                    dataspace_uri=dataspace_uri,
                    data_object_types=data_object_types,
                    count_objects=count_objects,
                    store_last_write_filter=store_last_write_filter,
                )

        return asyncio.run(list_objects_under_dataspace())

    def list_linked_objects(
        self,
        start_uri: DataObjectURI | str,
        data_object_types: list[str | typing.Type[ro.AbstractCitedDataObject]] = [],
        store_last_write_filter: datetime.datetime | int | None = None,
        depth: int = 1,
    ) -> LinkedObjects:
        """
        Method listing all objects that are linked to the provided object uri.
        That is, starting from the object indexed by the uri `start_uri` it
        finds all objects (sources) that links to it, and all objects (targets)
        it links to.

        Parameters
        ----------
        start_uri: DataObjectURI | str
            An ETP data object uri to start the query from.
        data_object_types: list[str | typing.Type[ro.AbstractCitedDataObject]]
            A filter to limit which types of objects to include in the results.
            As a string it is on the form `eml20.obj_EpcExternalPartReference`
            for a specific object, or `eml20.*` for all Energistics Common
            objects. For the RESQML v2.0.1 objects it is similarlarly
            `resqml20.*`, or a specific type instead of the wildcard `*`. This
            can also be classes from `resqml_objects.v201`, in which case the
            filter will be constructed. Default is `[]`, meaning no filter is
            applied.
        store_last_write_filter: datetime.datetime | int | None
            Filter to only include objects that are written after the provided
            datetime or timestamp. Default is `None`, meaning no filter is
            applied. Note that the timestamp should be in microsecond
            resolution.
        depth: int
            The number of links to return. Setting `depth = 1` will only return
            targets and sources that are directly linked to the start object.
            With `depth = 2` we get links to objects that linkes to the targets
            and sources of the start object. Default is `1`.

        Returns
        -------
        LinkedObjects
            A container (`NamedTuple`) with resources and edges for the sources
            and targets of the start-object.

        See Also
        --------
        [`RDDMSClient.list_linked_objects`][rddms_io.client.RDDMSClient.list_linked_objects]:
            The asynchronous version of this method.
        """

        async def list_linked_objects() -> LinkedObjects:
            async with rddms_connect(**self.connection_args) as rddms_client:
                return await rddms_client.list_linked_objects(
                    start_uri=start_uri,
                    data_object_types=data_object_types,
                    store_last_write_filter=store_last_write_filter,
                    depth=depth,
                )

        return asyncio.run(list_linked_objects())

    def list_array_metadata(
        self,
        ml_uris: list[str | DataObjectURI],
    ) -> dict[str, dict[str, DataArrayMetadata]]:
        """
        Method used for listing array metadata for all connected arrays to the
        provided data object uris. This method downloads the data objects from
        the uris, and calls `RDDMSClient.list_object_array_metadata` to get the
        actual metadata. If the objects have already been downloaded, then
        using `RDDMSClientSync.list_object_array_metadata` will be more
        efficient.

        The purpose of this method is to provide a more convenient way of
        exploring an RDDMS server without needing to handle data objects. It is
        recommended to use `RDDMSClientSync.list_object_array_metadata` if the
        objects have already been downloaded.

        Parameters
        ----------
        ml_uris
            A list of ETP data object uris.

        Returns
        -------
        dict[str, dict[str, DataArrayMetadata]]
            A dictionary indexed by the data object uri, containing a new
            dictionary with the path in resource as the key and the metadata
            (the ETP datatype `DataArrayMetadata`) as the value. Note that if
            there is no array connected to a data object uri, there will be no
            entry in the returned dict for this uri.

        See Also
        --------
        [`RDDMSClient.list_array_metadata`][rddms_io.client.RDDMSClient.list_array_metadata]:
            The asynchronous version of this method.

        [`RDDMSClientSync.list_object_array_metadata`][rddms_io.sync_client.RDDMSClientSync.list_object_array_metadata]:
            A similar method that fetches the metadata from the objects
            themselves along with a dataspace uri. It is recommended to use
            `list_object_array_metadata` if you already have the objects in
            memory.
        """

        async def list_array_metadata() -> dict[str, dict[str, DataArrayMetadata]]:
            async with rddms_connect(**self.connection_args) as rddms_client:
                return await rddms_client.list_array_metadata(ml_uris=ml_uris)

        return asyncio.run(list_array_metadata())

    def list_object_array_metadata(
        self,
        dataspace_uri: str | DataspaceURI,
        ml_objects: Sequence[ro.AbstractCitedDataObject],
    ) -> dict[str, dict[str, DataArrayMetadata]]:
        """
        Method used for listing array metadata for all connected arrays to the
        provided RESQML-objects. This method works by taking in a dataspace uri
        and the objects themselves (instead of their uris) as they would need
        to be downloaded to look up which arrays they link to.

        Parameters
        ----------
        dataspace_uri: str | DataspaceURI
            The ETP dataspace uri where the objects are located.
        ml_objects: Sequence[ro.AbstractCitedDataObject]
            A list (or any sequence) of objects that links to arrays.

        Returns
        -------
        dict[str, dict[str, DataArrayMetadata]]
            A dictionary indexed by the data object uri, containing a new
            dictionary with the path in resource as the key and the metadata
            (the ETP datatype `DataArrayMetadata`) as the value.

        See Also
        --------
        [`RDDMSClient.list_object_array_metadata`][rddms_io.client.RDDMSClient.list_object_array_metadata]:
            The asynchronous version of this method.

        [`RDDMSClientSync.list_array_metadata`][rddms_io.sync_client.RDDMSClientSync.list_array_metadata]:
            A similar method that looks up array metadata needing only the uris
            of the objects.
        """

        async def list_object_array_metadata() -> dict[
            str, dict[str, DataArrayMetadata]
        ]:
            async with rddms_connect(**self.connection_args) as rddms_client:
                return await rddms_client.list_object_array_metadata(
                    dataspace_uri=dataspace_uri, ml_objects=ml_objects
                )

        return asyncio.run(list_object_array_metadata())

    def delete_model(
        self,
        ml_uris: list[str | DataObjectURI],
        prune_contained_objects: bool = False,
        debounce: bool | float = False,
    ) -> None:
        """
        Method used for deleting a set of objects on an ETP server. In order
        for the deletion to be successful the objects to be deleted can not
        leave any dangling source-objects. That is, there can be no objects
        left on the ETP server that references the deleted objects.

        Parameters
        ----------
        ml_uris
            A list of ETP data object uris to delete.
        prune_contained_objects
            See section 9.3.4 in the ETP v1.2 standards documentation for an
            accurate description of this parameter. Default is `False` meaning
            no pruning is done.
        debounce
            Parameter to decide if `RDDMSClient.delete_model` should retry
            starting a transaction if it initially fails. See
            `RDDMSClient.start_transaction` for a more in-depth explanation of
            the parameter. Default is `False`, i.e., no debouncing will occur
            and the method will fail if it is unable to start a transaction.

        See Also
        --------
        [`RDDMSClient.delete_model`][rddms_io.client.RDDMSClient.delete_model]:
            The asynchronous version of this method.
        """

        async def delete_model() -> None:
            async with rddms_connect(**self.connection_args) as rddms_client:
                return await rddms_client.delete_model(
                    ml_uris=ml_uris,
                    prune_contained_objects=prune_contained_objects,
                    handle_transaction=True,
                    debounce=debounce,
                )

        return asyncio.run(delete_model())

    def upload_model(
        self,
        dataspace_uri: str | DataspaceURI,
        ml_objects: Sequence[ro.AbstractCitedDataObject],
        data_arrays: typing.Mapping[
            str, Sequence[npt.NDArray[LogicalArrayDTypes]]
        ] = {},
        debounce: bool | float = False,
    ) -> list[str]:
        """
        Method for uploading data to an ETP server. This method takes in a
        dataspace uri (for uploading to multiple dataspaces you need to call
        `RDDMSClient.upload_model` multiple times), a set of RESQML-objects,
        and a mapping of data arrays that are indexed by their path in resource
        (which is found in the RESQML-objects as well).

        Parameters
        ----------
        dataspace_uri
            An ETP dataspace uri.
        ml_objects
            A sequence of RESQML v2.0.1-objects.
        data_arrays
            A mapping, e.g., a dictionary, of data arrays where the path in
            resources (found in the RESQML-objects) are the keys. Default is
            `{}`, meaning that only the RESQML-objects will be uploaded.
        debounce
            Parameter to decide if `RDDMSClient.upload_model` should retry
            starting a transaction if it initially fails. See
            `RDDMSClient.start_transaction` for a more in-depth explanation of
            the parameter. Default is `False`, i.e., no debouncing will occur
            and the method will fail if it is unable to start a transaction.

        Returns
        -------
        list[str]
            A list of ETP data object uris to the uploaded objects.

        See Also
        --------
        [`RDDMSClient.upload_model`][rddms_io.client.RDDMSClient.upload_model]:
            The asynchronous version of this method.

        [`RDDMSClientSync.download_models`][rddms_io.sync_client.RDDMSClientSync.download_models]:
            The reverse operation.
        """

        async def upload_model() -> list[str]:
            async with rddms_connect(**self.connection_args) as rddms_client:
                return await rddms_client.upload_model(
                    dataspace_uri=dataspace_uri,
                    ml_objects=ml_objects,
                    data_arrays=data_arrays,
                    handle_transaction=True,
                    debounce=debounce,
                )

        return asyncio.run(upload_model())

    def download_models(
        self,
        ml_uris: list[str | DataObjectURI],
        download_arrays: bool = False,
        download_linked_objects: bool = False,
    ) -> list[RDDMSModel]:
        """
        Download RESQML-models from the RDDMS server. A model in this sense is
        a RESQML-object (with a given uri) and possibly with any connected
        arrays and referenced objects.

        Parameters
        ----------
        ml_uris
            A list of ETP data object uris.
        download_arrays
            A flag to toggle if any referenced arrays should be download
            alongside the RESQML-objects. Setting to `True` will populate
            [`RDDMSModel.arrays`][rddms_io.data_types.RDDMSModel.arrays]-field
            with a dictionary with the `path_in_hdf_file` as the key, and the
            arrays as the values. If the flag is set to `False` no arrays will
            be downloaded, and the corresponding field will be empty. Default
            is `False`.
        download_linked_objects
            Flag to toggle if linked objects (target-objects), i.e., objects
            referenced by objects from `ml_uris`. For example, setting the flag
            to `True` and passing in a single `obj_Grid2dRepresentation`-uri in
            the `ml_uris` will try to download any linked coordinate systems or
            any other referenced objects. The linked objects will be added to
            [`RDDMSModel.linked_models`][rddms_io.data_types.RDDMSModel.linked_models],
            along with arrays if `download_arrays=True`.
            Note that if any of the linked objects are already in `ml_uris`
            they will be included both as a top-level `RDDMSModel`, and as a
            linked-model under a model that references it. The method only
            looks for objects linked one level down (corresponding to `depth =
            1` in `GetResources`), and it will ignore
            `obj_EpcExternalPartReference`- and
            `EpcExternalPartReference`-objects.  Default is `False` meaning no
            linked objects will be downloaded.

        Returns
        -------
        list[RDDMSModel]
            A list of [`RDDMSModel`][rddms_io.data_types.RDDMSModel]-objects.

        See Also
        --------
        [`RDDMSClient.download_models`][rddms_io.client.RDDMSClient.download_models]:
            The asynchronous version of this method.
        """

        async def download_models() -> None:
            async with rddms_connect(**self.connection_args) as rddms_client:
                return await rddms_client.download_models(
                    ml_uris=ml_uris,
                    download_arrays=download_arrays,
                    download_linked_objects=download_linked_objects,
                )

        return asyncio.run(download_models())
