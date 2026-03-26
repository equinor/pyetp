import asyncio
import datetime
import logging
import sys
import typing
import uuid
from collections.abc import AsyncGenerator, Sequence
from types import TracebackType

import numpy as np
import numpy.typing as npt
from pydantic import SecretStr

import resqml_objects.v201 as ro
from energistics.etp.v12.datatypes import ArrayOfString, DataValue, Uuid
from energistics.etp.v12.datatypes.data_array_types import (
    DataArrayIdentifier,
    DataArrayMetadata,
    GetDataSubarraysType,
    PutDataArraysType,
    PutDataSubarraysType,
    PutUninitializedDataArrayType,
)
from energistics.etp.v12.datatypes.object import (
    ContextInfo,
    ContextScopeKind,
    DataObject,
    Dataspace,
    RelationshipKind,
    Resource,
)
from energistics.etp.v12.protocol.data_array import (
    GetDataArrayMetadata,
    GetDataArrayMetadataResponse,
    GetDataArrays,
    GetDataArraysResponse,
    GetDataSubarrays,
    GetDataSubarraysResponse,
    PutDataArrays,
    PutDataArraysResponse,
    PutDataSubarrays,
    PutDataSubarraysResponse,
    PutUninitializedDataArrays,
    PutUninitializedDataArraysResponse,
)
from energistics.etp.v12.protocol.dataspace import (
    DeleteDataspaces,
    DeleteDataspacesResponse,
    GetDataspaces,
    GetDataspacesResponse,
    PutDataspaces,
    PutDataspacesResponse,
)
from energistics.etp.v12.protocol.discovery import (
    GetResources,
    GetResourcesEdgesResponse,
    GetResourcesResponse,
)
from energistics.etp.v12.protocol.store import (
    Chunk,
    DeleteDataObjects,
    DeleteDataObjectsResponse,
    GetDataObjects,
    GetDataObjectsResponse,
    PutDataObjects,
    PutDataObjectsResponse,
)
from energistics.etp.v12.protocol.transaction import (
    CommitTransaction,
    CommitTransactionResponse,
    RollbackTransaction,
    RollbackTransactionResponse,
    StartTransaction,
    StartTransactionResponse,
)
from energistics.types import ETPNumpyArrayType
from energistics.uris import DataObjectURI, DataspaceURI
from pyetp.client import (
    ETPClient,
    ETPError,
    ETPMessageTooLarge,
    etp_connect,
    timeout_intervals,
)
from pyetp.errors import (
    ETPTransactionFailure,
    parse_and_raise_response_errors,
)
from rddms_io.block_array import get_array_block_sizes
from rddms_io.data_types import LinkedObjects, RDDMSModel
from resqml_objects import parse_resqml_v201_object, serialize_resqml_v201_object
from resqml_objects.v201.utils import find_data_object_references, find_hdf5_datasets

logger = logging.getLogger(__name__)

if sys.version_info >= (3, 12):
    from itertools import batched
else:
    # Construct `batched` for Python 3.11 as it was introduced in Python 3.12.
    # We make it specialized for our use-case.
    def batched(iterable: bytes, n: int) -> typing.Iterator[tuple[int, ...]]:
        num_chunks = len(iterable) // n + int(len(iterable) % n != 0)
        for i in range(num_chunks):
            yield tuple(iterable[slice(i * n, (i + 1) * n)])


class RDDMSClient:
    """
    Client using ETP to communicate with an RDDMS (Reservoir Domain Data
    Management Services) server. It is specifically tailored towards the OSDU
    [open-etp-server](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/open-etp-server)
    and made with the intention to make it easier to interact with RDDMS by
    exposing ergonomic user-facing functions.

    Notes
    -----
    The client is meant to be set up via
    [`rddms_connect`][rddms_io.client.rddms_connect].


    Parameters
    ----------
    etp_client
        An instance of [`ETPClient`][pyetp.client.ETPClient].
    """

    def __init__(self, etp_client: ETPClient) -> None:
        self.etp_client = etp_client

    async def close(self) -> None:
        """
        Method used for manual closing of the ETP-connection when the client
        has been set up outside a context manager. For example, if the client
        has been made via an `await`-statement then this method should be used
        to stop the connection.

        Examples
        --------

            rddms_client = await rddms_connect(...)
            ...
            await rddms_client.close()
        """
        await self.etp_client.close()

    async def list_dataspaces(
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
        """

        if isinstance(store_last_write_filter, datetime.datetime):
            # Convert `datetime`-object to a microsecond resolution timestamp.
            store_last_write_filter = int(store_last_write_filter.timestamp() * 1e6)

        responses = await self.etp_client.send_and_recv(
            GetDataspaces(store_last_write_filter=store_last_write_filter)
        )
        parse_and_raise_response_errors(
            responses,
            GetDataspacesResponse,
            "RDDMSClient.list_dataspaces",
        )
        # Here we know that `responses` has type `list[GetDataspacesResponse]`
        dataspaces = [
            ds
            for response in responses
            # Explicitly cast `response` to `GetDataspacesResponse` (this has
            # been checked with the `parse_and_raise_response_errors`-function
            # above.
            for ds in typing.cast(GetDataspacesResponse, response).dataspaces
        ]

        return dataspaces

    async def delete_dataspace(self, dataspace_uri: DataspaceURI | str) -> None:
        """
        Method deleting a dataspace.

        Parameters
        ----------
        dataspace_uri
            The ETP dataspace uri, or path, for the dataspace to delete. If it
            is a dataspace path (on the form `'foo/bar'`) it will be converted
            to the dataspace uri `"eml:///dataspace('foo/bar')"`.
        """
        dataspace_uri = str(DataspaceURI.from_any_etp_uri(dataspace_uri))
        responses = await self.etp_client.send_and_recv(
            DeleteDataspaces(uris={dataspace_uri: dataspace_uri})
        )
        parse_and_raise_response_errors(
            responses,
            DeleteDataspacesResponse,
            "RDDMSClient.delete_dataspace",
        )
        assert any(
            [
                # Cast `response` to `DeleteDataspacesResponse`. This has been
                # checked with `parse_and_raise_response_errors` above.
                dataspace_uri in typing.cast(DeleteDataspacesResponse, response).success
                for response in responses
            ]
        )

    async def create_dataspace(
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
        """
        dataspace_uri = DataspaceURI.from_any_etp_uri(dataspace_uri)

        # A UTC timestamp in microseconds.
        now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e6)

        def acl_parsing(acl_key: str, acl: list[str]) -> dict[str, DataValue]:
            if not acl:
                return {}
            return {acl_key: DataValue(item=ArrayOfString(values=acl))}

        custom_data = {
            **acl_parsing("legaltags", legal_tags),
            **acl_parsing("otherRelevantDataCountries", other_relevant_data_countries),
            **acl_parsing("owners", owners),
            **acl_parsing("viewers", viewers),
        }

        try:
            responses = await self.etp_client.send_and_recv(
                PutDataspaces(
                    dataspaces={
                        str(dataspace_uri): Dataspace(
                            uri=str(dataspace_uri),
                            path=dataspace_uri.dataspace,
                            store_created=now,
                            store_last_write=now,
                            custom_data=custom_data,
                        )
                    }
                )
            )
        except ETPError as e:
            if ignore_if_exists and e.code == 5:
                logger.info(
                    f"Ignoring error in RDDMSClient.create_dataspace with message '{e}'"
                )
                return

            raise

        parse_and_raise_response_errors(
            responses, PutDataspacesResponse, "RDDMSClient.create_dataspace"
        )
        assert any(
            [
                str(dataspace_uri)
                # Cast `response` to `PutDataspacesResponse`. This has been
                # checked with `parse_and_raise_response_errors` above.
                in typing.cast(PutDataspacesResponse, response).success
                for response in responses
            ]
        )

    async def start_transaction(
        self,
        dataspace_uri: str | DataspaceURI,
        read_only: bool,
        debounce: bool | float = False,
    ) -> uuid.UUID:
        """
        Method issuing a `StartTransaction`-ETP message, with optional
        debouncing to retry in case the dataspace is occupied with a different
        write transaction. Note that this method (unlike the raw ETP-message)
        is limited to starting a transaction on a single dataspace.

        Parameters
        ----------
        dataspace_uri: str | DataspaceURI
            A dataspace URI, either as a string or a `DataspaceURI`-object.
        read_only: bool
            Set to `False` for writing, and `True` for reading. It is mandatory
            to use a transaction when writing, but optional for reading.
        debounce: bool | float
            Flag to toggle debouncing or maximum total debouncing time.
            If set to `True`, the client will continue to debounce forever
            until a transaction is started. Setting debounce to a floating
            point number will set a maximum total debouncing time (it will
            potentially retry several times within that window). Default is
            `False`, i.e., no debouncing is done, and failure to start a
            transaction will result in an error.

        Returns
        -------
        uuid.UUID
            A standard library UUID with the transaction uuid.
        """

        dataspace_uri = str(DataspaceURI.from_any_etp_uri(dataspace_uri))

        if isinstance(debounce, bool):
            total_timeout = None
        else:
            total_timeout = debounce

        for ti in timeout_intervals(total_timeout):
            try:
                responses = await self.etp_client.send_and_recv(
                    StartTransaction(
                        read_only=read_only, dataspace_uris=[dataspace_uri]
                    ),
                )
            except ETPError as e:
                # Check if the error corresponds to the ETP Error
                # `EMAX_TRANSACTIONS_EXCEEDED`.
                if e.code != 15:
                    raise

                if debounce:
                    logger.info(f"Failed to start transaction retrying in {ti} seconds")
                    await asyncio.sleep(ti)
                    continue

                raise

            parse_and_raise_response_errors(
                responses, StartTransactionResponse, "RDDMSClient.start_transaction"
            )

            assert len(responses) == 1
            response = responses[0]
            assert isinstance(response, StartTransactionResponse)

            if not response.successful:
                raise ETPTransactionFailure(str(response.failure_reason))

            transaction_uuid = uuid.UUID(str(response.transaction_uuid.root))
            logger.debug("Started transaction with uuid: {transaction_uuid}")
            return transaction_uuid

        raise ETPTransactionFailure(
            f"Failed to start transaction after {total_timeout} seconds",
        )

    async def commit_transaction(
        self,
        transaction_uuid: bytes | str | uuid.UUID | Uuid,
    ) -> None:
        """
        Method for commiting a transaction after completing all tasks that
        needs to be synchronized between the client and the server.

        Parameters
        ----------
        transaction_uuid: bytes | str | uuid.UUID | Uuid
            The transaction uuid for the current transaction. This will
            typically be the uuid from the
            `RDDMSClient.start_transaction`-method.
        """
        responses = await self.etp_client.send_and_recv(
            CommitTransaction(transaction_uuid=Uuid(transaction_uuid)),
        )

        assert len(responses) == 1
        response = responses[0]
        assert isinstance(response, CommitTransactionResponse)

        if not response.successful:
            raise ETPTransactionFailure(str(response))

        logger.debug("Commited transaction with uuid: {transaction_uuid}")

    async def rollback_transaction(
        self, transaction_uuid: bytes | str | uuid.UUID | Uuid
    ) -> None:
        """
        Method for cancelling a running transaction. This will tell the server
        that it should disregard any changes incurred by the current
        transaction.

        Parameters
        ----------
        transaction_uuid: bytes | str | uuid.UUID | Uuid
            The transaction uuid for the current transaction. This will
            typically be the uuid from the
            `RDDMSClient.start_transaction`-method.
        """
        responses = await self.etp_client.send_and_recv(
            RollbackTransaction(transaction_uuid=Uuid(transaction_uuid))
        )

        parse_and_raise_response_errors(
            responses, RollbackTransactionResponse, "RDDMSClient.rollback_transaction"
        )

        assert len(responses) == 1
        response = responses[0]
        assert isinstance(response, RollbackTransactionResponse)

        if not response.successful:
            raise ETPTransactionFailure(str(response))

    async def list_objects_under_dataspace(
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
        """
        dataspace_uri = str(DataspaceURI.from_any_etp_uri(dataspace_uri))
        data_object_types = [
            dot if isinstance(dot, str) else dot.get_qualified_type()
            for dot in data_object_types
        ]

        gr = GetResources(
            context=ContextInfo(
                uri=dataspace_uri,
                depth=1,  # Ignored when `scope="self"`.
                data_object_types=data_object_types,
                # TODO: Check if `navigable_edges` give any different results.
                navigable_edges="Primary",
            ),
            scope="self",
            count_objects=count_objects,
            store_last_write_filter=store_last_write_filter,
            # Use the `list_linked_objects`-method below to see edges.
            include_edges=False,
        )

        responses = await self.etp_client.send_and_recv(gr)

        parse_and_raise_response_errors(
            responses, GetResourcesResponse, "RDDMSClient.list_objects_under_dataspace"
        )
        return [
            resource
            for response in responses
            # We have checked that `responses` is of type
            # `list[GetResourcesResponse]` in the function
            # `parse_and_raise_response_errors` above.
            for resource in typing.cast(GetResourcesResponse, response).resources
        ]

    async def list_linked_objects(
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
        """
        start_uri = str(start_uri)
        data_object_types = [
            dot if isinstance(dot, str) else dot.get_qualified_type()
            for dot in data_object_types
        ]
        if isinstance(store_last_write_filter, datetime.datetime):
            # Convert `datetime`-object to a microsecond resolution timestamp.
            store_last_write_filter = int(store_last_write_filter.timestamp() * 1e6)

        gr_sources = GetResources(
            context=ContextInfo(
                uri=start_uri,
                depth=depth,
                data_object_types=data_object_types,
                navigable_edges=RelationshipKind.PRIMARY,
            ),
            # Setting the scope to `SOURCES_OR_SELF` returns the start-object
            # resource _and_ the edge(s) between the start-object and its
            # sources.
            scope=ContextScopeKind.SOURCES_OR_SELF,
            count_objects=True,
            store_last_write_filter=store_last_write_filter,
            include_edges=True,
        )

        gr_targets = GetResources(
            context=ContextInfo(
                uri=start_uri,
                depth=depth,
                data_object_types=data_object_types,
                navigable_edges=RelationshipKind.PRIMARY,
            ),
            scope=ContextScopeKind.TARGETS_OR_SELF,
            count_objects=True,
            store_last_write_filter=store_last_write_filter,
            include_edges=True,
        )

        task_responses = await asyncio.gather(
            self.etp_client.send_and_recv(gr_sources),
            self.etp_client.send_and_recv(gr_targets),
        )

        sources_responses = task_responses[0]
        targets_responses = task_responses[1]

        source_edges = [
            e
            for grer in filter(
                lambda e: isinstance(e, GetResourcesEdgesResponse), sources_responses
            )
            # The filter above only selects instances of
            # `GetResourcesEdgesResponse`, so the cast is only included for
            # type checkers.
            for e in typing.cast(GetResourcesEdgesResponse, grer).edges
        ]
        source_resources = [
            r
            for grr in filter(
                lambda e: isinstance(e, GetResourcesResponse), sources_responses
            )
            for r in typing.cast(GetResourcesResponse, grr).resources
        ]

        target_edges = [
            e
            for grer in filter(
                lambda e: isinstance(e, GetResourcesEdgesResponse), targets_responses
            )
            for e in typing.cast(GetResourcesEdgesResponse, grer).edges
        ]
        target_resources = [
            r
            for grr in filter(
                lambda e: isinstance(e, GetResourcesResponse), targets_responses
            )
            for r in typing.cast(GetResourcesResponse, grr).resources
        ]

        self_resource = next(filter(lambda sr: sr.uri == start_uri, source_resources))

        # Remove "self" from list of resources.
        source_resources = list(
            filter(
                lambda sr: sr.uri != start_uri,
                source_resources,
            )
        )
        target_resources = list(
            filter(
                lambda tr: tr.uri != start_uri,
                target_resources,
            )
        )

        return LinkedObjects(
            start_uri=start_uri,
            self_resource=self_resource,
            source_resources=source_resources,
            source_edges=source_edges,
            target_resources=target_resources,
            target_edges=target_edges,
        )

    async def list_array_metadata(
        self,
        ml_uris: list[str | DataObjectURI],
    ) -> dict[str, dict[str, DataArrayMetadata]]:
        """
        Method used for listing array metadata for all connected arrays to the
        provided data object uris. This method downloads the data objects from
        the uris, and calls `RDDMSClient.list_object_array_metadata` to get the
        actual metadata. If the objects have already been downloaded, then
        using `RDDMSClient.list_object_array_metadata` will be more efficient.

        The purpose of this method is to provide a more convenient way of
        exploring an RDDMS server without needing to handle data objects. It is
        recommended to use `RDDMSClient.list_object_array_metadata` if the
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
        [`RDDMSClient.list_object_array_metadata`][rddms_io.client.RDDMSClient.list_object_array_metadata]:
            A similar method that fetches the metadata from the objects
            themselves along with a dataspace uri. It is recommended to use
            `list_object_array_metadata` if you already have the objects in
            memory.
        """
        ml_objects = await self.download_models(
            ml_uris=ml_uris,
            download_arrays=False,
            download_linked_objects=False,
        )

        if not ml_objects:
            return {}

        ml_uris = [DataObjectURI.from_uri(uri) for uri in ml_uris]
        dataspace_uris = [DataspaceURI.from_any_etp_uri(uri) for uri in ml_uris]

        array_metadata = await asyncio.gather(
            *[
                self.list_object_array_metadata(
                    dataspace_uri=dataspace_uri,
                    ml_objects=[ml_object.obj],
                )
                for dataspace_uri, ml_object in zip(dataspace_uris, ml_objects)
            ]
        )

        metadata_map: dict[str, dict[str, DataArrayMetadata]] = {}
        for am in array_metadata:
            metadata_map = {**metadata_map, **am}

        return metadata_map

    async def list_object_array_metadata(
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
        [`RDDMSClient.list_array_metadata`][rddms_io.client.RDDMSClient.list_array_metadata]:
            A similar method that looks up array metadata needing only the uris
            of the objects.
        """
        dataspace_uri = str(DataspaceURI.from_any_etp_uri(dataspace_uri))

        ml_uris = []
        tasks = []
        for obj in ml_objects:
            obj_dais = []
            for hdf5_dataset in find_hdf5_datasets(obj):
                path_in_resource = hdf5_dataset.path_in_hdf_file
                epc_uri = hdf5_dataset.hdf_proxy.get_etp_data_object_uri(
                    dataspace_path_or_uri=dataspace_uri,
                )

                dai = DataArrayIdentifier(
                    uri=epc_uri,
                    path_in_resource=path_in_resource,
                )
                obj_dais.append(dai)

            if obj_dais:
                ml_uris.append(
                    obj.get_etp_data_object_uri(dataspace_path_or_uri=dataspace_uri)
                )
                task = self.etp_client.send_and_recv(
                    GetDataArrayMetadata(
                        data_arrays={dai.path_in_resource: dai for dai in obj_dais},
                    )
                )
                tasks.append(task)

        if not tasks:
            logger.info(
                "There were no arrays connected to input objects with uris: "
                f"{[obj.get_etp_data_object_uri(dataspace_uri) for obj in ml_objects]}"
            )
            return {}

        task_responses = await asyncio.gather(*tasks)

        metadata_map = {}
        for uri, tr in zip(ml_uris, task_responses):
            parse_and_raise_response_errors(
                tr,
                GetDataArrayMetadataResponse,
                "RDDMSClient.list_array_metadata",
            )

            pirm: dict[str, DataArrayMetadata] = {}
            for response in tr:
                pirm = {
                    **pirm,
                    **typing.cast(
                        GetDataArrayMetadataResponse, response
                    ).array_metadata,
                }

            metadata_map[uri] = pirm

        return metadata_map

    async def upload_array(
        self,
        epc_uri: str | DataObjectURI,
        path_in_resource: str,
        data: npt.NDArray[ETPNumpyArrayType],
    ) -> None:
        """
        Method used for uploading a single array to an ETP server. This method
        will not work without the user setting up a transaction for writing to
        the relevant dataspace. It should not be necessary for a user to call
        this method, prefer `RDDMSClient.upload_model` instead.

        Parameters
        ----------
        epc_uri
            An ETP data object uri to an `obj_EpcExternalPartReference` that is
            connected to the object that links to the provided array.
        path_in_resource
            A key (typically a HDF5-key) that uniquely identifies the array
            along with the `epc_uri`. This key is found in the object that
            links to the provided array.
        data
            A NumPy-array with the data.

        See Also
        --------
        [`RDDMSClient.upload_model`][rddms_io.client.RDDMSClient.upload_model]:
            A higher-level method that wraps transaction handling, data object
            uploading and array uploading in one go.
        """
        # Fetch ETP logical and transport array types
        logical_array_type, transport_array_type = (
            utils_arrays.get_logical_and_transport_array_types(data.dtype)
        )

        # Create identifier for the data.
        dai = DataArrayIdentifier(
            uri=str(epc_uri),
            path_in_resource=path_in_resource,
        )

        # Get current time as a UTC-timestamp.
        now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e6)

        # Allocate space on server for the array.
        responses = await self.etp_client.send_and_recv(
            PutUninitializedDataArrays(
                data_arrays={
                    dai.path_in_resource: PutUninitializedDataArrayType(
                        uid=dai,
                        metadata=DataArrayMetadata(
                            dimensions=list(data.shape),
                            transport_array_type=transport_array_type,
                            logical_array_type=logical_array_type,
                            store_last_write=now,
                            store_created=now,
                        ),
                    ),
                },
            ),
        )

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, PutUninitializedDataArraysResponse)
        assert len(response.success) == 1 and dai.path_in_resource in response.success

        # Check if we can upload the entire array in go, or if we need to
        # upload it in smaller blocks.

        # TODO: Check if we should use `await` instead of `asyncio.gather`.
        # This is to ensure that the websockets heartbeat gets to run once in a
        # while.
        if data.nbytes > self.etp_client.max_array_size:
            tasks = []

            # Get list with starting indices in each block, and a list with the
            # number of elements along each axis for each block.
            block_starts, block_counts = get_array_block_sizes(
                data.shape, data.dtype, self.etp_client.max_array_size
            )

            for starts, counts in zip(block_starts, block_counts):
                # Create slice-objects for each block.
                slices = tuple(
                    map(
                        lambda s, c: slice(s, s + c),
                        np.array(starts).astype(int),
                        np.array(counts).astype(int),
                    )
                )

                # Slice the array, and convert to the relevant ETP-array type.
                # Note in the particular the extra `.data`-after the call. The
                # data should not be of type `DataArray`, but `AnyArray`, so we
                # need to fetch it from the `DataArray`.
                etp_subarray_data = utils_arrays.get_etp_data_array_from_numpy(
                    data[slices]
                ).data

                # Create an asynchronous task to upload a block to the
                # ETP-server.
                task = self.etp_client.send_and_recv(
                    PutDataSubarrays(
                        data_subarrays={
                            dai.path_in_resource: PutDataSubarraysType(
                                uid=dai,
                                data=etp_subarray_data,
                                starts=starts,
                                counts=counts,
                            ),
                        },
                    ),
                )
                tasks.append(task)

            # Upload all blocks.
            task_responses = await asyncio.gather(*tasks)

            # Flatten list of responses.
            responses = [
                response
                for task_response in task_responses
                for response in task_response
            ]

            # Check for successful responses.
            for response in responses:
                assert isinstance(response, PutDataSubarraysResponse)
                assert (
                    len(response.success) == 1
                    and dai.path_in_resource in response.success
                )

            # Return after uploading all sub arrays.
            return

        # Convert NumPy data-array to an ETP-transport array.
        etp_array_data = utils_arrays.get_etp_data_array_from_numpy(data)

        # Pass entire array in one message.
        responses = await self.etp_client.send_and_recv(
            PutDataArrays(
                data_arrays={
                    dai.path_in_resource: PutDataArraysType(
                        uid=dai,
                        array=etp_array_data,
                    ),
                }
            )
        )

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, PutDataArraysResponse)
        assert len(response.success) == 1 and dai.path_in_resource in response.success

    async def download_object_arrays(
        self,
        dataspace_uri: str | DataspaceURI,
        ml_object: ro.AbstractCitedDataObject,
    ) -> dict[str, npt.NDArray[ETPNumpyArrayType]]:
        """
        Method accepting a `dataspace_uri` (or dataspace path) and a
        RESQML-object, and downloading all attached arrays (if any). This
        method is mainly used as a helper method for
        [`RDDMSClient.download_models`][rddms_io.client.RDDMSClient.download_models].

        Parameters
        ----------
        dataspace_uri
            An ETP dataspace uri or path. This can be a string or a
            [`DataspaceURI`][pyetp.uri.DataspaceURI]-object.
        ml_object
            An instance of a RESQML-object.

        Returns
        -------
        dict[str, npt.NDArray[ETPNumpyArrayType]]
            A dictionary mapping the `path_in_hdf_file`-keys in `ml_object` to
            the corresponding array. Empty if `ml_object` does not reference
            any arrays.

        See Also
        --------
        [`RDDMSClient.download_models`][rddms_io.client.RDDMSClient.download_models]:
            The "full" method downloading objects, arrays and potentially
            linked objects.
        """
        dataspace_uri = str(DataspaceURI.from_any_etp_uri(dataspace_uri))
        ml_hds = find_hdf5_datasets(ml_object)

        if len(ml_hds) == 0:
            logger.info(
                f"Object {type(ml_object).__name__}, titled "
                f"'{ml_object.citation.title}', does not reference any arrays."
            )

            return {}

        tasks = []
        for hdf5_dataset in ml_hds:
            path_in_resource = hdf5_dataset.path_in_hdf_file
            epc_uri = hdf5_dataset.hdf_proxy.get_etp_data_object_uri(
                dataspace_path_or_uri=dataspace_uri
            )
            task = self.download_array(
                epc_uri=epc_uri,
                path_in_resource=path_in_resource,
            )
            tasks.append(task)

        arrays = await asyncio.gather(*tasks)
        data_arrays = {hdf.path_in_hdf_file: arr for hdf, arr in zip(ml_hds, arrays)}

        return data_arrays

    async def download_array(
        self,
        epc_uri: str | DataObjectURI,
        path_in_resource: str,
    ) -> npt.NDArray[ETPNumpyArrayType]:
        """
        Method used for downloading a single array from an ETP server. It
        should not be necessary for a user to call this method, prefer
        `RDDMSClient.download_models` instead.

        Parameters
        ----------
        epc_uri: str | DataObjectURI
            An ETP data object uri to an `obj_EpcExternalPartReference` that is
            connected to the object that links to the provided array.
        path_in_resource: str
            A key (typically a HDF5-key) that uniquely identifies the array
            along with the `epc_uri`. This key is found in the object that
            links to the provided array.

        Returns
        -------
        data: npt.NDArray[ETPNumpyArrayType]
            A NumPy-array with the data.

        See Also
        --------
        [`RDDMSClient.download_models`][rddms_io.client.RDDMSClient.download_models]:
            A higher-level method that wraps, data object and array downloading
            in one go.
        """
        # Create identifier for the data.
        dai = DataArrayIdentifier(
            uri=str(epc_uri),
            path_in_resource=path_in_resource,
        )

        responses = await self.etp_client.send_and_recv(
            GetDataArrayMetadata(data_arrays={dai.path_in_resource: dai}),
        )

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, GetDataArrayMetadataResponse)
        assert (
            len(response.array_metadata) == 1
            and dai.path_in_resource in response.array_metadata
        )

        metadata = response.array_metadata[dai.path_in_resource]

        # Check if we can download the full array in a single message.
        if (
            utils_arrays.get_transport_array_size(
                metadata.transport_array_type, metadata.dimensions
            )
            >= self.etp_client.max_array_size
        ):
            transport_dtype = utils_arrays.get_dtype_from_any_array_type(
                metadata.transport_array_type,
            )
            # NOTE: The logical array type is not yet supported by the
            # open-etp-server. As such the transport array type will be actual
            # array type used. We only add this call to prepare for when it
            # will be used.
            logical_dtype = utils_arrays.get_dtype_from_any_logical_array_type(
                metadata.logical_array_type,
            )
            if logical_dtype != np.dtype(np.bool_):
                # If this debug message is triggered we should test the
                # mapping.
                logger.debug(
                    "Logical array type has changed: "
                    f"{metadata.logical_array_type = }, with {logical_dtype = }"
                )

            # Create a buffer for the data.
            data = np.zeros(metadata.dimensions, dtype=transport_dtype)

            # Get list with starting indices in each block, and a list with the
            # number of elements along each axis for each block.
            block_starts, block_counts = get_array_block_sizes(
                data.shape, data.dtype, self.etp_client.max_array_size
            )

            def data_subarrays_key(pir: str, i: int) -> str:
                return pir + f" ({i})"

            # TODO: Consider using `await` instead of `asyncio.gather` to give
            # the websockets connection time to run the heartbeat
            # communication.
            tasks = []
            for i, (starts, counts) in enumerate(zip(block_starts, block_counts)):
                task = self.etp_client.send_and_recv(
                    GetDataSubarrays(
                        data_subarrays={
                            data_subarrays_key(
                                dai.path_in_resource, i
                            ): GetDataSubarraysType(
                                uid=dai,
                                starts=starts,
                                counts=counts,
                            ),
                        },
                    ),
                )
                tasks.append(task)

            task_responses = await asyncio.gather(*tasks)
            responses = [
                response
                for task_response in task_responses
                for response in task_response
            ]

            data_blocks = []
            for i, response in enumerate(responses):
                assert isinstance(response, GetDataSubarraysResponse)
                assert (
                    len(response.data_subarrays) == 1
                    and data_subarrays_key(dai.path_in_resource, i)
                    in response.data_subarrays
                )

                data_block = response.data_subarrays[
                    data_subarrays_key(dai.path_in_resource, i)
                ].to_numpy_array()
                data_blocks.append(data_block)

            for data_block, starts, counts in zip(
                data_blocks, block_starts, block_counts
            ):
                # Create slice-objects for each block.
                slices = tuple(
                    map(
                        lambda s, c: slice(s, s + c),
                        np.array(starts).astype(int),
                        np.array(counts).astype(int),
                    )
                )
                data[slices] = data_block

            # Return after fetching all sub arrays.
            return data

        # Download the full array in one go.
        responses = await self.etp_client.send_and_recv(
            GetDataArrays(data_arrays={dai.path_in_resource: dai}),
        )

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, GetDataArraysResponse)
        assert (
            len(response.data_arrays) == 1
            and dai.path_in_resource in response.data_arrays
        )

        return response.data_arrays[dai.path_in_resource].to_numpy_array()

    async def upload_model(
        self,
        dataspace_uri: str | DataspaceURI,
        ml_objects: Sequence[ro.AbstractCitedDataObject],
        data_arrays: typing.Mapping[str, npt.NDArray[ETPNumpyArrayType]] = {},
        handle_transaction: bool = True,
        debounce: bool | float = False,
    ) -> list[str]:
        """
        The main driver method for uploading data to an ETP server. This method
        takes in a dataspace uri (for uploading to multiple dataspaces you need
        to call `RDDMSClient.upload_model` multiple times), a set of
        RESQML-objects, and a mapping of data arrays that are indexed by their
        path in resource (which is found in the RESQML-objects as well).

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
        handle_transaction
            A flag to toggle if `RDDMSClient.upload_model` should start and
            commit the transaction towards the dataspace. Default is `True`,
            and the method will ensure that the transaction handling is done
            correctly.
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
        [`RDDMSClient.download_models`][rddms_io.client.RDDMSClient.download_models]:
            The reverse operation.

        [`RDDMSClient.start_transaction`][rddms_io.client.RDDMSClient.start_transaction]:
            Method for setting up a transaction. It is only necessary to
            interact with this method if `handle_transaction=False`.

        [`RDDMSClient.commit_transaction`][rddms_io.client.RDDMSClient.commit_transaction]:
            Method for committing a transaction. It is only necessary to
            interact with this method if `handle_transaction=False`.
        """
        if not ml_objects:
            return []

        dataspace_uri = str(DataspaceURI.from_any_etp_uri(dataspace_uri))

        if handle_transaction:
            transaction_uuid = await self.start_transaction(
                dataspace_uri=dataspace_uri,
                read_only=False,
                debounce=debounce,
            )

        ml_uris = await self._upload_model(
            dataspace_uri,
            ml_objects=ml_objects,
            data_arrays=data_arrays,
        )

        if handle_transaction:
            await self.commit_transaction(transaction_uuid=transaction_uuid)

        return ml_uris

    async def _send_put_data_objects(
        self, pdo: PutDataObjects
    ) -> list[PutDataObjectsResponse]:
        if len(pdo.data_objects) > 1:
            raise NotImplementedError(
                "We currently only support chunking a single data object at a time. "
                "Consider splitting up the data into separate "
                "`PutDataObjects`-messages first."
            )

        try:
            return await self.etp_client.send_and_recv(pdo)
        except ETPMessageTooLarge:
            logger.debug("The `PutDataObjects`-message is too big, starting chunking.")

        # We subtract space for header and body metadata in the chunk messages.
        # This should ideally be tested with the full size, then adjusted to a
        # lower number if we get an `ETPMessageTooLarge`-error.
        chunk_size = self.etp_client.max_size - 1000
        assert chunk_size > 0
        blob_id = str(uuid.uuid4())

        dob_key = list(pdo.data_objects)[0]
        dob = pdo.data_objects[dob_key]
        data = dob.data

        new_dob = DataObject(
            format=dob.format,
            blob_id=blob_id,
            resource=dob.resource,
        )
        new_pdo = PutDataObjects(data_objects={dob_key: new_dob})
        chunked_bytes = tuple(batched(data, n=chunk_size))

        chunks = [
            Chunk(
                blob_id=blob_id,
                data=b"".join([i.to_bytes() for i in chunk]),
                final=i == (len(chunked_bytes) - 1),
            )
            for i, chunk in enumerate(chunked_bytes)
        ]
        assert chunks[-1].final

        return await self.etp_client.send_and_recv(
            body=new_pdo,
            multi_part_bodies=chunks,
        )

    async def _upload_model(
        self,
        dataspace_uri: str,
        ml_objects: Sequence[ro.AbstractCitedDataObject],
        data_arrays: typing.Mapping[str, npt.NDArray[ETPNumpyArrayType]],
    ) -> list[str]:
        logger.debug(
            f"Starting to upload model of {len(ml_objects)} objects and "
            f"{len(data_arrays)} arrays to dataspace '{dataspace_uri}'"
        )

        # A UTC timestamp in microseconds.
        now = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e6)

        ml_uris = []
        ml_hds = []
        tasks = []
        for obj in ml_objects:
            # Find all `Hdf5Dataset`-objects in the object.
            ml_hds.extend(find_hdf5_datasets(obj))

            uri = obj.get_etp_data_object_uri(dataspace_uri)
            ml_uris.append(uri)
            obj_xml = serialize_resqml_v201_object(obj)

            # Note that all objects passed in to this method is
            # expected to contain a `citation`-field.
            last_changed = obj.citation.last_update or obj.citation.creation
            if not isinstance(last_changed, datetime.datetime):
                last_changed = last_changed.to_datetime()

            # Note that chunking is handled in the ETP-client if that is needed.
            dob = DataObject(
                format="xml",
                data=obj_xml,
                resource=Resource(
                    uri=uri,
                    name=obj.citation.title,
                    last_changed=int(last_changed.timestamp() * 1e6),
                    store_created=now,
                    store_last_write=now,
                    # This is only used by WITSML channel data objects.
                    active_status="Inactive",
                    # Not used in the Store-protocol.
                    source_count=None,
                    target_count=None,
                ),
            )
            pdo = PutDataObjects(
                data_objects={f"{dob.resource.name} -- {dob.resource.uri}": dob},
            )

            tasks.append(self._send_put_data_objects(pdo))

        task_responses = await asyncio.gather(*tasks)
        responses = [
            response for task_response in task_responses for response in task_response
        ]

        parse_and_raise_response_errors(
            responses, PutDataObjectsResponse, "RDDMSClient.upload_model"
        )

        logger.debug("Done uploading model objects. Starting on upload of arrays.")

        dataspace_path = DataspaceURI.from_uri(dataspace_uri).dataspace

        uploaded_data_keys = []
        tasks = []
        for hdf5_dataset in ml_hds:
            path_in_resource = hdf5_dataset.path_in_hdf_file
            epc_uri = hdf5_dataset.hdf_proxy.get_etp_data_object_uri(
                dataspace_path_or_uri=dataspace_path
            )

            data = data_arrays[path_in_resource]
            uploaded_data_keys.append(path_in_resource)

            tasks.append(
                self.upload_array(
                    epc_uri=epc_uri,
                    path_in_resource=path_in_resource,
                    data=data,
                )
            )

        if len(data_arrays) > len(uploaded_data_keys):
            logger.warning(
                "Not all arrays were uploaded. The remaining array keys "
                f"{set(data_arrays) - set(uploaded_data_keys)} were not found in the "
                "provided objects."
            )

        await asyncio.gather(*tasks)

        logger.debug("Done uploading arrays.")
        return ml_uris

    async def download_models(
        self,
        ml_uris: list[str | DataObjectURI],
        download_arrays: bool = False,
        download_linked_objects: bool = False,
    ) -> list[RDDMSModel]:
        """
        Download RESQML-models from the RDDMS server.
        A model in this sense is a RESQML-object (with a given uri) and
        possibly with any connected arrays and referenced objects.

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
        """
        if len(ml_uris) == 0:
            raise ValueError("No uris in input 'ml_uris'")

        return await asyncio.gather(
            *[
                self._download_model(
                    ml_uri=ml_uri,
                    download_arrays=download_arrays,
                    download_linked_objects=download_linked_objects,
                )
                for ml_uri in ml_uris
            ]
        )

    async def _recv_get_data_objects(
        self, gdo: GetDataObjects
    ) -> list[GetDataObjectsResponse]:
        if len(gdo.uris) > 1:
            raise NotImplementedError(
                "We only support chunking a single data object at at time. "
                "Split up the data into separate `GetDataObjects`-messages first."
            )

        responses = await self.etp_client.send_and_recv(gdo)

        if len(responses) == 1:
            return responses

        logger.debug("The `GetDataObjects`-message is too big, starting chunking.")
        gdor = responses.pop(0)
        assert isinstance(gdor, GetDataObjectsResponse)
        assert len(gdor.data_objects) == 1
        dob_key = list(gdor.data_objects)[0]
        dob = gdor.data_objects[dob_key]
        assert len(dob.data) == 0
        blob_id = dob.blob_id

        # TODO: This is a bug from the server sending an extra, empty
        # `GetDataObjectsResponse`-message after all the `Chunk`-messages.
        # For now we catch and pop it out of the list. This test should be
        # removed once the bug is fixed.
        if isinstance(responses[-1], GetDataObjectsResponse):
            bug_gdor = responses.pop(-1)
            assert isinstance(bug_gdor, GetDataObjectsResponse)
            assert bug_gdor.data_objects == {}

        # The returned `Chunk`-messages are sorted based on the message id in
        # the header (see `ETPClient.__receiver_loop`).
        assert all([isinstance(r, Chunk) for r in responses])
        assert all([typing.cast(Chunk, r).blob_id == blob_id for r in responses])
        assert typing.cast(Chunk, responses[-1]).final

        gdor.data_objects[dob_key].blob_id = None
        gdor.data_objects[dob_key].data = b"".join(
            [typing.cast(Chunk, r).data for r in responses]
        )

        return [gdor]

    async def _download_model(
        self,
        ml_uri: str,
        download_arrays: bool,
        download_linked_objects: bool,
    ) -> RDDMSModel:
        dataspace_uri = str(DataspaceURI.from_any_etp_uri(ml_uri))

        responses = await self._recv_get_data_objects(
            GetDataObjects(uris={ml_uri: ml_uri})
        )

        parse_and_raise_response_errors(
            responses, GetDataObjectsResponse, "RDDMSClient.download_model"
        )

        # This should be true when chunking is used as well. The ETP-client
        # should construct a (too large) `GetDataObjectsResponse`-message from
        # the different chunks.
        assert len(responses) == 1
        response = responses[0]
        assert isinstance(response, GetDataObjectsResponse)

        assert len(response.data_objects) == 1
        data_object = response.data_objects[ml_uri]

        ml_object = parse_resqml_v201_object(data_object.data)
        # We should only get top-level objects in return from the ETP-server.
        assert isinstance(ml_object, ro.AbstractCitedDataObject)

        linked_models = []
        if download_linked_objects:
            dors = find_data_object_references(ml_object)
            additional_uris = [
                dor.get_etp_data_object_uri(dataspace_uri) for dor in dors
            ]

            # Remove any `EpcExternalPartReference`-objects from the extra
            # uris.
            additional_uris = list(
                filter(lambda a: "EpcExternalPartReference" not in a, additional_uris)
            )

            if additional_uris:
                logger.info(f"Downloading linked objects with uris: {additional_uris}")
                linked_models = await asyncio.gather(
                    *[
                        self._download_model(
                            ml_uri=au,
                            download_arrays=download_arrays,
                            # TODO: Allow downloading objects at a lower depth?
                            download_linked_objects=False,
                        )
                        for au in additional_uris
                    ]
                )

        if not download_arrays:
            return RDDMSModel(
                obj=ml_object,
                arrays={},
                linked_models=linked_models,
            )

        arrays = await self.download_object_arrays(
            dataspace_uri=dataspace_uri,
            ml_object=ml_object,
        )

        return RDDMSModel(
            obj=ml_object,
            arrays=arrays,
            linked_models=linked_models,
        )

    async def delete_model(
        self,
        ml_uris: list[str | DataObjectURI],
        prune_contained_objects: bool = False,
        handle_transaction: bool = True,
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
        handle_transaction
            A flag to toggle if `RDDMSClient.delete_model` should start and
            commit the transaction towards the dataspace. Default is `True`,
            and the method will ensure that the transaction handling is done
            correctly.
        debounce
            Parameter to decide if `RDDMSClient.delete_model` should retry
            starting a transaction if it initially fails. See
            `RDDMSClient.start_transaction` for a more in-depth explanation of
            the parameter. Default is `False`, i.e., no debouncing will occur
            and the method will fail if it is unable to start a transaction.
        """
        if not ml_uris:
            return

        uris = list(map(str, ml_uris))
        ddo = DeleteDataObjects(
            uris=dict(zip(uris, uris)),
            prune_contained_objects=prune_contained_objects,
        )

        if handle_transaction:
            dataspace_uris = [str(DataspaceURI.from_any_etp_uri(u)) for u in ml_uris]
            assert all([dataspace_uris[0] == du for du in dataspace_uris])
            transaction_uuid = await self.start_transaction(
                dataspace_uri=dataspace_uris[0], read_only=False, debounce=debounce
            )

        responses = await self.etp_client.send_and_recv(ddo)

        parse_and_raise_response_errors(
            responses,
            DeleteDataObjectsResponse,
            "RDDMSClient.delete_model",
        )

        if handle_transaction:
            await self.commit_transaction(transaction_uuid)


class rddms_connect:
    """
    Connect to an RDDMS server via ETP.

    This class can act as:

    1. A context manager handling setup and tear-down of the connection.
    2. An asynchronous iterator which can be used to persistently retry to
    connect if the websockets connection drops.
    3. An awaitable connection that must be manually closed by the user.

    See below for examples of all three cases.

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
    use_compression
        Flag to toggle if compression of the messages should be applied. So far
        the client (and the server) only supports compression with gzip.
        Default is `True` and compression is applied.


    Examples
    --------
    An example of connecting to an RDDMS server using
    [`rddms_connect`][rddms_io.client.rddms_connect] as a context manager is:

        async with rddms_connect(...) as rddms_client:
            ...

    In this case the closing message is sent and the websockets connection is
    closed once the program exits the context manager.


    To persist a connection if the websockets connection is dropped (for any
    reason), use :func:`rddms_connect` as an asynchronous generator, viz.:

        import websockets

        async for rddms_client in rddms_connect(...):
            try:
                ...
            except websockets.ConnectionClosed:
                continue

            # Include `break` to avoid re-running the whole block if the
            # iteration runs without any errors.
            break

    Note that in this case the whole program under the `try`-block is re-run
    from the start if the iteration completes normally, or if the websockets
    connection is dropped. Therefore, make sure to include a `break` at the end
    of the `try`-block (as in the example above).


    The third option is to set up a connection via `await` and then manually
    close the connection once done:

        rddms_client = await rddms_connect(...)
        ...
        await rddms_client.close()

    See Also
    --------
    [`pyetp.client.etp_connect`][pyetp.client.etp_connect]:
        The [`rddms_connect`][rddms_io.client.rddms_connect]-class is a thin
        wrapper around [`etp_connect`][pyetp.client.etp_connect].
    """

    def __init__(
        self,
        uri: str,
        data_partition_id: str | None = None,
        authorization: str | SecretStr | None = None,
        etp_timeout: float | None = None,
        max_message_size: int = 2**20,
        use_compression: bool = True,
    ) -> None:
        self.uri = uri
        self.data_partition_id = data_partition_id

        if isinstance(authorization, SecretStr):
            self.authorization = authorization
        else:
            self.authorization = SecretStr(authorization)

        self.etp_timeout = etp_timeout
        self.max_message_size = max_message_size
        self.use_compression = use_compression

    def __await__(self) -> RDDMSClient:
        return self.__aenter__().__await__()

    async def __aenter__(self) -> RDDMSClient:
        etp_client = await etp_connect(
            uri=self.uri,
            data_partition_id=self.data_partition_id,
            authorization=self.authorization,
            etp_timeout=self.etp_timeout,
            max_message_size=self.max_message_size,
            use_compression=self.use_compression,
        )
        self.rddms_client = RDDMSClient(etp_client)

        return self.rddms_client

    async def __aexit__(
        self,
        exc_type: typing.Type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return await self.rddms_client.close()

    async def __aiter__(self) -> AsyncGenerator[RDDMSClient]:
        async for etp_client in etp_connect(
            uri=self.uri,
            data_partition_id=self.data_partition_id,
            authorization=self.authorization,
            etp_timeout=self.etp_timeout,
            max_message_size=self.max_message_size,
        ):
            yield RDDMSClient(etp_client)
