import asyncio
import datetime
import logging
import typing
import uuid
from collections.abc import AsyncGenerator, Sequence
from types import TracebackType

import numpy.typing as npt
from pydantic import SecretStr

import resqml_objects.v201 as ro
from energistics.etp.v12.datatypes import ArrayOfString, DataValue, Uuid
from energistics.etp.v12.datatypes.object import (
    ContextInfo,
    DataObject,
    Dataspace,
    Resource,
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
    GetResourcesResponse,
)
from energistics.etp.v12.protocol.store import (
    DeleteDataObjects,
    DeleteDataObjectsResponse,
    GetDataObjects,
    GetDataObjectsResponse,
    PutDataObjects,
    PutDataObjectsResponse,
)
from energistics.etp.v12.protocol.transaction import (
    CommitTransaction,
    RollbackTransaction,
    RollbackTransactionResponse,
    StartTransaction,
    StartTransactionResponse,
)
from pyetp import utils_arrays
from pyetp.client import ETPClient, etp_connect, timeout_intervals
from pyetp.errors import (
    ETPTransactionFailure,
    parse_and_raise_response_errors,
)
from pyetp.uri import DataObjectURI, DataspaceURI
from rddms_io.data_types import RDDMSDataspace
from resqml_objects import parse_resqml_v201_object, serialize_resqml_v201_object
from resqml_objects.v201.utils import find_hdf5_datasets, get_qualified_type

logger = logging.getLogger(__name__)


class RDDMSClient:
    def __init__(self, etp_client: ETPClient) -> None:
        self.etp_client = etp_client

    async def close(self) -> None:
        await self.etp_client.close()

    async def list_dataspaces(
        self, store_last_write_filter: int | None = None
    ) -> list[RDDMSDataspace]:
        response = await self.etp_client.send(
            GetDataspaces(store_last_write_filter=store_last_write_filter)
        )
        parse_and_raise_response_errors(
            [response],
            GetDataspacesResponse,
            "RDDMSClient.list_dataspaces",
        )
        dataspaces = [
            RDDMSDataspace.from_etp_dataspace(ds) for ds in response.dataspaces
        ]
        return dataspaces

    async def delete_dataspace(self, dataspace_uri: DataspaceURI | str) -> None:
        dataspace_uri = str(DataspaceURI.from_any(dataspace_uri))
        response = await self.etp_client.send(
            DeleteDataspaces(uris={dataspace_uri: dataspace_uri})
        )
        parse_and_raise_response_errors(
            [response],
            DeleteDataspacesResponse,
            "RDDMSClient.delete_dataspace",
        )
        assert dataspace_uri in response.success

    async def create_dataspace(
        self,
        dataspace_uri: str | DataspaceURI,
        legal_tags: list[str] = [],
        other_relevant_data_countries: list[str] = [],
        owners: list[str] = [],
        viewers: list[str] = [],
    ) -> None:
        dataspace_uri = DataspaceURI.from_any(dataspace_uri)

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

        response = await self.etp_client.send(
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

        parse_and_raise_response_errors(
            [response], PutDataspacesResponse, "RDDMSClient.create_dataspace"
        )
        assert str(dataspace_uri) in response.success

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

        dataspace_uri = str(DataspaceURI.from_any(dataspace_uri))
        total_timeout = debounce

        if isinstance(debounce, bool):
            total_timeout = None

        for ti in timeout_intervals(total_timeout):
            response = await self.etp_client.send(
                StartTransaction(read_only=read_only, dataspace_uris=[dataspace_uri]),
            )
            parse_and_raise_response_errors(
                [response], StartTransactionResponse, "RDDMSClient.start_transaction"
            )

            if response.successful:
                break

            if debounce:
                logger.info(
                    f"Failed to start transaction with response: {response}, retrying "
                    f"in {ti} seconds"
                )
                await asyncio.sleep(ti)
                continue

            raise ETPTransactionFailure(str(response))

        transaction_uuid = uuid.UUID(str(response.transaction_uuid))

        logger.debug("Started transaction with uuid: {transaction_uuid}")

        return transaction_uuid

    async def commit_transaction(
        self,
        transaction_uuid: bytes | str | uuid.UUID | Uuid,
    ) -> None:
        if isinstance(transaction_uuid, uuid.UUID):
            transaction_uuid = Uuid(transaction_uuid.bytes)
        elif isinstance(transaction_uuid, str | bytes):
            transaction_uuid = Uuid(transaction_uuid)

        response = await self.etp_client.send(
            CommitTransaction(transaction_uuid=transaction_uuid),
        )

        if not response.successful:
            raise ETPTransactionFailure(str(response))

        logger.debug("Commited transaction with uuid: {transaction_uuid}")

    async def rollback_transaction(
        self, transaction_uuid: bytes | str | uuid.UUID | Uuid
    ) -> None:
        if isinstance(transaction_uuid, uuid.UUID | str):
            transaction_uuid = Uuid(str(transaction_uuid).encode())
        elif isinstance(transaction_uuid, bytes):
            transaction_uuid = Uuid(transaction_uuid)

        response = await self.etp_client.send(
            RollbackTransaction(transaction_uuid=transaction_uuid)
        )
        parse_and_raise_response_errors(
            [response], RollbackTransactionResponse, "RDDMSClient.rollback_transaction"
        )
        if not response.successful:
            raise ETPTransactionFailure(str(response))

    async def list_objects_under_dataspace(
        self,
        dataspace_uri: DataspaceURI | str,
        data_object_types: list[str | typing.Type[ro.AbstractCitedDataObject]] = [],
        count_objects: bool = True,
        store_last_write_filter: int | None = None,
    ) -> None:
        """
        This method will list all objects under a given dataspace.

        Parameters
        ----------
        dataspace_uri: DataspaceURI | str
            The uri of the dataspace to list objects.
        data_object_types: list[str | typing.Type[ro.AbstractCitedDataObject]]
            Object types to look for. This can either be a list of strings,
            e.g., `["eml20.*", "resqml20.obj_Grid2dRepresentation"]` to query
            all Energistic Common version 2.0-objects and
            `obj_Grid2dRepresentation`-objects from RESQML v2.0.1, or it can be
            a list of objects from `resqml_objects.v201`, e.g.,
            `[ro.obj_Grid2dRepresentation, ro.obj_LocalDepth3dCrs]`. Default is
            `[]`, i.e., an empty list which means that all data object types
            will be returned.
        count_objects: bool
        """
        dataspace_uri = str(DataspaceURI.from_any(dataspace_uri))
        data_object_types = [
            dot if isinstance(dot, str) else get_qualified_type(dot)
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

        response = await self.etp_client.send(gr)
        parse_and_raise_response_errors(
            [response], GetResourcesResponse, "RDDMSClient.list_objects_under_dataspace"
        )
        print(response)

    async def list_linked_objects(
        self,
        start_uri: DataObjectURI | str,
    ) -> None:
        """
        This method lists all objects that are linked to the provided object
        uri. That is, starting from the object with the given uri it finds all
        objects that links to it, and all objects it links to.
        """
        pass

    async def list_array_metadata(
        self,
        dataspace_uri: str | DataspaceURI,
        ml_objects: Sequence[ro.AbstractCitedDataObject],
    ) -> None:
        pass

    async def upload_array(
        self,
        epc_uri: str | DataObjectURI,
        path_in_resource: str,
        data: npt.NDArray[utils_arrays.LogicalArrayDTypes],
    ) -> None:
        await self.etp_client.upload_array(
            epc_uri=epc_uri,
            path_in_resource=path_in_resource,
            data=data,
        )

    async def download_array(
        self,
        epc_uri: str | DataObjectURI,
        path_in_resource: str,
    ) -> npt.NDArray[utils_arrays.LogicalArrayDTypes]:
        return await self.etp_client.download_array(
            epc_uri=epc_uri,
            path_in_resource=path_in_resource,
        )

    async def upload_model(
        self,
        dataspace_uri: str | DataspaceURI,
        ml_objects: Sequence[ro.AbstractCitedDataObject],
        data_arrays: typing.Mapping[
            str, Sequence[npt.NDArray[utils_arrays.LogicalArrayDTypes]]
        ] = {},
        handle_transaction: bool = True,
        debounce: bool | float = False,
    ) -> list[str]:
        dataspace_uri = str(DataspaceURI.from_any(dataspace_uri))

        if not handle_transaction:
            return await self._upload_model(
                dataspace_uri=dataspace_uri,
                ml_objects=ml_objects,
                data_arrays=data_arrays,
            )

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

        await self.commit_transaction(transaction_uuid=transaction_uuid)

        return ml_uris

    async def _upload_model(
        self,
        dataspace_uri: str,
        ml_objects: Sequence[ro.AbstractCitedDataObject],
        data_arrays: typing.Mapping[
            str, Sequence[npt.NDArray[utils_arrays.LogicalArrayDTypes]]
        ],
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

            uri = str(DataObjectURI.from_obj(dataspace_uri, obj))
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

            tasks.append(self.etp_client.send(pdo))

        responses = await asyncio.gather(*tasks)

        parse_and_raise_response_errors(
            responses, PutDataObjectsResponse, "RDDMSClient.upload_model"
        )

        logger.debug("Done uploading model objects. Starting on upload of arrays.")

        dataspace_path = DataspaceURI(dataspace_uri).dataspace

        tasks = []
        for hdf5_dataset in ml_hds:
            path_in_resource = hdf5_dataset.path_in_hdf_file
            epc_uri = hdf5_dataset.hdf_proxy.get_etp_data_object_uri(
                dataspace_path_or_uri=dataspace_path
            )
            data = data_arrays.pop(path_in_resource)
            tasks.append(
                self.upload_array(
                    epc_uri=epc_uri,
                    path_in_resource=path_in_resource,
                    data=data,
                )
            )

        if len(data_arrays) > 0:
            logger.warning(
                "Not all arrays were uploaded. The remaining array keys "
                f"{sorted(data_arrays)} were not found in the provided objects."
            )

        await asyncio.gather(*tasks)

        logger.debug("Done uploading arrays.")
        return ml_uris

    async def download_model(
        self,
        ml_uris: list[str | DataObjectURI],
        download_arrays: bool = False,
    ) -> tuple[
        list[ro.AbstractCitedDataObject],
        dict[str, list[npt.NDArray[utils_arrays.LogicalArrayDTypes]]],
    ]:
        tasks = []
        for uri in ml_uris:
            task = self.etp_client.send(GetDataObjects(uris={str(uri): str(uri)}))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        parse_and_raise_response_errors(
            responses, GetDataObjectsResponse, "RDDMSClient.download_model"
        )

        data_objects = [r.data_objects[str(u)] for u, r in zip(ml_uris, responses)]
        ml_objects = [parse_resqml_v201_object(d.data) for d in data_objects]

        if not download_arrays:
            return ml_objects, {}

        dataspace_path = DataspaceURI.from_any(ml_uris[0]).dataspace
        ml_hds = [ml_hd for obj in ml_objects for ml_hd in find_hdf5_datasets(obj)]

        tasks = []
        for hdf5_dataset in ml_hds:
            path_in_resource = hdf5_dataset.path_in_hdf_file
            epc_uri = hdf5_dataset.hdf_proxy.get_etp_data_object_uri(
                dataspace_path_or_uri=dataspace_path
            )
            task = self.download_array(
                epc_uri=epc_uri,
                path_in_resource=path_in_resource,
            )
            tasks.append(task)

        arrays = await asyncio.gather(*tasks)
        data_arrays = {hdf.path_in_hdf_file: arr for hdf, arr in zip(ml_hds, arrays)}

        return ml_objects, data_arrays

    async def delete_model(
        self,
        ml_uris: list[str | DataObjectURI],
        prune_contained_objects: bool = False,
        handle_transaction: bool = True,
        debounce: bool | float = False,
    ) -> None:
        uris = list(map(str, ml_uris))
        ddo = DeleteDataObjects(
            uris=dict(zip(uris, uris)),
            prune_contained_objects=prune_contained_objects,
        )

        if handle_transaction:
            dataspace_uris = [str(DataspaceURI.from_any(u)) for u in ml_uris]
            assert all([dataspace_uris[0] == du for du in dataspace_uris])
            transaction_uuid = await self.start_transaction(
                dataspace_uri=dataspace_uris[0], read_only=False, debounce=debounce
            )

        response = await self.etp_client.send(ddo)

        parse_and_raise_response_errors(
            [response],
            DeleteDataObjectsResponse,
            "RDDMSClient.delete_model",
        )

        if handle_transaction:
            await self.commit_transaction(transaction_uuid)


class rddms_connect:
    def __init__(
        self,
        uri: str,
        data_partition_id: str | None = None,
        authorization: str | SecretStr | None = None,
        etp_timeout: float | None = None,
        max_message_size: float = 2**20,
    ) -> None:
        self.uri = uri
        self.data_partition_id = data_partition_id
        self.authorization = SecretStr(authorization)
        self.etp_timeout = etp_timeout
        self.max_message_size = max_message_size

    def __await__(self) -> RDDMSClient:
        return self.__aenter__().__await__()

    async def __aenter__(self) -> RDDMSClient:
        etp_client = await etp_connect(
            uri=self.uri,
            data_partition_id=self.data_partition_id,
            authorization=self.authorization,
            etp_timeout=self.etp_timeout,
            max_message_size=self.max_message_size,
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
