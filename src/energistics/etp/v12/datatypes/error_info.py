import enum
import typing

import energistics.base


class ErrorCode(enum.IntEnum):
    ENOROLE = 1
    ENOSUPPORTEDPROTOCOLS = 2
    EINVALID_MESSAGETYPE = 3
    EUNSUPPORTED_PROTOCOL = 4
    EINVALID_ARGUMENT = 5
    EREQUEST_DENIED = 6
    ENOTSUPPORTED = 7
    EINVALID_STATE = 8
    EINVALID_URI = 9
    EAUTHORIZATION_EXPIRED = 10
    ENOT_FOUND = 11
    ELIMIT_EXCEEDED = 12
    ECOMPRESSION_NOTSUPPORTED = 13
    EINVALID_OBJECT = 14
    EMAX_TRANSACTIONS_EXCEEDED = 15
    EDATAOBJECTTYPE_NOTSUPPORTED = 16
    EMAXSIZE_EXCEEDED = 17
    EMULTIPART_CANCELLED = 18
    EINVALID_MESSAGE = 19
    EINVALID_INDEXKIND = 20
    ENOSUPPORTEDFORMATS = 21
    EREQUESTUUID_REJECTED = 22
    EUPDATEGROWINGOBJECT_DENIED = 23
    EBACKPRESSURE_LIMIT_EXCEEDED = 24
    EBACKPRESSURE_WARNING = 25
    ETIMED_OUT = 26
    EAUTHORIZATION_REQUIRED = 27
    EAUTHORIZATION_EXPIRING = 28
    ENOSUPPORTEDDATAOBJECTTYPES = 29
    ERESPONSECOUNT_EXCEEDED = 30
    EINVALID_APPEND = 31
    EINVALID_OPERATION = 32
    EINVALID_CHANNELID = 1002
    ENOCASCADE_DELETE = 4003
    EPLURAL_OBJECT = 4004
    ERETENTION_PERIOD_EXCEEDED = 5001
    ENOTGROWINGOBJECT = 6001


@energistics.base.add_avro_metadata
class ErrorInfo(energistics.base.ETPBaseModel):
    _avro_schema: typing.ClassVar[energistics.base.AvroSchemaType] = {
        "type": "record",
        "namespace": "Energistics.Etp.v12.Datatypes",
        "name": "ErrorInfo",
        "fields": [
            {"name": "message", "type": "string"},
            {"name": "code", "type": "int"},
        ],
        "fullName": "Energistics.Etp.v12.Datatypes.ErrorInfo",
        "depends": [],
    }

    message: str
    code: ErrorCode
