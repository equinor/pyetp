# Serializers

The RESQML standard is formulated as XML objects.
We have used [`xsdata`](https://xsdata.readthedocs.io/en/latest/) to generate
Python dataclasses to avoid working directly with XML.
When passing the data to RDDMS, or other RESQML readers, we have to serialize
the RESQML dataclasses into XML.
This is done via the functions listed below.


## RESQML v2.0.1 serialization

::: resqml_objects.serializers.serialize_resqml_v201_object
