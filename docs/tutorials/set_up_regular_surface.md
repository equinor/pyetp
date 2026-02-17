# Setting up a regular surface

In this tutorial we will show to set up a regular surface in RESQML v2.0.1
using `resqml_objects` which is installed alongside `pyetp`.

???+ Note
    Due to the flexibility of RESQML there are multiple ways to represent a
    regular surface.
    Different providers make different choices, and there is no guarantee that
    their software will understand all the possible ways of representing a
    surface.
    If compatibility towards a software platform is important, make sure that
    you write your surfaces in such a way that it can be read by that platform.

To represent a regular surface we utilize three objects from RESQML v2.0.1.
They are:

1. An
    [`obj_EpcExternalPartReference`][resqml_objects.v201.generated.obj_EpcExternalPartReference]
    which is a RESQML v2.0.1 specific object to explain (loosely speaking) that
    the raw array data is stored alongside the objects.
    This object is needed when uploading data to an RDDMS server.
2. A local coordinate system, either
    [`obj_LocalDepth3dCrs`][resqml_objects.v201.generated.obj_LocalDepth3dCrs]
    if the surface describes depth in units of distance
    or [`obj_LocalTime3dCrs`][resqml_objects.v201.generated.obj_LocalTime3dCrs]
    if it is in units of time.
3. An
    [`obj_Grid2dRepresentation`][resqml_objects.v201.generated.obj_Grid2dRepresentation]
    for the actual grid metadata and references to 1. and 2. above.
    We will often denote the `obj_Grid2dRepresentation` as _"the grid-object"_.

There are two important considerations when setting up these objects.
The first is that multiple `obj_Grid2dRepresentation` can reference the same
coordinate system and `obj_EpcExternalPartReference`.
Secondly, RESQML allows both [_active_ and _passive_
transformations](https://en.wikipedia.org/wiki/Active_and_passive_transformation)
of the surface data.
That is, the coordinates of the surface is described by both the coordinate
system and the grid-object.
The coordinate system describes passive transformations and the grid-object
active transformations.


We will demonstrate a few variations for setting up these objects below.


## Single, stand-alone surface in an unrotated coordinate system
The first example we show is for a regular surface that is not connected to any
other RESQML-objects, and where we keep any potential rotation in the
`obj_Grid2dRepresentation`-object and leave the coordinate system unrotated and
aligned with the global coordinate system (which in this case is an EPSG-code).

This example demonstrates the minimally required information needed to
represent a regular surface in RESQML v2.0.1.

???+ Tip
    The constructed objects are normal Python dataclasses, and can be printed
    to console for a good overview of the content and default values.
    Use [`rich`](https://github.com/Textualize/rich) for pretty-printing.
    It is included as a sub-dependency in `pyetp`.

In this example we will use a fixed creation datetime, and fixed uuids for the
three objects.
These parameters are optional.
If not specified the `creation`-field in the
[`Citation`][resqml_objects.v201.generated.Citation]-object will be set to
`#!python datetime.datetime.now(datetime.timezone.utc)`, and the uuids will be
set to a random `#!python str(uuid.uuid4())`-value.
```python
--8<--
examples/tutorials/set_up_regular_surface/set_up_regular_surface.py::14
--8<--
```
The first object we set up is the
[`obj_EpcExternalPartReference`][resqml_objects.v201.generated.obj_EpcExternalPartReference]-object.
```python
--8<--
examples/tutorials/set_up_regular_surface/set_up_regular_surface.py:16:23
--8<--
```
We can then print the object to get the ouput below:
```
--8<--
examples/tutorials/set_up_regular_surface/epc_obj.txt
--8<--
```
and serializing the object into XML (using
[`serialize_resqml_v201_object`][resqml_objects.serializers.serialize_resqml_v201_object])
we get the corresponding output:
```
--8<--
examples/tutorials/set_up_regular_surface/epc_xml.txt
--8<--
```

### Setting up a coordinate system
Here we set up a "default"
[`obj_LocalDepth3dCrs`][resqml_objects.v201.generated.obj_LocalDepth3dCrs].
```python
--8<--
examples/tutorials/set_up_regular_surface/set_up_regular_surface.py:25:34
--8<--
```
In this example we have chosen a nondescript `"Mean sea level"` for the
vertical coordinate system, and the EPSG code 23031 covering much of Europe and
the North Sea.

???+ "Choosing a global coordinate system"

    A global coordinate system is described by the two fields `vertical_crs`
    and `projected_crs`, and the local coordinate system describes additional
    transformations on top of this global system.
    There are only three built-in choices (for RESQML v2.0.1) for the field
    `vertical_crs` and `projected_crs`.
    These are:

    1. [`VerticalCrsEpsgCode`][resqml_objects.v201.generated.VerticalCrsEpsgCode]
        and
        [`ProjectedCrsEpsgCode`][resqml_objects.v201.generated.ProjectedCrsEpsgCode]
        where the global coordinate system is described via an [EPSG
        code](https://en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset).
    2. [`GmlVerticalCrsDefinition`][resqml_objects.v201.generated.GmlVerticalCrsDefinition]
        and
        [`GmlProjectedCrsDefinition`][resqml_objects.v201.generated.GmlProjectedCrsDefinition]
        with the global coordinate system described by the [Geography Markup
        Language](https://en.wikipedia.org/wiki/Geography_Markup_Language) (GML).
    3. [`VerticalUnknownCrs`][resqml_objects.v201.generated.VerticalUnknownCrs]
        and
        [`ProjectedUnknownCrs`][resqml_objects.v201.generated.ProjectedUnknownCrs],
        which are used when the global coordinate system is irrelevant or anonymized.

    If the global coordinate system is not an EPSG code or in GML, we use the
    unknown-option and add a custom coordinate system description, e.g., a
    [well-known
    text](https://en.wikipedia.org/wiki/Well-known_text_representation_of_coordinate_reference_systems),
    in the `custom_data`-field of the
    [`obj_LocalDepth3dCrs`][resqml_objects.v201.generated.obj_LocalDepth3dCrs] or
    [`obj_LocalTime3dCrs`][resqml_objects.v201.generated.obj_LocalTime3dCrs].

Printing the `crs`-object we get:
```
--8<--
examples/tutorials/set_up_regular_surface/crs_obj.txt
--8<--
```
and the corresponding serialized XML:
```
--8<--
examples/tutorials/set_up_regular_surface/crs_xml.txt
--8<--
```
We note that the `obj_LocalDepth3dCrs`-object by default has zero offset
(`#!python xoffset == yoffset == zoffset == 0.0`) and zero rotation (`#!python
areal_rotation = ro.PlaneAngleMeasure(value=0.0, uom=ro.PlaneAngleUom.RAD)`).
The units for the vertical axis and the projected axes is set to meters
(`#!python vertical_uom=ro.LengthUom.M` and `#!python
projected_uom=ro.LengthUom.M`, respectively), and the surface is expected to
have the z-axis pointing downwards (`#!python zincreasing_downward=True`).


### Setting up the grid object
A regular surface is a two-dimensional height map with coordinates that have
uniform spacing along each coordinate direction.
The height map will be given as a dense array whereas the coordinates can be
compactly represented as an _origin_, a _shape_, a _spacing_, and a _rotation
angle_ or a pair of _orthonormal vectors_.
```python
--8<--
examples/tutorials/set_up_regular_surface/set_up_regular_surface.py:25:34
--8<--
```

```
--8<--
examples/tutorials/set_up_regular_surface/gri_obj.txt
--8<--
```


```
--8<--
examples/tutorials/set_up_regular_surface/gri_xml.txt
--8<--
```


### Full script
```python
--8<--
examples/tutorials/set_up_regular_surface/set_up_regular_surface.py
--8<--
```
