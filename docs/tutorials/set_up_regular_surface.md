# Setting up a regular surface

In this tutorial we will show how to represent a regular surface in RESQML
v2.0.1 using `resqml_objects` which is installed alongside `pyetp`.

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

???+ Nomenclature

    We call a collection of linked RESQML objects a _model_.
    For example, a regular surface consisting of an
    `obj_EpcExternalPartReference`, an `obj_LocalDepth3dCrs` and an
    `obj_Grid2dRepresentation` constitutes a model.


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

???+ "Printing the `obj_EpcExternalPartReference`-object"

    We can use `#!python print(epc)` directly, or `#!python rich.print(epc)`
    from the [`rich`](https://github.com/Textualize/rich) library to get an
    impression on the different fields of the object.
    ```
    --8<--
    examples/tutorials/set_up_regular_surface/epc_obj.txt
    --8<--
    ```
    To see the corresponding XML representation we use
    [`serialize_resqml_v201_object`][resqml_objects.serializers.serialize_resqml_v201_object].
    ```
    --8<--
    examples/tutorials/set_up_regular_surface/epc_xml.txt
    --8<--
    ```
    See the documentation for
    [`Citation`][resqml_objects.v201.generated.Citation] and
    [`obj_EpcExternalPartReference`][resqml_objects.v201.generated.obj_EpcExternalPartReference]
    (and for its superclasses) for an explanation of the various fields.

### Setting up a coordinate system
Here we set up a default
[`obj_LocalDepth3dCrs`][resqml_objects.v201.generated.obj_LocalDepth3dCrs].
That is, we leave the local coordinate system untransformed relative to the
global coordinate system, we choose our first axis to describe _eastings_ and
our second axis _northings_, and let the $z$-axis point downwards.
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
    See
    [here](https://community.opengroup.org/osdu/platform/domain-data-mgmt-services/reservoir/home/-/blob/main/Resources/CrsHandler.md)
    for a wider discussion of coordinate systems in RESQML and their relation
    to OSDU coordinate systems.

???+ "Printing the `obj_LocalDepth3dCrs`-object"

    Printing the `crs`-object with `#!python rich.print(crs)` we get:
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
    have the $z$-axis pointing downwards (`#!python
    zincreasing_downward=True`).


### Setting up the grid object
A regular surface is a gridded two-dimensional height map with coordinates that
have uniform spacing along each plane axis.
The height map will be given as a dense array whereas the coordinates can be
compactly represented as an _origin_, a _shape_, a _spacing_, and a _rotation
angle_ or a pair of _orthonormal vectors_.

???+ "Arrays in RESQML"

    In RESQML the array data is stored alongside the objects, and the objects
    keep a reference to the arrays.
    The reference to the array is a combination of the uri of an
    [`obj_EpcExternalPartReference`][resqml_objects.v201.generated.obj_EpcExternalPartReference]
    and a key called `path_in_hdf_file` (typically in RESQML) or `path_in_resource`
    (in ETP).
    The `obj_EpcExternalPartReference` acts as a proxy to an external storage
    location, often a
    [HDF5-file](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) (this
    applies only when the data is stored to disk), and the `path_in_hdf_file` is a
    key into that HDF5-file.
    If multiple arrays are stored in the same HDF5-file, then the objects that
    own these arrays must point to the same `obj_EpcExternalPartReference` and use
    unique `path_in_hdf_file`-keys.


In this example we use the following values for the coordinates:
```python
--8<--
examples/tutorials/set_up_regular_surface/set_up_regular_surface.py:36:40
--8<--
```
This describes a regular surface with $101 \times 103$ elements, that has its
first axis rotated by an angle $\pi / 6$ counter-clockwise to the first axis
described by the `crs`-object (the `#!python u1` unit vector), and the second
axis rotated by an additional angle $\pi / 2$ (the `#!python u2` unit vector).
The origin of the surface is placed in $(10, 11)$ relative to the offset in the
`crs`-object and in the same units as listed in the `crs`-object.
Finally, the `#!python spacing`-array gives the spacing between points for the
first axis and the second axis.

???+ "Mapping out the coordinates"

    The coordinates of the regular surface, $\mathbf{r}_{ij}$, are given by
    $$
        \mathbf{r}_{ij} = \mathbf{r}_0 + i \delta_1 \mathbf{u}_1
            + j \delta_2 \mathbf{u}_2,
    $$
    where $\mathbf{r}_0$ corresponds to the `origin`, $\delta_1$ and $\delta_2$
    the first and second component of the `spacing`-array, $\mathbf{u}_1$ and
    $\mathbf{u}_2$ the two unit vectors, and $i$ and $j$ are integers limited
    to the `shape` of the surface array.
    In total this will give coordinates $\mathbf{r}_{ij}$ represented in the
    given local coordinate system.


???+ Warning

    Rotation and translation is stored two places for a regular surface in
    RESQML v2.0.1.
    Rotation is stored as an _angle_ in the
    [`obj_LocalDepth3dCrs`][resqml_objects.v201.generated.obj_LocalDepth3dCrs] or
    [`obj_LocalTime3dCrs`][resqml_objects.v201.generated.obj_LocalTime3dCrs], and
    it is stored as a _pair of unit vectors_ for the coordinates in the
    [`obj_Grid2dRepresentation`][resqml_objects.v201.generated.obj_Grid2dRepresentation].
    Translation is stored as `xoffset`, `yoffset`, and `zoffset` in the
    `obj_LocalDepth3dCrs` and `obj_LocalTime3dCrs` and as `origin` in the
    `obj_Grid2dRepresentation`.
    As seen in the output of the the
    [`crs`-object](#setting-up-a-coordinate-system) the default local
    coordinate system has zero rotation and zero offset relative to the global
    coordinate system.


We use the classmethod
[`obj_Grid2dRepresentation.from_regular_surface`][resqml_objects.v201.generated.obj_Grid2dRepresentation.from_regular_surface]
to set up the RESQML-object.
This method is opinionated in choosing a specific set of RESQML array types.
```python
--8<--
examples/tutorials/set_up_regular_surface/set_up_regular_surface.py:42:56
--8<--
```

???+ "Printing the `obj_Grid2dRepresentation`-object"

    Printing the `gri`-object with `#!python rich.print(gri)` we get:
    ```
    --8<--
    examples/tutorials/set_up_regular_surface/gri_obj.txt
    --8<--
    ```
    and the serialized XML:
    ```
    --8<--
    examples/tutorials/set_up_regular_surface/gri_xml.txt
    --8<--
    ```
    Other representations of a surface using `obj_Grid2dRepresentation` can
    differ in their choice of arrays types, starting from the field
    `obj_Grid2dRepresentation.grid2d_patch.geometry.points` (called `Points` in
    the XML output).
    In our implementation of `resqml_objects` we see that the XML attribute
    `xsi:type` is _only_ included on the top-level element, and any element that
    supports multiple types.


#### Retrieving the coordinates
We have included a method,
[`obj_Grid2dRepresentation.get_xy_grid`][resqml_objects.v201.generated.obj_Grid2dRepresentation.get_xy_grid],
that simplifies the process of fleshing out the coordinate arrays `X` and `Y`
from the representation of the `obj_Grid2dRepresentation` described above, and
a given local coordinate system, viz.,
```python
--8<--
examples/tutorials/set_up_regular_surface/set_up_regular_surface.py:58:58
--8<--
```
This method is limited to the specific set of array types shown in the
`obj_Grid2dRepresentation`-above.


### Full script
We include the full script for the sake of completeness.
```python
--8<--
examples/tutorials/set_up_regular_surface/set_up_regular_surface.py
--8<--
```
