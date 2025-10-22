import logging
import pathlib
import sys
import zipfile

import h5py
import numpy as np
import numpy.typing as npt

import resqml_objects.v201 as ro_201

from .parsers import parse_resqml_v201_object


def get_resqml_v201_objects(
    epc_filename: str | pathlib.Path, log_failed_objects: bool = False
) -> list[ro_201.AbstractObject]:
    """This function is fairly brute-force; the function reads all files in the
    provided EPC-file, and tries to parse it as one of the auto-generated
    RESQML v2.0.1 objects. If you are unsure that all objects were included you
    should set `log_failed_objects=True` and see if there are any missed
    RESQML-objects. Objects such as `[Content_Types].xml`, `docProps/core.xml`,
    and anything under the `_rels/`-directory are not RESQML-objects, and will
    be logged as failed.


    Parameters
    ----------
    epc_filename: str | pathlib.Path
        The name of the EPC-file. Typically with an `.epc`-extension.
    log_failed_objects: bool
        Set to `True` in order to log the objects that were not parsed as
        RESQML v2.0.1 objects along with their errors. Default is `False`.


    Returns
    -------
    list[ro_201.AbstractObject]
        A list of the parsed RESQML v2.0.1 objects.


    See Also
    --------
    get_arrays_and_paths_in_hdf_file
    """

    if sys.version_info.major == 3 and sys.version_info.minor == 10:
        raise NotImplementedError(
            "The epc-reader 'get_resqml_v201_objects'-function does not work for "
            "Python 3.10. Consider switching to Python 3.11 or up."
        )

    robjs = []
    fail = {}

    with zipfile.ZipFile(epc_filename, "r") as zf:
        for zi in zf.infolist():
            with zf.open(zi.filename) as f:
                c = f.read()

                try:
                    robjs.append(parse_resqml_v201_object(c))
                except AttributeError as e:
                    fail[zi.filename] = e

    if log_failed_objects:
        logging.info(f"Failed to parse: {fail}")

    return robjs


def get_arrays_and_paths_in_hdf_file(
    epc_hdf_filename: str | pathlib.Path,
) -> dict[str, npt.NDArray[np.bool_ | np.number | np.character]]:
    """In this function we read all arrays and their full paths in an
    HDF5-file. In the context of an EPC-file this corresponds to the
    `pathInHdfFile`-attribute in a RESQML/WITSML/PRODML-object that has
    connected array data.


    Parameters
    ----------
    epc_hdf_filename: str | pathlib.Path
        The filename of the HDF5-file that is connected to an EPC-file. This is
        typically the same filename as the EPC-file, but with the
        `.epc`-extension replaced by `.h5`.


    Returns
    -------
    dict[str | npt.NDArray[np.bool_ | np.number | np.character]
        A dictionary mapping the `pathInHdfFile`-key to the actual array as a
        NumPy-array.


    See Also
    --------
    get_resqml_v201_objects
    """
    datasets = {}

    def populate_datasets(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            # The name does not include the top level group "/". This creates a
            # mismatch between the PathInHdfFile-element in the RESQML
            # documents.
            datasets["/" + name] = np.array(obj)

    with h5py.File(epc_hdf_filename, "r") as f:
        f.visititems(populate_datasets)

    return datasets
