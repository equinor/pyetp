import pathlib

import rich

from using_the_rddms_client import dataspaces, gri_resources, gri_lo

p = pathlib.Path("examples") / "tutorials" / "using_the_rddms_client"
width = 120

with open(p / "dataspaces.txt", "w") as f:
    rich.console.Console(width=width, file=f).print(dataspaces)

with open(p / "gri_resources.txt", "w") as f:
    rich.console.Console(width=width, file=f).print(gri_resources)

with open(p / "gri_lo.txt", "w") as f:
    rich.console.Console(width=width, file=f).print(gri_lo)
