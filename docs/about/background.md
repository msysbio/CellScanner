Background
============


CellScanner is a tool that counts different microbial species in flow cytometry data of communities separately 
with the help of a classifier that is trained on mono-cultures. 
It is implemented in Python and compatible with Windows, Mac and Linux.

Flow cytometry measures thousands of particles (events) per sample, but on its own it does not tell
which species an event belongs to. CellScanner learns this from **monocultures**, i.e. samples of a single species,
and **blanks**, i.e. samples of the cell-free medium:

1. Optionally, stains such as SYBR-Green and propidium iodide are used to gate out debris and dead cells (*line gating*).
2. Events from the monocultures and blanks are embedded with UMAP, and events that do not resemble
   other events of their own species or blank are filtered out.
3. A neural network is trained on the remaining events to tell the species and the blank apart.
4. The trained network classifies each event of a **co-culture** (a community sample). Events it cannot
   assign with confidence are reported as `Unknown`, and gating can split each species into live, dead and debris.

The result is a count of each species in every co-culture sample, along with plots and quality metrics of the model.
CellScanner can be used through a graphical interface ([GUI tutorial](../tutorials/gui.md))
or the command line ([CLI tutorial](../tutorials/cli.md)).

CellScanner v2 builds on the [first version of the tool](https://github.com/Clem-Jos/CellScanner).
Bugs and feature requests are tracked on [GitHub](https://github.com/msysbio/CellScanner/issues).


Social
------
