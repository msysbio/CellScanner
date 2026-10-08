Background
============


CellScanner is a tool that counts different microbial species in flow cytometry data of communities separately 
with the help of a classifier that is trained on mono-cultures. 
It is implemented in Python and compatible with Windows, Mac and Linux.
See the [installation](../tutorials/install.md) page for how to set it up.

Flow cytometry measures thousands of particdriles (events) per sample, but on its own it does not tell which species an event belongs to. 
CellScanner learns this from **monocultures**, i.e. samples of a single species, and **blanks**, i.e. samples of the cell-free medium:

1. Optionally, stains such as SYBR-Green and propidium iodide are used to gate out debris and dead cells (*line gating*).
2. Events from the monocultures and blanks are embedded with UMAP, and events that do not resemble
   other events of their own species or blank are filtered out.
3. A neural network is trained on the remaining events to tell the species and the blank apart.
4. The trained network classifies each event of a **co-culture** (a community sample). Events it cannot
   assign with confidence are reported as `Unknown`, and gating can split each species into live, dead and debris.

The result is a count of each species in every co-culture sample, along with plots and quality metrics of the model.
You can use CellScanner through its [graphical user interface (GUI)](../tutorials/gui.md) or its
[command line interface (CLI)](../tutorials/cli.md); each comes with a step-by-step tutorial.

CellScanner v2 builds on the [first version of the tool](https://github.com/Clem-Jos/CellScanner).

Contact
-------

- **Bugs, questions, feature requests and ideas:** please open an issue on [GitHub](https://github.com/msysbio/CellScanner/issues).

If you have any ideas for future features, feel free to get in touch.
