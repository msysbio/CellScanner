Installation
============

CellScanner has been tested with Python 3.12 on Linux, macOS and Windows.
You can run it from the source code, or with Docker.


## From the source code

### Linux and macOS

Get CellScanner and create a `conda` environment for it:

```bash
git clone https://github.com/msysbio/CellScanner.git
cd CellScanner
conda create -n cellscanner python=3.12.2
conda activate cellscanner
pip install -r requirements.txt
```

Then, from the root folder of the repository, start the GUI:

```bash
python cellscanner/Cellscanner.py
```

or use the command line interface (see the [CLI tutorial](./cli.md)):

```bash
python cellscanner/CellscannerCLI.py -c config.yml
```

The GUI is a PyQt5 app and needs a desktop session to open its window (on Linux, an X11 or Wayland session).
The CLI does not.

### Windows

The steps are the same, run in the **Anaconda Prompt** (or **Miniforge Prompt**) that comes with your conda installation.
If you do not have `git`, you can instead download the repository as a ZIP file from
[GitHub](https://github.com/msysbio/CellScanner) (*Code* → *Download ZIP*) and unzip it.

```bash
git clone https://github.com/msysbio/CellScanner.git
cd CellScanner
conda create -n cellscanner python=3.12.2
conda activate cellscanner
pip install -r requirements.txt
```

Start the GUI or the CLI from the root folder of the repository (note the `\` in the paths):

```bash
python cellscanner\Cellscanner.py
```

```bash
python cellscanner\CellscannerCLI.py -c config.yml
```

Run CellScanner directly in Windows like this rather than in WSL: Windows shows the GUI window itself,
whereas in WSL the GUI depends on WSL's support for Linux graphical apps.


## With Docker

The Docker image contains CellScanner with all its dependencies, for both the GUI and the CLI.
It needs [Docker](https://docs.docker.com/get-docker/) (Docker Desktop on Windows and macOS).

Get the image, either from Docker Hub:

```bash
docker pull hariszaf/cell_scanner
```

or by building it from the repository's root folder:

```bash
docker build -t hariszaf/cell_scanner .
```

Put your `.fcs` files (and, for the CLI, your configuration file) in one folder and mount it to `/media` in the container:
CellScanner reads its inputs from there and writes its results back to it.

**CLI** (on any system; in the configuration file, refer to your files as `/media/...`):

```bash
docker run --rm -v ./Testfiles:/media hariszaf/cell_scanner python CellscannerCLI.py -c /media/config.yml
```

On Linux, add `--user $(id -u):$(id -g)` so that the result files belong to you rather than to root.
In Windows PowerShell, write the folder as `-v ${PWD}\Testfiles:/media`.

**GUI** (Linux): allow the container to use your display, then start it:

```bash
xhost +local:
docker run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v ./Testfiles:/media hariszaf/cell_scanner
```

On Windows and macOS, the GUI in Docker needs an X server (e.g. VcXsrv or XQuartz);
there, running the GUI from the source code, as described above, is simpler.
