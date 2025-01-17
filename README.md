# CellScanner v2.0

CellScanner is a tool that counts different microbial species in flow cytometry data of communities separately with the help of a classifier that is trained on mono-cultures. It is implemented in Python and compatible with Windows, Mac and Linux.


## How to use it 


### Graphical User Interface (GUI)


#### Linux and macOS

To run CellScanner you first need to get it along with its corresponding dependencies. 
To this end, you may run the following chunk to get CellScanner and create a `conda` environment for it:

```bash
git clone https://github.com/msysbio/CellScanner.git
cd CellScanner
conda env create -n cellscanner python=3.12.2 
conda activate cellscanner
pip install -r requirements.txt
```

To run CellScanner with its GUI, you may now run:

```bash
# Always remember to activate your conda environment, if you set one for CellScanner
conda activate cellscanner
# If `python` returns an error message that is not there, try with `python3` instead
python Cellscanner.py
```

This will pop-up CellScanner where you can now import your data, fill in your training parameters and 
predict your species. 

![gui_main](./GUI.png)


Keep in mind that the CellScanner GUI is a PyQt5 app, meaning it requires from the host to have a working X11 windowing system.


#### Windows

In Windows, you may also follow the steps described above for the Linux and macOS systems.
If you try through a WSL however, as already mentioned an X11 is required, which you would have to set up on your own.

Alternatively, you can build an `.exe` on your own
**Attention!** Do not use a WSL. Also, that [`pyinstaller`](https://pyinstaller.org/en/stable/) is available. 

Then, after you make sure you have activated the `cellscanner` conda environment, you may run:

```bash
pyinstaller --onefile --icon=logo.ico --add-data "logo.png:." Cellscanner.py
```

<!-- REMEMBER to add link -->
**.. or** 

you can simply download the `.exe` of CellScanner v2.0 from [here](). 



### Command Line Interface (CLI)


Assuming you already have CellScanner locally in a `conda` environment (see [above](./README.md#linux-and-macos)),
to run CellScanner using its CLI, you have first to fill in the [`config.yml`](./config.yml) file.
In this file, you may provide all the arguments you would do in the GUI case as well. 

Required arguments are mentioned as such, while it is important to remember that when providing monoculture data for the training step, their names (`species_names` in the yaml file) need to be in the same order as their filenames.

```yaml
species_files:
  directories:
    - directory:
        path: ./Testfiles/
        filenames:
          - BH_mono_48h_100.fcs
          - FP_mono_48h_100.fcs
          - RI_mono_48h_100.fcs
          - BT_mono_48h_100.fcs
        species_names:
          - Bacteroides
          - Faecalibacterium
          - Roseburia
          - Bacteroides
```
Once your configuration file is ready, you may run CellScanner CLI :

```bash
conda activate cellscanner
python CellscannerCLI.py --config config.yml
```


When using the CLI version, CellScanner is independent of PyQt5, thus no X11 issues should be occurred. 



## The logic 

CellScanner v2.0 is based on the [first version of the tool](https://github.com/Clem-Jos/CellScanner/tree/main). 

For an overview of what is under the hood, you may have a look on the [User manual of CellScanner v1.0](https://github.com/Clem-Jos/CellScanner/blob/main/CellScanner_1.1.0/CellScanner_user_manual.pdf).
For the new features that have been added, a manuscript is in process. :pencil:





