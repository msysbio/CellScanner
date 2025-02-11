Installation 
============

CellScanner has been tested in Python 3.12.x in both Linux, macOS and Windows platforms. 

In the following sections we describe how you can access it.


## Linux and macOS

To run CellScanner you first need to get it along with its corresponding dependencies. 
To this end, you may run the following chunk to get CellScanner and create a `conda` environment for it:

```bash
git clone https://github.com/msysbio/CellScanner.git
cd CellScanner
conda create -n cellscanner python=3.12.2 
conda activate cellscanner
pip install -r requirements.txt
```

Then, to fire the GUI you just need to run 

```bash
./cellscanner/Cellscanner.py
```

from the root folder of the repo.


## Windows

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



