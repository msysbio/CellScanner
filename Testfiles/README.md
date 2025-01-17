# Use cases 

You may use the following files both with the GUI and the CLI. 

You may use these files to test CellScanner. 
More specifically,  

- `BH_mono_48h_100.fcs`  
- `FP_mono_48h_100.fcs`
- `RI_mono_48h_100.fcs`

should be considered as you species files, while 

- control_RCM_48h_100.fcs

would be your blank, and 

- `coculture1_48h_100.fcs`

the coculture you wish to predict. 

The template [`config.yml` file](../config.yml) represents this case.

Files taken from: https://flowrepository.org/public_experiment_representations/4026


## Tricks for the `config.yml` file

To give an example of how to fill in the `config.yml` file, we keep the same files 
under different folders specifying what they stand for:

- `blanks`
- `monocultures`
- `cocultures`

You can now set 

```yaml
blank_files:
  directories:
    - directory:
        path: ./Testfiles/banks
        filenames:
```

without specifying any file under the `filenames` and all the files of the folder will be used as blanks.
The same applies for the cocultures. 

> **ATTENTION** 
> This is not the case for the **monocultures**. 
> In their case, we need to map each monoculture .fcs file to the *species name* it's coming from. 
> However, you may use more than one .fcs for the same species. 
> In that case, your `yaml` should look like this:

```yaml
species_files:
  directories:
    - directory:
        path: ./to_my_test_files/
        filenames:
          - BH_mono_1.fcs
          - BH_mono_2.fcs
          - RI_mono_1.fcs
        species_names:
          - BH
          - BH
          - RI
```

It is **essential** you give the exact same name (in our case `BH`) to the .fcs files of the same species, and to 
provide them **in the same order**.