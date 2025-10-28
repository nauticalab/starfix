# Overview
Hashing Lib for Arrow Data Tables using C_Pointers with Arrow_Digest from: https://github.com/kamu-data/arrow-digest

# Usage
The repo is setup to use dev containers of VSCode. After starting up the container and connecting to it the process to install the rust lib and a python package is:
```maturin develop --uv```

NOTE: After every code edit in rust code, you will need to rerun the command to rebuild it then restart the kernel in the Jupyter Notebook side