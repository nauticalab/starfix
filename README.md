# Overview
Hashing Lib for Arrow Data Tables using C_Pointers with Arrow_Digest from: https://github.com/kamu-data/arrow-digest

# Usage
The repo is setup to use dev containers of VSCode. After starting up the container and connecting to it the process to install the rust lib and a python package is:
```maturin develop --uv```

NOTE: After every code edit in rust code, you will need to rerun the command to rebuild it then restart the kernel in the Jupyter Notebook side


# Hashing System Overview
ArrowDigester stores the digest for multiple components of the arrow data table before combining them

- schema: Each field name is serialized via PostCard: https://docs.rs/postcard/latest/postcard/
    - Was chosen since I was originally using JSON but wanted something even faster, hence postcard. It is design to be very resource efficient

- fields_digest_buffer: Flattens all nested schema with the '__' delimiter between the parent and sub level in this format parent_field_name__child_field_name

- Upon finalization of the hash, the instance consume itself due to digest.finalize consuming self under ``field_digest_buffer``. Following that it adds it to a final digest in this order: schema + field_digest_buffer (lexical order of the field name)
