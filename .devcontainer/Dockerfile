# DevContainer image
FROM ubuntu:24.04

RUN apt-get update &&\ 
    apt-get install software-properties-common -y && \
    apt-add-repository ppa:git-core/ppa && \
    apt-get install git -y

RUN \
    # dev setup
    apt update && \
    apt-get install build-essential sudo jq bash-completion graphviz rsync software-properties-common curl gnupg lsb-release -y && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    echo '. /etc/bash_completion' >> /root/.bashrc && \
    echo 'export PS1="\[\e[32;1m\]\u\[\e[m\]@\[\e[34;1m\]\H\[\e[m\]:\[\e[33;1m\]\w\[\e[m\]$ "' >> /root/.bashrc && \
    apt-get clean

ENV PATH="/root/.cargo/bin:${PATH}"
RUN apt-get update && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | bash -s -- -y && \
    # rust auto formatting
    rustup component add rustfmt && \
    # rust style linter
    rustup component add clippy && \
    # rust code coverage
    cargo install cargo-llvm-cov && \
    rustup component add llvm-tools-preview && \
    # rust crate structure diagram
    cargo install cargo-modules && \
    # expand rust macros (useful in debugging)
    cargo install cargo-expand && \
    apt-get clean

ENV PATH=${PATH}:/root/.local/bin
RUN \
    # install python manager
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    uv venv -p 3.10 ~/.local/share/base && \
    # pip package based on C lib/client
    uv pip install cffi maturin[patchelf] -p ~/.local/share/base && \
    # useful in examples
    uv pip install ipykernel eclipse-zenoh pyarrow -p ~/.local/share/base && \
    echo '. ~/.local/share/base/bin/activate' >> ~/.bashrc
ENV VIRTUAL_ENV=/root/.local/share/base
