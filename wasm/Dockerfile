FROM rust:latest

RUN cargo install wasm-pack \
    && rustup target add wasm32-unknown-unknown

RUN cargo install wasm-bindgen-cli
RUN cargo install wasm-opt

CMD ["/bin/bash"]
