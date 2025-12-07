FROM rust:latest

RUN cargo install wasm-pack \
    && rustup target add wasm32-unknown-unknown

CMD ["/bin/bash"]
