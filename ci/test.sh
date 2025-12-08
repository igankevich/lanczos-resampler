#!/bin/sh

main() {
    set -ex
    os="$1"
    workdir="$(mktemp -d)"
    trap cleanup EXIT
    cargo_clippy
    cargo_test
    cargo_build
    enable_cdylib
    wasm_pack_build
    wasm_pack_test
    wasm_integration_tests
}

cargo_clippy() {
    cargo clippy --workspace --quiet --all-features --all-targets -- --deny warnings
}

cargo_test() {
    cargo test --workspace --quiet --no-fail-fast --all-features
}

cargo_build() {
    cargo build
    cargo build --features alloc
    cargo build --features std
}

enable_cdylib() {
    sed -i -e 's/crate-type = .*/crate-type = ["cdylib", "rlib"]/' Cargo.toml
}

wasm_pack_build() {
    wasm-pack build
    wasm-pack build . --features alloc
    wasm-pack build . --features std
    du -chd0 pkg
}

wasm_pack_test() {
    wasm-pack test --release --node
    case "$os" in
    ubuntu)
        env WASM_BINDGEN_USE_BROWSER=1 wasm-pack test --release --headless --firefox
        env WASM_BINDGEN_USE_BROWSER=1 wasm-pack test --release --headless --chrome
        ;;
    # TODO Fails with "Error: driver failed to bind port during startup"
    #macos)
    #    env WASM_BINDGEN_USE_BROWSER=1 wasm-pack test --release --headless --safari
    #    ;;
    esac
}

wasm_integration_tests() {
    node --experimental-wasm-modules tests.js
}

cleanup() {
    rm -rf "$workdir"
}

main "$@"
