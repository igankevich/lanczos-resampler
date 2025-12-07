#!/bin/sh

main() {
    set -ex
    os="$1"
    cargo_clippy
    cargo_test
    wasm_pack_build
    wams_pack_test
    wasm_integration_tests
}

cargo_clippy() {
    cargo clippy --workspace --quiet --all-features --all-targets -- --deny warnings
}

cargo_test() {
    cargo test --workspace --quiet --no-fail-fast --all-features
}

wasm_pack_build() {
    wasm-pack build
    du -chd0 pkg
}

do_wasm_pack_test() {
    wasm-pack test --headless --node --firefox --chrome "$@"
}

wasm_pack_test() {
    case "$os" in
    linux | windows) do_wasm_pack_test ;;
    macos) do_wasm_pack_test --safari ;;
    *)
        printf "Unknown os \"%s\"\n" "$os" >&2
        exit 1
        ;;
    esac
}

wasm_integration_tests() {
    node tests.js
}

main "$@"
