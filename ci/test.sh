#!/bin/sh

main() {
    set -ex
    os="$1"
    workdir="$(mktemp -d)"
    trap cleanup EXIT
    cargo_clippy
    cargo_test
    wasm_pack_build
    add_browsers_to_path
    wasm_pack_test
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

add_browsers_to_path() {
    mkdir "$workdir"/browsers
    ln -s "$CHROME_PATH" "$workdir"/browsers/chrome
    ln -s "$CHROMEDRIVER_PATH" "$workdir"/browsers/chromedriver
    ln -s "$FIREFOX_PATH" "$workdir"/browsers/firefox
    export PATH="$workdir"/browsers:"$PATH"
}

do_wasm_pack_test() {
    env WASM_BINDGEN_USE_BROWSER=1 \
        wasm-pack test --headless --node --firefox --chrome "$@"
    wasm-pack test --node
}

wasm_pack_test() {
    case "$os" in
    macos) do_wasm_pack_test --safari ;;
    *) do_wasm_pack_test ;;
    esac
}

wasm_integration_tests() {
    node tests.js
}

cleanup() {
    rm -rf "$workdir"
}

main "$@"
