#!/bin/sh

main() {
    set -ex
    workdir="$(mktemp -d)"
    trap cleanup EXIT
    generate_documentation
    push_to_gh_pages
}

generate_documentation() {
    rm -rf pkg
    wasm-pack build --no-typescript . --release
    tsc pkg/*.js --declaration --allowJs --emitDeclarationOnly --outDir pkg
    typedoc --out "$workdir"/docs
}

push_to_gh_pages() {
    mkdir "$workdir"/gh-pages
    git worktree add "$workdir"/gh-pages gh-pages
    rsync -av --exclude .git --delete "$workdir"/docs/ "$workdir"/gh-pages/
    cd "$workdir"/gh-pages
    git config --global user.name "$GITHUB_ACTOR"
    git config --global user.email "$GITHUB_ACTOR@users.noreply.github.com"
    git commit -m "Update"
    git push
}

cleanup() {
    rm -rf "$workdir"
}

main
