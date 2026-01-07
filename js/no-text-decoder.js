if (!globalThis.TextDecoder) {
    globalThis.TextDecoder = class {
        decode() {
            return ''
        }
    }
}
