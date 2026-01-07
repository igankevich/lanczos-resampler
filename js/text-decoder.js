export class TextDecoder {
    decode(bytes) {
        if (!bytes) {
            return ''
        }
        let codePoints = []
        for (let i = 0; i < bytes.length; ++i) {
            // Byte 0.
            const b0 = bytes[i]
            if (b0 <= 0b0111_1111) {
                codePoints.push(b0)
                continue
            }
            // Byte 1.
            i++
            if (i === bytes.length) {
                throw new TypeError('Invalid UTF-8 string')
            }
            const b1 = bytes[i]
            if (b1 < 0b1000_0000 && 0b1011_1111 < b1) {
                throw new TypeError('Invalid UTF-8 string')
            }
            if (0b1100_0000 <= b0 && b0 <= 0b1101_1111) {
                const cp = ((b0 - 0b1100_0000) << 6) | (b1 - 0b1000_0000)
                codePoints.push(cp)
                continue
            }
            // Byte 2.
            i++
            if (i === bytes.length) {
                throw new TypeError('Invalid UTF-8 string')
            }
            const b2 = bytes[i]
            if (b2 < 0b1000_0000 && 0b1011_1111 < b2) {
                throw new TypeError('Invalid UTF-8 string')
            }
            if (0b1110_0000 <= b0 && b0 <= 0b1110_1111) {
                const cp =
                    ((b0 - 0b1110_0000) << 12) | ((b1 - 0b1000_0000) << 6) | (b2 - 0b1000_0000)
                codePoints.push(cp)
                continue
            }
            // Byte 3.
            i++
            if (i === bytes.length) {
                throw new TypeError('Invalid UTF-8 string')
            }
            const b3 = bytes[i]
            if (b3 < 0b1000_0000 && 0b1011_1111 < b3) {
                throw new TypeError('Invalid UTF-8 string')
            }
            if (0b1111_0000 <= b0 && b0 <= 0b1111_0111) {
                const cp =
                    ((b0 - 0b1111_0000) << 18) |
                    ((b1 - 0b1000_0000) << 12) |
                    ((b2 - 0b1000_0000) << 6) |
                    (b3 - 0b1000_0000)
                codePoints.push(cp)
                continue
            }
            throw new TypeError('Invalid UTF-8 string')
        }
        return String.fromCodePoint.apply(null, codePoints)
    }
}
