export default {
    compilerOptions: {
        lib: ['esnext'],
        allowJs: true,
    },
    highlightLanguages: ['rust', 'javascript', 'bash'],
    excludeNotDocumented: true,
    entryPoints: ['pkg/lanczos_resampler.d.ts', 'pkg/loader.js'],
}
