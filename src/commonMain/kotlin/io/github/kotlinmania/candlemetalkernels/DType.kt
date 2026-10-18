package io.github.kotlinmania.candlemetalkernels

enum class DType {
    BF16,
    F16,
    F32,
    I64,
    U32,
    U8;

    fun sizeInBytes(): Long = when (this) {
        U8 -> 1L
        U32 -> 4L
        I64 -> 8L
        BF16 -> 2L
        F16 -> 2L
        F32 -> 4L
    }
}
