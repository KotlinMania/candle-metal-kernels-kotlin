// port-lint: source kernels/macros.rs
package io.github.kotlinmania.candlemetalkernels.kernels

data class Kernel(val name: String)

class OpKernelGroup(prefix: String, suffix: String = "") {
    val FLOAT: Kernel = Kernel("${prefix}_f32$suffix")
    val HALF: Kernel = Kernel("${prefix}_f16$suffix")
    val BFLOAT: Kernel = Kernel("${prefix}_bf16$suffix")
    val I64: Kernel = Kernel("${prefix}_i64$suffix")
    val U32: Kernel = Kernel("${prefix}_u32$suffix")
    val U8: Kernel = Kernel("${prefix}_u8$suffix")
}
