// port-lint: source metal/fence.rs
package io.github.kotlinmania.candlemetalkernels.metal

class Fence(
    val rawVal: Any? = null,
) {
    fun raw(): Any? = rawVal

    companion object {
        fun new(device: Device): Fence = Fence(device)
    }
}
