// port-lint: source metal/residency_set.rs
package io.github.kotlinmania.candlemetalkernels.metal

class ResidencySet(
    val rawVal: Any? = null,
) {
    private val buffers: MutableSet<Buffer> = mutableSetOf()

    fun raw(): Any? = rawVal

    fun insert(buf: Buffer) {
        asAllocation(buf)
        buffers.add(buf)
    }

    fun remove(buf: Buffer) {
        asAllocation(buf)
        buffers.remove(buf)
    }

    companion object {
        fun new(device: Device): ResidencySet = ResidencySet(device)
    }
}

private fun asAllocation(buf: Buffer): Any = buf
