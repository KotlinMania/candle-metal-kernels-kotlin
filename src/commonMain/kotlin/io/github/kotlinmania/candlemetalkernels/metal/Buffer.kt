// port-lint: source metal/buffer.rs
package io.github.kotlinmania.candlemetalkernels.metal

data class NSRange(val location: Long, val length: Long)

data class MTLResourceOptions(val value: ULong) {
    companion object {
        val StorageModeShared: MTLResourceOptions = MTLResourceOptions(0uL)
        val StorageModePrivate: MTLResourceOptions = MTLResourceOptions(2uL)
        val HazardTrackingModeUntracked: MTLResourceOptions = MTLResourceOptions(0x100uL)
    }
}

typealias MetalResource = Any
typealias BufferMap = Map<Long, List<Buffer>>

class Buffer(
    val lengthVal: Long,
    val dataVal: ByteArray = ByteArray(lengthVal.toInt()),
    var labelVal: String? = null,
) {
    fun contents(): ByteArray = dataVal

    fun data(): ByteArray = dataVal

    fun rawPtr(): Long = hashCode().toLong()

    fun length(): Long = lengthVal

    fun didModifyRange(range: NSRange) {}

    fun setLabel(label: String) {
        this.labelVal = label
    }

    fun asRef(): Buffer = this

    companion object {
        fun new(lengthVal: Long, dataVal: ByteArray = ByteArray(lengthVal.toInt())): Buffer =
            Buffer(lengthVal, dataVal)
    }
}
