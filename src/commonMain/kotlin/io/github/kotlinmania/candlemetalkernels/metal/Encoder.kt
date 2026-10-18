// port-lint: source metal/encoder.rs
package io.github.kotlinmania.candlemetalkernels.metal

import io.github.kotlinmania.candlemetalkernels.EncoderProvider

data class MTLSize(
    val width: Long,
    val height: Long,
    val depth: Long,
)

internal typealias PrevCeOutputs = MutableMap<Long, Fence>

internal class EncoderState {
    val prevOutputs: MutableSet<Long> = mutableSetOf()
    val nextOutputs: MutableSet<Long> = mutableSetOf()
    val prevInputs: MutableSet<Long> = mutableSetOf()
    val nextInputs: MutableSet<Long> = mutableSetOf()
    var needsBarrier: Boolean = false
    val allInputs: MutableSet<Long> = mutableSetOf()
    val allOutputs: MutableSet<Long> = mutableSetOf()

    companion object {
        fun new(): EncoderState = EncoderState()
    }
}

class DebugGroupGuard(
    private val encoder: ComputeCommandEncoder,
    private val label: String,
) {
    fun drop() {}
}

class ComputeCommandEncoder internal constructor(
    internal val state: EncoderState = EncoderState.new(),
    internal val fence: Fence? = null,
    private var labelVal: String? = null,
    internal var currentPipeline: ComputePipeline? = null,
) : EncoderProvider {
    fun label(): String? = labelVal

    override fun encoder(): ComputeCommandEncoder = this

    fun asRef(): ComputeCommandEncoder = this

    fun setThreadgroupMemoryLength(length: Long, index: Long) {}

    fun dispatchThreads(threadsPerGrid: MTLSize, threadsPerThreadgroup: MTLSize) {
        autoBarrier()
    }

    fun dispatchThreadGroups(threadgroupsPerGrid: MTLSize, threadsPerThreadgroup: MTLSize) {
        autoBarrier()
    }

    private fun autoBarrier() {}

    fun setInputBuffer(index: Long, buffer: Buffer?, offset: Long) {
        if (buffer != null) {
            val ptr = buffer.rawPtr()
            state.allInputs.add(ptr)
            state.nextInputs.add(ptr)
        }
    }

    fun setOutputBuffer(index: Long, buffer: Buffer?, offset: Long) {
        if (buffer != null) {
            val ptr = buffer.rawPtr()
            state.allOutputs.add(ptr)
            state.nextOutputs.add(ptr)
        }
    }

    fun setBytesDirectly(index: Long, length: Long, pointer: Any?) {}

    fun setBytes(index: Long, data: Any?) {}

    fun setComputePipelineState(pipeline: ComputePipeline) {
        this.currentPipeline = pipeline
    }

    fun insertMemoryBarrier() {}

    fun waitForFence(fence: Fence) {}

    fun updateFence(fence: Fence) {}

    fun endEncoding() {}

    fun encodePipeline(pipeline: ComputePipeline, f: (ComputeCommandEncoder) -> Unit) {
        setComputePipelineState(pipeline)
        f(this)
    }

    fun setLabel(label: String) {
        this.labelVal = label
    }

    fun debugGroup(label: String): DebugGroupGuard = DebugGroupGuard(this, label)

    fun drop() {
        endEncoding()
    }

    companion object {
        fun new(fence: Fence? = null): ComputeCommandEncoder =
            ComputeCommandEncoder(EncoderState.new(), fence)

        internal fun new(state: EncoderState, fence: Fence? = null): ComputeCommandEncoder =
            ComputeCommandEncoder(state, fence)
    }
}

data class BufferRange(
    val start: Long,
    val length: Long,
)

class BlitCommandEncoder internal constructor(
    internal val fence: Fence? = null,
    internal val prevCeOutputs: PrevCeOutputs? = null,
    private var labelVal: String? = null,
) {
    fun label(): String? = labelVal

    fun asRef(): BlitCommandEncoder = this

    fun waitForFence(fence: Fence) {}

    fun updateFence(fence: Fence) {}

    fun endEncoding() {}

    fun setLabel(label: String) {
        this.labelVal = label
    }

    fun copyFromBuffer(
        sourceBuffer: Buffer,
        sourceOffset: Long,
        destinationBuffer: Buffer,
        destinationOffset: Long,
        size: Long,
    ) {
        val src = sourceBuffer.data()
        val dst = destinationBuffer.data()
        val srcOff = sourceOffset.toInt()
        val dstOff = destinationOffset.toInt()
        val len = size.toInt()
        if (srcOff + len <= src.size && dstOff + len <= dst.size) {
            src.copyInto(dst, dstOff, srcOff, srcOff + len)
        }
    }

    fun fillBuffer(buffer: Buffer, range: BufferRange, value: UByte) {
        val data = buffer.data()
        val start = range.start.toInt()
        val len = range.length.toInt()
        val b = value.toByte()
        val end = minOf(data.size, start + len)
        for (i in start until end) {
            data[i] = b
        }
    }

    fun fillBuffer(buffer: Buffer, range: Pair<Long, Long>, value: UByte) {
        fillBuffer(buffer, BufferRange(range.first, range.second), value)
    }

    companion object {
        internal fun new(fence: Fence? = null, prevCeOutputs: PrevCeOutputs? = null): BlitCommandEncoder =
            BlitCommandEncoder(fence, prevCeOutputs)
    }
}
