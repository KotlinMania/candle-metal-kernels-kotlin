// port-lint: source metal/encoder.rs
package io.github.kotlinmania.candlemetalkernels.metal

data class MTLSize(val width: Long, val height: Long, val depth: Long)

typealias PrevCeOutputs = MutableMap<Long, Fence>

class EncoderState(
    val prevOutputs: MutableSet<Long> = mutableSetOf(),
    val nextOutputs: MutableSet<Long> = mutableSetOf(),
    val prevInputs: MutableSet<Long> = mutableSetOf(),
    val nextInputs: MutableSet<Long> = mutableSetOf(),
    var needsBarrier: Boolean = false,
    val allInputs: MutableSet<Long> = mutableSetOf(),
    val allOutputs: MutableSet<Long> = mutableSetOf(),
) {
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

class ComputeCommandEncoder(
    val state: EncoderState = EncoderState.new(),
    val fence: Fence? = null,
    var label: String? = null,
    var currentPipeline: ComputePipeline? = null,
) {
    fun asRef(): ComputeCommandEncoder = this

    fun setThreadgroupMemoryLength(length: Long, index: Long) {}

    fun dispatchThreads(threadsPerGrid: MTLSize, threadsPerThreadgroup: MTLSize) {}

    fun dispatchThreadGroups(threadgroupsPerGrid: MTLSize, threadsPerThreadgroup: MTLSize) {}

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
        this.label = label
    }

    fun debugGroup(label: String): DebugGroupGuard = DebugGroupGuard(this, label)

    private fun drop() {
        endEncoding()
    }

    companion object {
        fun new(state: EncoderState = EncoderState.new(), fence: Fence? = null): ComputeCommandEncoder =
            ComputeCommandEncoder(state, fence)
    }
}

class BlitCommandEncoder(
    val fence: Fence? = null,
    val prevCeOutputs: PrevCeOutputs? = null,
    var label: String? = null,
) {
    fun asRef(): BlitCommandEncoder = this

    fun waitForFence(fence: Fence) {}

    fun updateFence(fence: Fence) {}

    fun endEncoding() {}

    fun setLabel(label: String) {
        this.label = label
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

    fun fillBuffer(buffer: Buffer, range: Pair<Long, Long>, value: UByte) {
        val data = buffer.data()
        val start = range.first.toInt()
        val len = range.second.toInt()
        val b = value.toByte()
        val end = minOf(data.size, start + len)
        for (i in start until end) {
            data[i] = b
        }
    }

    companion object {
        fun new(fence: Fence? = null, prevCeOutputs: PrevCeOutputs? = null): BlitCommandEncoder =
            BlitCommandEncoder(fence, prevCeOutputs)
    }
}
