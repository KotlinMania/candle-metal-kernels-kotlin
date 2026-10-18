// port-lint: source utils.rs
package io.github.kotlinmania.candlemetalkernels

import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.CommandsGuard
import io.github.kotlinmania.candlemetalkernels.metal.ComputeCommandEncoder
import io.github.kotlinmania.candlemetalkernels.metal.ComputePipeline
import io.github.kotlinmania.candlemetalkernels.metal.MTLSize

class WrappedEncoder(
    val inner: ComputeCommandEncoder,
    val endEncodingOnDrop: Boolean = false,
) {
    fun asRef(): ComputeCommandEncoder = inner

    fun drop() {
        if (endEncodingOnDrop) {
            inner.endEncoding()
        }
    }
}

fun linearSplit(pipeline: ComputePipeline, length: Long): Pair<MTLSize, MTLSize> {
    val size = length
    val width = minOf(pipeline.maxTotalThreadsPerThreadgroup(), size)
    val count = if (width > 0L) (size + width - 1L) / width else 1L
    val threadGroupCount = MTLSize(count, 1L, 1L)
    val threadGroupSize = MTLSize(width, 1L, 1L)
    return Pair(threadGroupCount, threadGroupSize)
}

fun getBlockDims(dim0: Long, dim1: Long, dim2: Long): MTLSize {
    var pows0 = 0
    var pows1 = 0
    var pows2 = 0
    var sum = 0
    while (true) {
        val presum = sum
        if (dim0 >= (1L shl (pows0 + 1))) {
            pows0 += 1
            sum += 1
        }
        if (sum == 10) break
        if (dim1 >= (1L shl (pows1 + 1))) {
            pows1 += 1
            sum += 1
        }
        if (sum == 10) break
        if (dim2 >= (1L shl (pows2 + 1))) {
            pows2 += 1
            sum += 1
        }
        if (sum == presum || sum == 10) break
    }
    return MTLSize(1L shl pows0, 1L shl pows1, 1L shl pows2)
}

fun getTileSize(dtypeSize: Long): Long = maxOf(1L, 8L / dtypeSize)

interface EncoderParam {
    fun setParam(encoder: ComputeCommandEncoder, position: Long)
}

fun setParam(encoder: ComputeCommandEncoder, position: Long, data: Any?) {
    when (data) {
        is Input -> data.setParam(encoder, position)
        is Output -> data.setParam(encoder, position)
        is BufferOffset -> encoder.setInputBuffer(position, data.buffer, data.offsetInBytes)
        is Buffer -> encoder.setInputBuffer(position, data, 0L)
        is EncoderParam -> data.setParam(encoder, position)
        else -> encoder.setBytes(position, data)
    }
}

class BufferOffset(
    val buffer: Buffer,
    val offsetInBytes: Long = 0L,
) {
    companion object {
        fun zeroOffset(buffer: Buffer): BufferOffset = BufferOffset(buffer, 0L)
    }
}

class Input(
    val buffer: Buffer,
    val offset: Long = 0L,
) : EncoderParam {
    override fun setParam(encoder: ComputeCommandEncoder, position: Long) {
        encoder.setInputBuffer(position, buffer, offset)
    }

    companion object {
        fun new(buffer: Buffer): Input = Input(buffer, 0L)

        fun withOffset(buffer: Buffer, offset: Long): Input = Input(buffer, offset)

        fun fromBufferOffset(bo: BufferOffset): Input = Input(bo.buffer, bo.offsetInBytes)
    }
}

class Output(
    val buffer: Buffer,
    val offset: Long = 0L,
) : EncoderParam {
    override fun setParam(encoder: ComputeCommandEncoder, position: Long) {
        encoder.setOutputBuffer(position, buffer, offset)
    }

    companion object {
        fun new(buffer: Buffer): Output = Output(buffer, 0L)

        fun withOffset(buffer: Buffer, offset: Long): Output = Output(buffer, offset)

        fun fromBufferOffset(bo: BufferOffset): Output = Output(bo.buffer, bo.offsetInBytes)
    }
}

interface EncoderProvider {
    fun encoder(): ComputeCommandEncoder
}

class ComputeCommandEncoderProvider(
    private val encoder: ComputeCommandEncoder,
) : EncoderProvider {
    override fun encoder(): ComputeCommandEncoder = encoder
}

class CommandsGuardProvider(
    private val guard: CommandsGuard,
) : EncoderProvider {
    override fun encoder(): ComputeCommandEncoder = guard.asRef()
}

class RwLockGuard<T>(val targetVal: T) {
    typealias Target = T

    fun deref(): T = targetVal

    companion object {
        fun <T> from(g: T): RwLockGuard<T> = RwLockGuard(g)
    }
}

private fun isTruthy(s: String): Boolean = when (s.lowercase()) {
    "true", "t", "yes", "y", "1" -> true
    else -> false
}

fun getEnvBool(key: String, default: Boolean): Boolean = default
