// port-lint: source metal/commands.rs
package io.github.kotlinmania.candlemetalkernels.metal

import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.MetalKernelError

const val DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER: Long = 50L

class CommandQueue(
    val rawVal: Any? = null,
    var labelVal: String? = null,
) {
    fun setLabel(label: String?) {
        this.labelVal = label
    }

    fun addResidencySet(raw: Any?) {}

    companion object {
        fun new(raw: Any? = null): CommandQueue = CommandQueue(raw)
    }
}

fun createCommandBuffer(commandQueue: CommandQueue): CommandBuffer = CommandBuffer.new()

class CommandsGuard(
    val currentEncoder: ComputeCommandEncoder,
) : EncoderProvider {
    override fun encoder(): ComputeCommandEncoder = currentEncoder

    fun asRef(): ComputeCommandEncoder = currentEncoder

    fun setLabel(label: String) {
        asRef().setLabel(label)
    }

    fun setComputePipelineState(pipeline: ComputePipeline) {
        asRef().setComputePipelineState(pipeline)
    }

    fun debugGroup(label: String): DebugGroupGuard = asRef().debugGroup(label)
}

class BlitCommandsGuard(
    var stateVal: BlitCommandEncoder,
) {
    fun asRef(): BlitCommandEncoder = stateVal

    fun asMut(): BlitCommandEncoder = stateVal

    fun setLabel(label: String) {
        asRef().setLabel(label)
    }

    fun copyFromBuffer(
        srcBuffer: Buffer,
        srcOffset: Long,
        dstBuffer: Buffer,
        dstOffset: Long,
        size: Long,
    ) {
        asMut().copyFromBuffer(srcBuffer, srcOffset, dstBuffer, dstOffset, size)
    }

    fun fillBuffer(buffer: Buffer, range: BufferRange, value: UByte) {
        asMut().fillBuffer(buffer, range, value)
    }

    fun fillBuffer(buffer: Buffer, range: Pair<Long, Long>, value: UByte) {
        asMut().fillBuffer(buffer, range, value)
    }

    fun drop() {
        asRef().endEncoding()
    }
}

internal class EntryState(
    var current: CommandBuffer,
    val inFlight: MutableList<CommandBuffer> = mutableListOf(),
    var currentEncoder: ComputeCommandEncoder? = null,
) {
    companion object {
        fun new(cb: CommandBuffer): EntryState = EntryState(cb)
    }
}

class Commands(
    val commandQueue: CommandQueue,
    val device: Device,
    val computePerBuffer: Long = DEFAULT_CANDLE_METAL_COMPUTE_PER_BUFFER,
) {
    private val state: EntryState = EntryState.new(createCommandBuffer(commandQueue))
    private var computeCount: Long = 0L
    private val prevCeOutputs: PrevCeOutputs = mutableMapOf()

    fun commandEncoder(): CommandsGuard {
        val count = computeCount
        computeCount = count + 1L
        if (computeCount >= computePerBuffer) {
            commitSwapLocked(state, 1L)
        }
        var enc = state.currentEncoder
        if (enc == null) {
            val fence = Fence.new(device)
            enc = state.current.computeCommandEncoder(fence)
            state.currentEncoder = enc
        }
        return CommandsGuard(enc)
    }

    fun blitCommandEncoder(): BlitCommandsGuard {
        val count = computeCount
        computeCount = count + 1L
        if (computeCount >= computePerBuffer) {
            commitSwapLocked(state, 1L)
        }
        val enc = state.currentEncoder
        if (enc != null) {
            endEncoding(enc)
            state.currentEncoder = null
        }
        val fence = Fence.new(device)
        val blit = state.current.blitCommandEncoder(fence, prevCeOutputs)
        return BlitCommandsGuard(blit)
    }

    fun waitUntilCompleted() {
        flushAndWait()
    }

    fun flushAndWait() {
        if (computeCount > 0L) {
            commitSwapLocked(state, 0L)
        }
        val toWait = state.inFlight.toList()
        state.inFlight.clear()
        val last = toWait.lastOrNull()
        if (last != null) {
            ensureCompleted(last)
        }
        prevCeOutputs.clear()
    }

    fun flushAndWaitCurrent() {
        commitSwapLocked(state, 0L)
        val cb = state.inFlight.lastOrNull()
        if (cb != null) {
            ensureCompleted(cb)
            state.inFlight.removeAll { it.status() == MTLCommandBufferStatus.Completed }
        }
    }

    fun flush() {
        if (computeCount > 0L) {
            commitSwapLocked(state, 0L)
        }
    }

    private fun commitSwapLocked(state: EntryState, resetTo: Long) {
        val enc = state.currentEncoder
        if (enc != null) {
            endEncoding(enc)
            state.currentEncoder = null
        }
        when (state.current.status()) {
            MTLCommandBufferStatus.NotEnqueued, MTLCommandBufferStatus.Enqueued -> {
                state.current.commit()
            }

            else -> {}
        }
        val newCb = createCommandBuffer(commandQueue)
        val oldCb = state.current
        state.current = newCb
        state.inFlight.add(oldCb)
        computeCount = resetTo
    }

    private fun ensureCompleted(cb: CommandBuffer) {
        when (cb.status()) {
            MTLCommandBufferStatus.NotEnqueued, MTLCommandBufferStatus.Enqueued -> {
                cb.commit()
                cb.waitUntilCompleted()
            }

            MTLCommandBufferStatus.Committed, MTLCommandBufferStatus.Scheduled -> {
                cb.waitUntilCompleted()
            }

            MTLCommandBufferStatus.Completed -> {}

            MTLCommandBufferStatus.Error -> {
                val msg = cb.error() ?: "unknown error"
                MetalKernelError.CommandBufferError(msg).raise()
            }
        }
    }

    private fun endEncoding(encoder: ComputeCommandEncoder) {
        val allOutputs = encoder.state.allOutputs.toSet()
        val fence = encoder.fence ?: Fence.new(device)
        for (output in allOutputs) {
            prevCeOutputs[output] = fence
        }
        encoder.updateFence(fence)
        encoder.endEncoding()
    }

    fun drop() {
        flush()
    }

    companion object {
        fun new(
            commandQueue: CommandQueue,
            residencySet: ResidencySet? = null,
        ): Commands {
            if (residencySet != null) {
                commandQueue.addResidencySet(residencySet.raw())
            }
            val device = Device.systemDefault() ?: Device.new()
            return Commands(commandQueue, device)
        }
    }
}
