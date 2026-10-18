// port-lint: source metal/command_buffer.rs
package io.github.kotlinmania.candlemetalkernels.metal

enum class MTLCommandBufferStatus {
    NotEnqueued,
    Enqueued,
    Committed,
    Scheduled,
    Completed,
    Error,
}

class CommandBuffer(
    var labelVal: String? = null,
    var statusVal: MTLCommandBufferStatus = MTLCommandBufferStatus.NotEnqueued,
) {
    fun label(): String? = labelVal

    fun asRef(): CommandBuffer = this

    fun computeCommandEncoder(fence: Fence): ComputeCommandEncoder =
        ComputeCommandEncoder.new(fence = fence)

    fun computeCommandEncoderNoFence(): ComputeCommandEncoder =
        ComputeCommandEncoder.new()

    fun blitCommandEncoder(fence: Fence, prevCeOutputs: PrevCeOutputs): BlitCommandEncoder =
        BlitCommandEncoder.new(fence, prevCeOutputs)

    fun commit() {
        statusVal = MTLCommandBufferStatus.Committed
    }

    fun enqueue() {
        statusVal = MTLCommandBufferStatus.Enqueued
    }

    fun setLabel(label: String) {
        this.labelVal = label
    }

    fun status(): MTLCommandBufferStatus = statusVal

    fun error(): String? = null

    fun waitUntilCompleted() {
        statusVal = MTLCommandBufferStatus.Completed
    }

    companion object {
        fun new(label: String? = null): CommandBuffer = CommandBuffer(label)
    }
}
