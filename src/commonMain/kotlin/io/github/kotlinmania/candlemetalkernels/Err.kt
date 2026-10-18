// port-lint: source err.rs
package io.github.kotlinmania.candlemetalkernels

import io.github.kotlinmania.candlemetalkernels.kernels.SdpaDType

sealed class MetalKernelError(message: String? = null, cause: Throwable? = null) : Exception(message, cause) {
    data class CommandBufferError(val error: String) : MetalKernelError("Command buffer had following error: $error")
    data class LockError(val error: String) : MetalKernelError("Could not lock resource: $error")
    data class LoadLibraryError(val error: String) : MetalKernelError("Error while loading library: $error")
    data class LoadFunctionError(val error: String) : MetalKernelError("Error while loading function: $error")
    data class UnsupportedDTypeForOp(val dtype: String, val op: String) : MetalKernelError("Unsupported dtype $dtype for operation $op")
    data object FailedToCreateComputeFunction : MetalKernelError("Failed to create compute function")
    data class InvalidInput(val error: String) : MetalKernelError("Invalid input: $error")
    data class FailedToCreateResource(val resource: String) : MetalKernelError("Failed to create metal resource: $resource")
    data class FailedToCreatePipeline(val error: String) : MetalKernelError("Failed to create pipeline: $error")
    data class MatMulNonContiguous(
        val lhsStride: List<Long>,
        val rhsStride: List<Long>,
        val mnk: Triple<Long, Long, Long>,
    ) : MetalKernelError("Invalid matmul arguments $lhsStride $rhsStride $mnk")
    data class SdpaHeadSizeMismatch(
        val variation: String,
        val got: Long,
        val expected: List<Long>,
    ) : MetalKernelError("Sdpa $variation head size was $got, expectd $expected")
    data class SdpaHeadDTypeMismatch(
        val variation: String,
        val got: SdpaDType,
    ) : MetalKernelError("Sdpa $variation got dtype $got")
    data class WithBacktrace(
        val inner: MetalKernelError,
        val backtrace: String,
    ) : MetalKernelError("${inner.message}\n$backtrace", inner)

    fun bt(): MetalKernelError = this

    companion object {
        fun from(e: Throwable): MetalKernelError = LockError(e.message ?: "Lock error")
    }
}
