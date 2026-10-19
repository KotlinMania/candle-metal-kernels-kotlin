// port-lint: source err.rs
package io.github.kotlinmania.candlemetalkernels

import io.github.kotlinmania.candlemetalkernels.kernels.SdpaDType

sealed class MetalKernelError(
    val message: String? = null,
    val cause: MetalKernelError? = null,
) {
    data class CommandBufferError(
        val error: String,
    ) : MetalKernelError("Command buffer had following error: $error")

    data class LockError(
        val error: String,
    ) : MetalKernelError("Could not lock resource: $error")

    data class LoadLibraryError(
        val error: String,
    ) : MetalKernelError("Error while loading library: $error")

    data class LoadFunctionError(
        val error: String,
    ) : MetalKernelError("Error while loading function: $error")

    data class UnsupportedDTypeForOp(
        val dtype: String,
        val op: String,
    ) : MetalKernelError("Unsupported dtype $dtype for operation $op")

    data object FailedToCreateComputeFunction : MetalKernelError("Failed to create compute function")

    data class InvalidInput(
        val error: String,
    ) : MetalKernelError("Invalid input: $error")

    data class FailedToCreateResource(
        val resource: String,
    ) : MetalKernelError("Failed to create metal resource: $resource")

    data class FailedToCreatePipeline(
        val error: String,
    ) : MetalKernelError("Failed to create pipeline: $error")

    data class Mnk(
        val m: Long,
        val n: Long,
        val k: Long,
    )

    data class MatMulNonContiguous(
        val lhsStride: List<Long>,
        val rhsStride: List<Long>,
        val mnk: Mnk,
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

    fun raise(): Nothing = throw MetalKernelException(this)

    companion object {
        fun fromException(exception: Any?): MetalKernelError? = (exception as? MetalKernelException)?.error

        fun from(message: String): MetalKernelError = LockError(message)

        internal fun from(e: Throwable): MetalKernelError = from(e.message ?: "Lock error")
    }
}

internal class MetalKernelException(
    val error: MetalKernelError,
) : Exception(error.message, error.cause?.let { MetalKernelException(it) })
