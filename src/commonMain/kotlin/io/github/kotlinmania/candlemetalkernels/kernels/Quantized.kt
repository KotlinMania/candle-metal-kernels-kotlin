// port-lint: source kernels/quantized.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.MetalKernelError
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.ComputeCommandEncoder
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.metal.MTLSize
import io.github.kotlinmania.candlemetalkernels.setParam

enum class GgmlDType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    F16,
    F32,
    BF16,
}

private fun divide(m: Long, b: Long): Long = if (b > 0L) (m + b - 1L) / b else m

fun callQuantizedMatmulMvT(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    dtype: GgmlDType,
    b: Long,
    m: Long,
    n: Long,
    k: Long,
    lhs: Buffer,
    lhsOffset: Long,
    rhs: Buffer,
    dstOffset: Long,
    dst: Buffer,
) {
    val ne00 = k
    val ne01 = n
    val ne02 = b
    val ne03 = 1L

    val nb00 = 0L
    val nb01 = 0L
    val nb02 = 0L

    val ne10 = k
    val ne11 = m
    val ne12 = b
    val ne13 = 1L

    val nb10 = 0L
    val nb11 = 0L
    val nb12 = 0L

    val ne0 = n
    val ne1 = m
    val r2 = (ne12 / ne02).toUInt()
    val r3 = (ne13 / ne03).toUInt()

    val (nth0, nth1, align) =
        when (dtype) {
            GgmlDType.Q4_0,
            GgmlDType.Q4_1,
            GgmlDType.Q5_0,
            GgmlDType.Q5_1,
            GgmlDType.Q8_0,
            GgmlDType.Q8_1,
            -> Triple(8L, 8L, 8L)

            GgmlDType.Q2K -> Triple(2L, 32L, 4L)

            GgmlDType.Q4K -> Triple(4L, 8L, 4L)

            GgmlDType.Q3K, GgmlDType.Q5K -> Triple(2L, 32L, 4L)

            GgmlDType.Q6K -> Triple(2L, 32L, 2L)

            GgmlDType.F16, GgmlDType.BF16, GgmlDType.Q8K -> Triple(32L, 1L, 8L)

            GgmlDType.F32 -> Triple(32L, 1L, 8L)
        }

    val threadGroupsCount =
        MTLSize(
            divide(ne01, align),
            ne11,
            ne12 * ne13,
        )
    val threadsPerThreadgroup = MTLSize(nth0, nth1, 1L)

    val name =
        when (dtype) {
            GgmlDType.Q4_0 -> "kernel_mul_mv_q4_0_f32"
            GgmlDType.Q4_1 -> "kernel_mul_mv_q4_1_f32"
            GgmlDType.Q5_0 -> "kernel_mul_mv_q5_0_f32"
            GgmlDType.Q5_1 -> "kernel_mul_mv_q5_1_f32"
            GgmlDType.Q8_0 -> "kernel_mul_mv_q8_0_f32"
            GgmlDType.Q8_1 -> "kernel_mul_mv_q8_1_f32"
            GgmlDType.Q2K -> "kernel_mul_mv_q2_K_f32"
            GgmlDType.Q3K -> "kernel_mul_mv_q3_K_f32"
            GgmlDType.Q4K -> "kernel_mul_mv_q4_K_f32"
            GgmlDType.Q5K -> "kernel_mul_mv_q5_K_f32"
            GgmlDType.Q6K -> "kernel_mul_mv_q6_K_f32"
            GgmlDType.Q8K -> "kernel_mul_mv_q8_K_f32"
            GgmlDType.F16 -> "kernel_mul_mv_f16_f32"
            GgmlDType.BF16 -> "kernel_mul_mv_bf16_f32"
            GgmlDType.F32 -> "kernel_mul_mv_f32_f32"
        }

    val pipeline = kernels.loadPipeline(device, Source.Quantized, name)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("qmm_mv $name B=$b M=$m K=$k N=$n")

    setParam(encoder, 0L, rhs)
    setParam(encoder, 1L, BufferOffset(lhs, lhsOffset))
    setParam(encoder, 2L, Output.withOffset(dst, dstOffset))
    setParam(encoder, 3L, ne00)
    setParam(encoder, 4L, ne01)
    setParam(encoder, 5L, ne02)
    setParam(encoder, 6L, nb00)
    setParam(encoder, 7L, nb01)
    setParam(encoder, 8L, nb02)
    setParam(encoder, 9L, ne10)
    setParam(encoder, 10L, ne11)
    setParam(encoder, 11L, ne12)
    setParam(encoder, 12L, nb10)
    setParam(encoder, 13L, nb11)
    setParam(encoder, 14L, nb12)
    setParam(encoder, 15L, ne0)
    setParam(encoder, 16L, ne1)
    setParam(encoder, 17L, r2)
    setParam(encoder, 18L, r3)

    encoder.dispatchThreadGroups(threadGroupsCount, threadsPerThreadgroup)
}

fun callQuantizedMatmulMvT(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    dtype: GgmlDType,
    bmnk: List<Long>,
    lhs: Buffer,
    lhsOffset: Long,
    rhs: Buffer,
    dstOffset: Long,
    dst: Buffer,
) = callQuantizedMatmulMvT(
    device,
    ep,
    kernels,
    dtype,
    bmnk[0],
    bmnk[1],
    bmnk[2],
    bmnk[3],
    lhs,
    lhsOffset,
    rhs,
    dstOffset,
    dst,
)

fun callQuantizedMatmulMmT(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    dtype: GgmlDType,
    src0Shape: List<Long>,
    src0Stride: List<Long>,
    src0: Buffer,
    src1Shape: List<Long>,
    src1Stride: List<Long>,
    src1: Buffer,
    src1Offset: Long,
    dstShape: List<Long>,
    dstOffset: Long,
    dst: Buffer,
) {
    val ne00 = src0Shape[src0Shape.size - 1]
    val ne01 = src0Shape[src0Shape.size - 2]
    val ne02 = src0Shape[src0Shape.size - 3]
    val ne03 = src0Shape[src0Shape.size - 4]

    val nb01 = src0Stride[src0Stride.size - 2]
    val nb02 = src0Stride[src0Stride.size - 3]
    val nb03 = src0Stride[src0Stride.size - 4]

    val ne11 = src1Shape[src1Shape.size - 2]
    val ne12 = src1Shape[src1Shape.size - 3]
    val ne13 = src1Shape[src1Shape.size - 4]

    val nb10 = src1Stride[src1Stride.size - 1]
    val nb11 = src1Stride[src1Stride.size - 2]
    val nb12 = src1Stride[src1Stride.size - 3]
    val nb13 = src1Stride[src1Stride.size - 4]

    val ne0 = dstShape[dstShape.size - 1]
    val ne1 = dstShape[dstShape.size - 2]
    val r2 = (ne12 / ne02).toUInt()
    val r3 = (ne13 / ne03).toUInt()

    val threadGroupsCount =
        MTLSize(
            divide(ne11, 32L),
            divide(ne01, 64L),
            ne12 * ne13,
        )
    val threadsPerThreadgroup = MTLSize(128L, 1L, 1L)
    val name =
        when (dtype) {
            GgmlDType.Q4_0 -> "kernel_mul_mm_q4_0_f32"
            GgmlDType.Q4_1 -> "kernel_mul_mm_q4_1_f32"
            GgmlDType.Q5_0 -> "kernel_mul_mm_q5_0_f32"
            GgmlDType.Q5_1 -> "kernel_mul_mm_q5_1_f32"
            GgmlDType.Q8_0 -> "kernel_mul_mm_q8_0_f32"
            GgmlDType.Q2K -> "kernel_mul_mm_q2_K_f32"
            GgmlDType.Q3K -> "kernel_mul_mm_q3_K_f32"
            GgmlDType.Q4K -> "kernel_mul_mm_q4_K_f32"
            GgmlDType.Q5K -> "kernel_mul_mm_q5_K_f32"
            GgmlDType.Q6K -> "kernel_mul_mm_q6_K_f32"
            GgmlDType.F16 -> "kernel_mul_mm_f16_f32"
            GgmlDType.BF16 -> "kernel_mul_mm_bf16_f32"
            GgmlDType.F32 -> "kernel_mul_mm_f32_f32"
            GgmlDType.Q8_1 -> throw MetalKernelError.UnsupportedDTypeForOp("Q8_1", "qmatmul")
            GgmlDType.Q8K -> throw MetalKernelError.UnsupportedDTypeForOp("Q8K", "qmatmul")
        }

    val pipeline = kernels.loadPipeline(device, Source.Quantized, name)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("qmm_mm $name M=$ne11 K=$ne00 N=$ne01")

    setParam(encoder, 0L, src0)
    setParam(encoder, 1L, BufferOffset(src1, src1Offset))
    setParam(encoder, 2L, Output.withOffset(dst, dstOffset))
    setParam(encoder, 3L, ne00)
    setParam(encoder, 4L, ne02)
    setParam(encoder, 5L, nb01)
    setParam(encoder, 6L, nb02)
    setParam(encoder, 7L, nb03)
    setParam(encoder, 8L, ne12)
    setParam(encoder, 9L, nb10)
    setParam(encoder, 10L, nb11)
    setParam(encoder, 11L, nb12)
    setParam(encoder, 12L, nb13)
    setParam(encoder, 13L, ne0)
    setParam(encoder, 14L, ne1)
    setParam(encoder, 15L, r2)
    setParam(encoder, 16L, r3)

    encoder.setThreadgroupMemoryLength(8192L, 0L)
    encoder.dispatchThreadGroups(threadGroupsCount, threadsPerThreadgroup)
}
