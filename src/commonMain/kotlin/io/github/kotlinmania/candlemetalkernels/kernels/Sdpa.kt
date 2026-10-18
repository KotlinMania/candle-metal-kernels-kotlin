// port-lint: source kernels/sdpa.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.EncoderParam
import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.MetalKernelError
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.ComputeCommandEncoder
import io.github.kotlinmania.candlemetalkernels.metal.ConstantValue
import io.github.kotlinmania.candlemetalkernels.metal.ConstantValues
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.metal.MTLSize
import io.github.kotlinmania.candlemetalkernels.metal.Value
import io.github.kotlinmania.candlemetalkernels.setParam

enum class SdpaDType {
    BF16,
    F16,
    F32,
}

data class AttnParams(
    val b: Int,
    val h: Int,
    val d: Int,
    val ql: Int,
    val kl: Int,
    val gqaFactor: Int,
    val scale: Float,
    val softcapping: Float,
    val nq: Int,
    val nk: Int,
    val nqAligned: Int,
    val nkAligned: Int,
    val qlRem: Int,
    val klRem: Int,
    val qlOff: Int,
    val qStrides: List<Long>,
    val kStrides: List<Long>,
    val vStrides: List<Long>,
    val oStrides: List<Long>,
) : EncoderParam {
    override fun setParam(encoder: ComputeCommandEncoder, position: Long) {
        encoder.setBytes(position, this)
    }
}

data class AttnMaskParams(
    val mStrides: List<Long>,
) : EncoderParam {
    override fun setParam(encoder: ComputeCommandEncoder, position: Long) {
        encoder.setBytes(position, this)
    }
}

fun callSdpaFull(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    qOffset: Long,
    qShape: List<Long>,
    qStrides: List<Long>,
    qBuffer: Buffer,
    kOffset: Long,
    kShape: List<Long>,
    kStrides: List<Long>,
    kBuffer: Buffer,
    vOffset: Long,
    vBuffer: Buffer,
    vStrides: List<Long>,
    maskType: SdpaDType?,
    maskBuffer: Buffer?,
    mStrides: List<Long>?,
    output: Buffer,
    oStrides: List<Long>,
    scale: Float,
    doCausal: Boolean,
    itype: SdpaDType,
) {
    val bd = qShape[qShape.size - 1]
    if (bd !in listOf(32L, 64L, 72L, 80L, 96L, 128L, 256L, 512L)) {
        throw MetalKernelError.SdpaHeadSizeMismatch(
            variation = "full",
            got = bd,
            expected = listOf(32L, 64L, 72L, 80L, 96L, 128L, 256L, 512L),
        )
    }

    val config =
        if (bd == 512L) {
            if (itype == SdpaDType.F32) {
                throw MetalKernelError.SdpaHeadSizeMismatch(
                    variation = "full (f32 unsupported at head_dim=512)",
                    got = bd,
                    expected = listOf(32L, 64L, 72L, 80L, 96L, 128L, 256L),
                )
            }
            listOf(8L, 8L, 1L, 1L)
        } else {
            val bkVal = if (bd < 128L) 32L else 16L
            listOf(32L, bkVal, 4L, 1L)
        }
    val bq = config[0]
    val bk = config[1]
    val wm = config[2]
    val wn = config[3]

    val b = qShape[0]
    val h = qShape[1]
    val d = qShape[3]
    val gqaFactor = (qShape[1] / kShape[1]).toInt()

    val ql = qShape[2]
    val kl = kShape[2]

    val alignQ = (ql % bq) == 0L
    val alignK = (kl % bk) == 0L
    val hasMask = maskBuffer != null

    val effectiveDoCausal = doCausal && !hasMask

    val itypeRepr =
        when (itype) {
            SdpaDType.BF16 -> "bfloat16"
            SdpaDType.F16 -> "float16"
            SdpaDType.F32 -> "float32"
        }
    val maskRepr =
        when (maskType) {
            SdpaDType.BF16 -> "bfloat16"
            SdpaDType.F16 -> "float16"
            SdpaDType.F32 -> "float32"
            null -> itypeRepr
        }
    val name = "steel_attention_${itypeRepr}_bq${bq}_bk${bk}_bd${bd}_wm${wm}_wn${wn}_mask$maskRepr"

    val constants =
        ConstantValues.new(
            listOf(
                ConstantValue(200L, Value.Bool(alignQ)),
                ConstantValue(201L, Value.Bool(alignK)),
                ConstantValue(300L, Value.Bool(hasMask)),
                ConstantValue(301L, Value.Bool(effectiveDoCausal)),
            ),
        )

    val pipeline = kernels.loadPipelineWithConstants(device, Source.Sdpa, name, constants)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("sdpa_full $name B=$b H=$h D=$d QL=$ql KL=$kl")

    val nq = (ql + bq - 1L) / bq
    val nk = (kl + bk - 1L) / bk

    val nqAligned = ql / bq
    val nkAligned = kl / bk

    val params =
        AttnParams(
            b = b.toInt(),
            h = h.toInt(),
            d = d.toInt(),
            ql = ql.toInt(),
            kl = kl.toInt(),
            gqaFactor = gqaFactor,
            scale = scale,
            softcapping = 1.0f,
            nq = nq.toInt(),
            nk = nk.toInt(),
            nqAligned = nqAligned.toInt(),
            nkAligned = nkAligned.toInt(),
            qlRem = (ql - nqAligned * bq).toInt(),
            klRem = (kl - nkAligned * bk).toInt(),
            qlOff = (kl - ql).toInt(),
            qStrides = listOf(qStrides[0], qStrides[1], qStrides[2]),
            kStrides = listOf(kStrides[0], kStrides[1], kStrides[2]),
            vStrides = listOf(vStrides[0], vStrides[1], vStrides[2]),
            oStrides = listOf(oStrides[0], oStrides[1], oStrides[2]),
        )

    if (maskBuffer != null) {
        val maskStrides = mStrides!!
        val maskParams =
            AttnMaskParams(
                listOf(maskStrides[0], maskStrides[1], maskStrides[2]),
            )
        setParam(encoder, 0L, BufferOffset(qBuffer, qOffset))
        setParam(encoder, 1L, BufferOffset(kBuffer, kOffset))
        setParam(encoder, 2L, BufferOffset(vBuffer, vOffset))
        setParam(encoder, 3L, Output.new(output))
        setParam(encoder, 4L, params)
        setParam(encoder, 5L, maskParams)
        setParam(encoder, 6L, maskBuffer)
    } else {
        setParam(encoder, 0L, BufferOffset(qBuffer, qOffset))
        setParam(encoder, 1L, BufferOffset(kBuffer, kOffset))
        setParam(encoder, 2L, BufferOffset(vBuffer, vOffset))
        setParam(encoder, 3L, Output.new(output))
        setParam(encoder, 4L, params)
    }

    val gridDims = MTLSize(nq, h, b)
    val groupDims = MTLSize(32L, wm, wn)
    encoder.dispatchThreadGroups(gridDims, groupDims)
}

fun callSdpaVector(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    qOffset: Long,
    qShape: List<Long>,
    qBuffer: Buffer,
    kOffset: Long,
    kShape: List<Long>,
    kStride: List<Long>,
    kBuffer: Buffer,
    vOffset: Long,
    vStride: List<Long>,
    vBuffer: Buffer,
    output: Buffer,
    alpha: Float,
    softcapping: Float,
    itype: SdpaDType,
) {
    val bk = qShape.last()

    val gqaFactor = (qShape[1] / kShape[1]).toInt()
    val n = kShape[2].toInt()
    val b = (qShape[0] * qShape[1]).toInt()
    val kstride = kStride[1]
    val vstride = vStride[1]

    val name =
        when (Pair(bk, itype)) {
            Pair(32L, SdpaDType.F16) -> "sdpa_vector_float16_t_32"

            Pair(64L, SdpaDType.F16) -> "sdpa_vector_float16_t_64"

            Pair(96L, SdpaDType.F16) -> "sdpa_vector_float16_t_96"

            Pair(128L, SdpaDType.F16) -> "sdpa_vector_float16_t_128"

            Pair(256L, SdpaDType.F16) -> "sdpa_vector_float16_t_256"

            Pair(512L, SdpaDType.F16) -> "sdpa_vector_float16_t_512"

            Pair(32L, SdpaDType.BF16) -> "sdpa_vector_bfloat16_t_32"

            Pair(64L, SdpaDType.BF16) -> "sdpa_vector_bfloat16_t_64"

            Pair(96L, SdpaDType.BF16) -> "sdpa_vector_bfloat16_t_96"

            Pair(128L, SdpaDType.BF16) -> "sdpa_vector_bfloat16_t_128"

            Pair(256L, SdpaDType.BF16) -> "sdpa_vector_bfloat16_t_256"

            Pair(512L, SdpaDType.BF16) -> "sdpa_vector_bfloat16_t_512"

            Pair(32L, SdpaDType.F32) -> "sdpa_vector_float_32"

            Pair(64L, SdpaDType.F32) -> "sdpa_vector_float_64"

            Pair(96L, SdpaDType.F32) -> "sdpa_vector_float_96"

            Pair(128L, SdpaDType.F32) -> "sdpa_vector_float_128"

            Pair(256L, SdpaDType.F32) -> "sdpa_vector_float_256"

            Pair(512L, SdpaDType.F32) -> "sdpa_vector_float_512"

            else -> throw MetalKernelError.SdpaHeadSizeMismatch(
                variation = "vector",
                got = bk,
                expected = listOf(32L, 64L, 96L, 128L, 256L, 512L),
            )
        }

    val effectiveAlpha =
        if (softcapping != 1.0f) {
            alpha / softcapping
        } else {
            alpha
        }

    val constants =
        ConstantValues.new(
            listOf(
                ConstantValue(20L, Value.Bool(false)),
            ),
        )

    val pipeline = kernels.loadPipelineWithConstants(device, Source.Sdpa, name, constants)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("sdpa_vector bk=$bk B=$b N=$n")

    setParam(encoder, 0L, BufferOffset(qBuffer, qOffset))
    setParam(encoder, 1L, BufferOffset(kBuffer, kOffset))
    setParam(encoder, 2L, BufferOffset(vBuffer, vOffset))
    setParam(encoder, 3L, Output.new(output))
    setParam(encoder, 4L, gqaFactor)
    setParam(encoder, 5L, n)
    setParam(encoder, 6L, kstride)
    setParam(encoder, 7L, vstride)
    setParam(encoder, 8L, effectiveAlpha)
    setParam(encoder, 9L, softcapping)

    val gridDims = MTLSize(1L, b.toLong(), 1L)
    val groupDims = MTLSize(1024L, 1L, 1L)
    encoder.dispatchThreadGroups(gridDims, groupDims)
}

const val SDPA_2PASS_BLOCKS: Long = 32L

fun callSdpaVector2pass(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    qOffset: Long,
    qShape: List<Long>,
    qBuffer: Buffer,
    kOffset: Long,
    kShape: List<Long>,
    kStride: List<Long>,
    kBuffer: Buffer,
    vOffset: Long,
    vStride: List<Long>,
    vBuffer: Buffer,
    output: Buffer,
    intermediate: Buffer,
    sums: Buffer,
    maxs: Buffer,
    alpha: Float,
    softcapping: Float,
    itype: SdpaDType,
) {
    val bk = qShape.last()

    // First pass
    run {
        val namePass1 =
            when (Pair(bk, itype)) {
                Pair(32L, SdpaDType.F16) -> "sdpa_vector_2pass_1_float16_t_32"

                Pair(64L, SdpaDType.F16) -> "sdpa_vector_2pass_1_float16_t_64"

                Pair(96L, SdpaDType.F16) -> "sdpa_vector_2pass_1_float16_t_96"

                Pair(128L, SdpaDType.F16) -> "sdpa_vector_2pass_1_float16_t_128"

                Pair(256L, SdpaDType.F16) -> "sdpa_vector_2pass_1_float16_t_256"

                Pair(512L, SdpaDType.F16) -> "sdpa_vector_2pass_1_float16_t_512"

                Pair(32L, SdpaDType.BF16) -> "sdpa_vector_2pass_1_bfloat16_t_32"

                Pair(64L, SdpaDType.BF16) -> "sdpa_vector_2pass_1_bfloat16_t_64"

                Pair(96L, SdpaDType.BF16) -> "sdpa_vector_2pass_1_bfloat16_t_96"

                Pair(128L, SdpaDType.BF16) -> "sdpa_vector_2pass_1_bfloat16_t_128"

                Pair(256L, SdpaDType.BF16) -> "sdpa_vector_2pass_1_bfloat16_t_256"

                Pair(512L, SdpaDType.BF16) -> "sdpa_vector_2pass_1_bfloat16_t_512"

                Pair(32L, SdpaDType.F32) -> "sdpa_vector_2pass_1_float_32"

                Pair(64L, SdpaDType.F32) -> "sdpa_vector_2pass_1_float_64"

                Pair(96L, SdpaDType.F32) -> "sdpa_vector_2pass_1_float_96"

                Pair(128L, SdpaDType.F32) -> "sdpa_vector_2pass_1_float_128"

                Pair(256L, SdpaDType.F32) -> "sdpa_vector_2pass_1_float_256"

                Pair(512L, SdpaDType.F32) -> "sdpa_vector_2pass_1_float_512"

                else -> throw MetalKernelError.SdpaHeadSizeMismatch(
                    variation = "vector_2pass_1",
                    got = bk,
                    expected = listOf(32L, 64L, 96L, 128L, 256L, 512L),
                )
            }

        val gqaFactor = (qShape[1] / kShape[1]).toInt()
        val n = kShape[2].toInt()
        val b = (qShape[0] * qShape[1]).toInt()
        val kstride = kStride[1]
        val vstride = vStride[1]

        val effectiveAlpha =
            if (softcapping != 1.0f) {
                alpha / softcapping
            } else {
                alpha
            }

        val constants =
            ConstantValues.new(
                listOf(
                    ConstantValue(20L, Value.Bool(false)),
                ),
            )

        val pipeline = kernels.loadPipelineWithConstants(device, Source.Sdpa, namePass1, constants)
        val encoder: ComputeCommandEncoder = ep.encoder().asRef()
        encoder.setComputePipelineState(pipeline)
        encoder.debugGroup("sdpa_vector_2pass pass1 bk=$bk B=$b N=$n")

        setParam(encoder, 0L, BufferOffset(qBuffer, qOffset))
        setParam(encoder, 1L, BufferOffset(kBuffer, kOffset))
        setParam(encoder, 2L, BufferOffset(vBuffer, vOffset))
        setParam(encoder, 3L, Output.new(intermediate))
        setParam(encoder, 4L, Output.new(sums))
        setParam(encoder, 5L, Output.new(maxs))
        setParam(encoder, 6L, gqaFactor)
        setParam(encoder, 7L, n)
        setParam(encoder, 8L, kstride)
        setParam(encoder, 9L, vstride)
        setParam(encoder, 10L, effectiveAlpha)
        setParam(encoder, 11L, softcapping)

        val gridDims = MTLSize(1L, b.toLong(), SDPA_2PASS_BLOCKS)
        val groupDims = MTLSize(8L * 32L, 1L, 1L)

        encoder.dispatchThreadGroups(gridDims, groupDims)
    }

    // Final pass
    run {
        val namePass2 =
            when (Pair(bk, itype)) {
                Pair(32L, SdpaDType.F16) -> "sdpa_vector_2pass_2_float16_t_32"

                Pair(64L, SdpaDType.F16) -> "sdpa_vector_2pass_2_float16_t_64"

                Pair(96L, SdpaDType.F16) -> "sdpa_vector_2pass_2_float16_t_96"

                Pair(128L, SdpaDType.F16) -> "sdpa_vector_2pass_2_float16_t_128"

                Pair(256L, SdpaDType.F16) -> "sdpa_vector_2pass_2_float16_t_256"

                Pair(512L, SdpaDType.F16) -> "sdpa_vector_2pass_2_float16_t_512"

                Pair(32L, SdpaDType.BF16) -> "sdpa_vector_2pass_2_bfloat16_t_32"

                Pair(64L, SdpaDType.BF16) -> "sdpa_vector_2pass_2_bfloat16_t_64"

                Pair(96L, SdpaDType.BF16) -> "sdpa_vector_2pass_2_bfloat16_t_96"

                Pair(128L, SdpaDType.BF16) -> "sdpa_vector_2pass_2_bfloat16_t_128"

                Pair(256L, SdpaDType.BF16) -> "sdpa_vector_2pass_2_bfloat16_t_256"

                Pair(512L, SdpaDType.BF16) -> "sdpa_vector_2pass_2_bfloat16_t_512"

                Pair(32L, SdpaDType.F32) -> "sdpa_vector_2pass_2_float_32"

                Pair(64L, SdpaDType.F32) -> "sdpa_vector_2pass_2_float_64"

                Pair(96L, SdpaDType.F32) -> "sdpa_vector_2pass_2_float_96"

                Pair(128L, SdpaDType.F32) -> "sdpa_vector_2pass_2_float_128"

                Pair(256L, SdpaDType.F32) -> "sdpa_vector_2pass_2_float_256"

                Pair(512L, SdpaDType.F32) -> "sdpa_vector_2pass_2_float_512"

                else -> throw MetalKernelError.SdpaHeadSizeMismatch(
                    variation = "vector_2pass_2",
                    got = bk,
                    expected = listOf(32L, 64L, 96L, 128L, 256L, 512L),
                )
            }

        val b = qShape[0] * qShape[1]

        val pipeline = kernels.loadPipeline(device, Source.Sdpa, namePass2)
        val encoder: ComputeCommandEncoder = ep.encoder().asRef()
        encoder.setComputePipelineState(pipeline)
        encoder.debugGroup("sdpa_vector_2pass pass2 bk=$bk B=$b")

        setParam(encoder, 0L, intermediate)
        setParam(encoder, 1L, sums)
        setParam(encoder, 2L, maxs)
        setParam(encoder, 3L, Output.new(output))

        val gridDims = MTLSize(1L, b, 1L)
        val groupDims = MTLSize(1024L, 1L, 1L)

        encoder.dispatchThreadGroups(gridDims, groupDims)
    }
}
