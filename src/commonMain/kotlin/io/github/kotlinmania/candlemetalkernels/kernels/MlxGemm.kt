// port-lint: source kernels/mlx_gemm.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.EncoderParam
import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Input
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.MetalKernelError
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.ComputeCommandEncoder
import io.github.kotlinmania.candlemetalkernels.metal.ConstantValues
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.metal.MTLSize
import io.github.kotlinmania.candlemetalkernels.metal.MetalDeviceType
import io.github.kotlinmania.candlemetalkernels.metal.Value
import io.github.kotlinmania.candlemetalkernels.setParam

enum class GemmDType {
    BF16,
    F16,
    F32,
}

data class TileConfig(
    val bm: Long,
    val bn: Long,
    val bk: Long,
    val wm: Long,
    val wn: Long,
) {
    companion object {
        fun new(bm: Long, bn: Long, bk: Long, wm: Long, wn: Long): TileConfig =
            TileConfig(bm, bn, bk, wm, wn)
    }
}

data class GemvTileConfig(
    val bm: Long,
    val bn: Long,
    val sm: Long,
    val sn: Long,
    val tm: Long,
    val tn: Long,
)

val TILE_32_32_16_2_2: TileConfig = TileConfig.new(32L, 32L, 16L, 2L, 2L)
val TILE_64_64_16_2_2: TileConfig = TileConfig.new(64L, 64L, 16L, 2L, 2L)
val TILE_64_64_16_1_2: TileConfig = TileConfig.new(64L, 64L, 16L, 1L, 2L)
val TILE_64_32_32_2_2: TileConfig = TileConfig.new(64L, 32L, 32L, 2L, 2L)
val TILE_32_64_16_1_2: TileConfig = TileConfig.new(32L, 64L, 16L, 1L, 2L)

fun selectTileConfig(
    dtype: GemmDType,
    m: Long,
    n: Long,
    k: Long,
    batchSize: Long,
    aTrans: Boolean,
    bTrans: Boolean,
    deviceType: MetalDeviceType,
): TileConfig {
    if (m < 16L) {
        return TILE_32_32_16_2_2
    }

    val totalOutput = batchSize * m * n
    val isLargeMatmul = totalOutput >= (1L shl 20)

    return when (deviceType) {
        MetalDeviceType.Phone, MetalDeviceType.BasePro -> {
            if (!aTrans && bTrans) {
                TILE_64_32_32_2_2
            } else if (dtype != GemmDType.F32) {
                TILE_64_64_16_1_2
            } else {
                TILE_64_64_16_2_2
            }
        }

        MetalDeviceType.Ultra -> {
            if (isLargeMatmul) {
                if (dtype != GemmDType.F32) {
                    if (2L * maxOf(m, n) > k) {
                        TILE_64_64_16_1_2
                    } else if (!aTrans && bTrans) {
                        TILE_64_32_32_2_2
                    } else {
                        TILE_32_64_16_1_2
                    }
                } else {
                    TILE_64_64_16_2_2
                }
            } else {
                if (dtype != GemmDType.F32) {
                    if (!aTrans && bTrans) {
                        TILE_64_32_32_2_2
                    } else {
                        TILE_64_64_16_1_2
                    }
                } else {
                    if (!aTrans && bTrans) {
                        TILE_32_64_16_1_2
                    } else {
                        TILE_64_32_32_2_2
                    }
                }
            }
        }

        MetalDeviceType.Max, MetalDeviceType.Medium -> {
            when (dtype) {
                GemmDType.F32 -> {
                    if (!isLargeMatmul) {
                        if (!aTrans && bTrans) {
                            TILE_32_64_16_1_2
                        } else {
                            TILE_64_32_32_2_2
                        }
                    } else {
                        TILE_64_64_16_2_2
                    }
                }

                GemmDType.F16, GemmDType.BF16 -> {
                    if (isLargeMatmul) {
                        if (2L * maxOf(m, n) > k) {
                            TILE_64_64_16_1_2
                        } else if (!aTrans && bTrans) {
                            TILE_64_32_32_2_2
                        } else {
                            TILE_32_64_16_1_2
                        }
                    } else if (!aTrans && bTrans) {
                        TILE_64_32_32_2_2
                    } else {
                        TILE_64_64_16_1_2
                    }
                }
            }
        }
    }
}

fun checkBatchCollapse(
    b: Long,
    m: Long,
    k: Long,
    aTrans: Boolean,
    lhsStride: List<Long>,
    rhsStride: List<Long>,
): Triple<Long, Long, Boolean> {
    if (b <= 1L) {
        return Triple(b, m, false)
    }
    if (aTrans) {
        return Triple(b, m, false)
    }

    val aBatchStride =
        if (lhsStride.size > 2) {
            lhsStride[lhsStride.size - 3]
        } else {
            m * k
        }

    val bBatchStride =
        if (rhsStride.size > 2) {
            rhsStride[rhsStride.size - 3]
        } else {
            0L
        }

    val aContiguous = aBatchStride == m * k
    val bBroadcasted = bBatchStride == 0L

    return if (aContiguous && bBroadcasted) {
        Triple(1L, b * m, true)
    } else {
        Triple(b, m, false)
    }
}

fun shouldUseSplitK(b: Long, m: Long, n: Long, k: Long): Boolean {
    if (b != 1L) return false
    val tm = m / 16L
    val tn = n / 16L
    val tk = k / 16L
    return (tm * tn) <= 32L && tk >= 8L
}

data class GemmParams(
    val m: Int,
    val n: Int,
    val k: Int,
    val lda: Int,
    val ldb: Int,
    val ldd: Int,
    val tilesN: Int,
    val tilesM: Int,
    val batchStrideA: Long,
    val batchStrideB: Long,
    val batchStrideD: Long,
    val swizzleLog: Int,
    val gemmKIterationsAligned: Int,
    val batchNdim: Int,
) : EncoderParam {
    override fun setParam(encoder: ComputeCommandEncoder, position: Long) {
        encoder.setBytes(position, this)
    }
}

fun callMlxGemv(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    dtype: GemmDType,
    b: Long,
    m: Long,
    n: Long,
    k: Long,
    lhsStride: List<Long>,
    lhsOffset: Long,
    lhsBuffer: Buffer,
    rhsStride: List<Long>,
    rhsOffset: Long,
    rhsBuffer: Buffer,
    output: Buffer,
) {
    require(m == 1L || n == 1L) { "callMlxGemv requires M=1 or N=1" }
    require(rhsStride.size >= 2)
    require(lhsStride.size >= 2)

    val rhsM1 = rhsStride[rhsStride.size - 1]
    val rhsM2 = rhsStride[rhsStride.size - 2]
    val lhsM1 = lhsStride[lhsStride.size - 1]
    val lhsM2 = lhsStride[lhsStride.size - 2]

    val (lda, aTrans) =
        if ((lhsM1 == 1L || k == 1L) && (lhsM2 == k || m == 1L)) {
            Pair(k.toInt(), false)
        } else if ((lhsM1 == m || k == 1L) && (lhsM2 == 1L || m == 1L)) {
            Pair(m.toInt(), true)
        } else {
            MetalKernelError.MatMulNonContiguous(lhsStride, rhsStride, MetalKernelError.Mnk(m, n, k)).bt().raise()
        }

    val (ldb, bTrans) =
        if ((rhsM1 == 1L || n == 1L) && (rhsM2 == n || k == 1L)) {
            Pair(n.toInt(), false)
        } else if ((rhsM1 == k || n == 1L) && (rhsM2 == 1L || k == 1L)) {
            Pair(k.toInt(), true)
        } else {
            MetalKernelError.MatMulNonContiguous(lhsStride, rhsStride, MetalKernelError.Mnk(m, n, k)).bt().raise()
        }

    val isBMatrix = n != 1L
    val transposeMat = if (isBMatrix) !bTrans else aTrans
    val matLd = if (isBMatrix) ldb else lda
    val inVecSize = k
    val outVecSize = if (isBMatrix) n else m

    val matBuffer = if (isBMatrix) rhsBuffer else lhsBuffer
    val matOffset = if (isBMatrix) rhsOffset else lhsOffset
    val vecBuffer = if (isBMatrix) lhsBuffer else rhsBuffer
    val vecOffset = if (isBMatrix) lhsOffset else rhsOffset

    val vecBatchStride: Long =
        if (isBMatrix) {
            if (lhsStride.size > 2) lhsStride[lhsStride.size - 3] else k
        } else {
            if (rhsStride.size > 2) rhsStride[rhsStride.size - 3] else k
        }

    val matBatchStride: Long =
        if (isBMatrix) {
            if (rhsStride.size > 2) rhsStride[rhsStride.size - 3] else 0L
        } else {
            if (lhsStride.size > 2) lhsStride[lhsStride.size - 3] else 0L
        }

    val tileConfig =
        if (transposeMat) {
            val (sm, sn) =
                if (inVecSize >= 8192L && outVecSize >= 2048L) {
                    Pair(4L, 8L)
                } else {
                    Pair(8L, 4L)
                }
            val bn =
                if (outVecSize >= 2048L) {
                    16L
                } else if (outVecSize >= 512L) {
                    4L
                } else {
                    2L
                }
            val tn = if (outVecSize < 4L) 1L else 4L
            GemvTileConfig(1L, bn, sm, sn, 4L, tn)
        } else {
            val tConfig =
                if (inVecSize <= 64L) {
                    listOf(1L, 1L, 8L, 4L)
                } else if (inVecSize >= 16L * outVecSize) {
                    listOf(1L, 8L, 1L, 32L)
                } else if (outVecSize >= 4096L) {
                    listOf(8L, 1L, 1L, 32L)
                } else {
                    listOf(4L, 1L, 1L, 32L)
                }
            val tmVal = if (outVecSize < 4L) 1L else 4L
            GemvTileConfig(tConfig[0], tConfig[1], tConfig[2], tConfig[3], tmVal, 4L)
        }
    val bm = tileConfig.bm
    val bn = tileConfig.bn
    val sm = tileConfig.sm
    val sn = tileConfig.sn
    val tm = tileConfig.tm
    val tn = tileConfig.tn

    val dtypeStr =
        when (dtype) {
            GemmDType.F32 -> "float32"
            GemmDType.F16 -> "float16"
            GemmDType.BF16 -> "bfloat16"
        }
    val kernelPrefix = if (transposeMat) "gemv_t" else "gemv"
    val name = "${kernelPrefix}_${dtypeStr}_bm${bm}_bn${bn}_sm${sm}_sn${sn}_tm${tm}_tn${tn}_nc0_axpby0"

    val pipeline = kernels.loadPipeline(device, Source.Gemv, name)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, Input.withOffset(matBuffer, matOffset))
    setParam(encoder, 1L, Input.withOffset(vecBuffer, vecOffset))
    setParam(encoder, 2L, null)
    setParam(encoder, 3L, Output.new(output))
    setParam(encoder, 4L, inVecSize.toInt())
    setParam(encoder, 5L, outVecSize.toInt())
    setParam(encoder, 6L, matLd)
    setParam(encoder, 7L, 1.0f)
    setParam(encoder, 8L, 0.0f)
    setParam(encoder, 9L, 1)
    setParam(encoder, 10L, listOf(b.toInt()))
    setParam(encoder, 11L, listOf(vecBatchStride))
    setParam(encoder, 12L, listOf(matBatchStride))
    setParam(encoder, 13L, listOf(0L))
    setParam(encoder, 14L, 1)

    val nOutPerTgp = if (transposeMat) bn * sn * tn else bm * sm * tm
    val nTgp = if (nOutPerTgp > 0L) (outVecSize + nOutPerTgp - 1L) / nOutPerTgp else outVecSize
    val gridSize = MTLSize(nTgp, 1L, b)
    val groupSize = MTLSize(32L, bn, bm)
    encoder.dispatchThreadGroups(gridSize, groupSize)
}

fun callMlxGemv(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    dtype: GemmDType,
    bmnk: List<Long>,
    lhsStride: List<Long>,
    lhsOffset: Long,
    lhsBuffer: Buffer,
    rhsStride: List<Long>,
    rhsOffset: Long,
    rhsBuffer: Buffer,
    output: Buffer,
) = callMlxGemv(
    device,
    ep,
    kernels,
    dtype,
    bmnk[0],
    bmnk[1],
    bmnk[2],
    bmnk[3],
    lhsStride,
    lhsOffset,
    lhsBuffer,
    rhsStride,
    rhsOffset,
    rhsBuffer,
    output,
)

fun callMlxGemm(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    dtype: GemmDType,
    b: Long,
    m: Long,
    n: Long,
    k: Long,
    lhsStride: List<Long>,
    lhsOffset: Long,
    lhsBuffer: Buffer,
    rhsStride: List<Long>,
    rhsOffset: Long,
    rhsBuffer: Buffer,
    output: Buffer,
) {
    require(rhsStride.size >= 2)
    require(lhsStride.size >= 2)
    val rhsM1 = rhsStride[rhsStride.size - 1]
    val rhsM2 = rhsStride[rhsStride.size - 2]
    val lhsM1 = lhsStride[lhsStride.size - 1]
    val lhsM2 = lhsStride[lhsStride.size - 2]

    val (lda, aTrans) =
        if ((lhsM1 == 1L || k == 1L) && (lhsM2 == k || m == 1L)) {
            Pair(k.toInt(), false)
        } else if ((lhsM1 == m || k == 1L) && (lhsM2 == 1L || m == 1L)) {
            Pair(m.toInt(), true)
        } else {
            MetalKernelError.MatMulNonContiguous(lhsStride, rhsStride, MetalKernelError.Mnk(m, n, k)).bt().raise()
        }

    val (ldb, bTrans) =
        if ((rhsM1 == 1L || n == 1L) && (rhsM2 == n || k == 1L)) {
            Pair(n.toInt(), false)
        } else if ((rhsM1 == k || n == 1L) && (rhsM2 == 1L || k == 1L)) {
            Pair(k.toInt(), true)
        } else {
            MetalKernelError.MatMulNonContiguous(lhsStride, rhsStride, MetalKernelError.Mnk(m, n, k)).bt().raise()
        }

    if (m == 1L || n == 1L) {
        return callMlxGemv(
            device,
            ep,
            kernels,
            dtype,
            b,
            m,
            n,
            k,
            lhsStride,
            lhsOffset,
            lhsBuffer,
            rhsStride,
            rhsOffset,
            rhsBuffer,
            output,
        )
    }

    val (effectiveBatch, effectiveM, batchCollapsed) =
        checkBatchCollapse(b, m, k, aTrans, lhsStride, rhsStride)

    val effM = effectiveM
    val effB = effectiveBatch

    val deviceType = device.deviceType()
    val tile = selectTileConfig(dtype, effM, n, k, effB, aTrans, bTrans, deviceType)
    val bm = tile.bm
    val bn = tile.bn
    val bk = tile.bk
    val wm = tile.wm
    val wn = tile.wn

    val hasBatch = effB > 1L

    val constants =
        ConstantValues.new(
            listOf(
                io.github.kotlinmania.candlemetalkernels.metal
                    .ConstantValue(10L, Value.Bool(hasBatch)),
                io.github.kotlinmania.candlemetalkernels.metal
                    .ConstantValue(100L, Value.Bool(false)),
                io.github.kotlinmania.candlemetalkernels.metal
                    .ConstantValue(110L, Value.Bool(false)),
                io.github.kotlinmania.candlemetalkernels.metal
                    .ConstantValue(200L, Value.Bool(effM % bm == 0L)),
                io.github.kotlinmania.candlemetalkernels.metal
                    .ConstantValue(201L, Value.Bool(n % bn == 0L)),
                io.github.kotlinmania.candlemetalkernels.metal
                    .ConstantValue(202L, Value.Bool(k % bk == 0L)),
                io.github.kotlinmania.candlemetalkernels.metal
                    .ConstantValue(300L, Value.Bool(false)),
            ),
        )

    val swizzleLog = 0
    val tileSwizzle = 1L shl swizzleLog
    var tn = if (bn > 0L) (n + bn - 1L) / bn else n
    var tm = if (bm > 0L) (effM + bm - 1L) / bm else effM
    tn *= tileSwizzle
    tm = if (tileSwizzle > 0L) (tm + tileSwizzle - 1L) / tileSwizzle else tm

    val (batchStrideA, batchStrideB) =
        if (batchCollapsed) {
            Pair(0L, 0L)
        } else {
            val aStride = if (lhsStride.size > 2) lhsStride[lhsStride.size - 3] else (effM * k)
            val bStride = if (rhsStride.size > 2) rhsStride[rhsStride.size - 3] else (n * k)
            Pair(aStride, bStride)
        }

    val gemmParams =
        GemmParams(
            m = effM.toInt(),
            n = n.toInt(),
            k = k.toInt(),
            lda = if (batchCollapsed) k.toInt() else lda,
            ldb = ldb,
            ldd = n.toInt(),
            tilesN = tn.toInt(),
            tilesM = tm.toInt(),
            batchStrideA = batchStrideA,
            batchStrideB = batchStrideB,
            batchStrideD = effM * n,
            swizzleLog = swizzleLog,
            gemmKIterationsAligned = (k / bk).toInt(),
            batchNdim = 1,
        )

    val dtypeStr =
        when (dtype) {
            GemmDType.F32 -> "f32"
            GemmDType.F16 -> "f16"
            GemmDType.BF16 -> "bf16"
        }
    val transStr =
        when (Pair(aTrans, bTrans)) {
            Pair(false, false) -> "nn"
            Pair(true, false) -> "tn"
            Pair(false, true) -> "nt"
            Pair(true, true) -> "tt"
            else -> "nn"
        }
    val name = "gemm_${transStr}_${dtypeStr}_${dtypeStr}_${bm}_${bn}_${bk}_${wm}_$wn"

    val pipeline = kernels.loadPipelineWithConstants(device, Source.Gemm, name, constants)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("mlx_gemm $name B=$effB M=$effM N=$n K=$k")

    val batchStrides = listOf(batchStrideA, batchStrideB)

    setParam(encoder, 0L, BufferOffset(lhsBuffer, lhsOffset))
    setParam(encoder, 1L, BufferOffset(rhsBuffer, rhsOffset))
    setParam(encoder, 2L, null)
    setParam(encoder, 3L, Output.new(output))
    setParam(encoder, 4L, gemmParams)
    setParam(encoder, 5L, null)
    setParam(encoder, 6L, effB.toInt())
    setParam(encoder, 7L, batchStrides)

    val gridSize = MTLSize(tn, tm, effB)
    val groupSize = MTLSize(32L, wn, wm)
    encoder.dispatchThreadGroups(gridSize, groupSize)
}

fun callMlxGemm(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    dtype: GemmDType,
    bmnk: List<Long>,
    lhsStride: List<Long>,
    lhsOffset: Long,
    lhsBuffer: Buffer,
    rhsStride: List<Long>,
    rhsOffset: Long,
    rhsBuffer: Buffer,
    output: Buffer,
) = callMlxGemm(
    device,
    ep,
    kernels,
    dtype,
    bmnk[0],
    bmnk[1],
    bmnk[2],
    bmnk[3],
    lhsStride,
    lhsOffset,
    lhsBuffer,
    rhsStride,
    rhsOffset,
    rhsBuffer,
    output,
)
