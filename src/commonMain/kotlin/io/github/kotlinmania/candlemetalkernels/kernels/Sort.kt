// port-lint: source kernels/sort.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.DType
import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.RESOURCE_OPTIONS
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.ComputeCommandEncoder
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.metal.MTLSize
import io.github.kotlinmania.candlemetalkernels.setParam

fun callArgSort(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    nrows: Long,
    ncols: Long,
    ncolsPad: Long,
    src: BufferOffset,
    dst: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Sort, name)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("arg_sort $name nrows=$nrows ncols=$ncols")

    setParam(encoder, 0L, src)
    setParam(encoder, 1L, Output.new(dst))
    setParam(encoder, 2L, ncols)
    setParam(encoder, 3L, ncolsPad)

    val threadGroupCount = MTLSize(1L, nrows, 1L)
    val threadGroupSize = MTLSize(ncolsPad, 1L, 1L)

    encoder.setThreadgroupMemoryLength(maxOf(ncolsPad * 4L, 16L), 0L)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

private fun mlxDTypeStr(dtype: DType): String = when (dtype) {
    DType.U8 -> "uint8"
    DType.U32 -> "uint32"
    DType.I64 -> "int64"
    DType.F16 -> "float16"
    DType.BF16 -> "bfloat16"
    DType.F32 -> "float32"
}

private fun multiBlockSort(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    dtype: DType,
    bn: Long,
    tn: Long,
    nblocks: Long,
    nrows: Long,
    ncols: Long,
    src: BufferOffset,
    dst: Buffer,
) {
    val dtypeStr = mlxDTypeStr(dtype)
    val elCount = nrows * ncols
    val bytesLen = elCount * dtype.sizeInBytes()
    val devVals0 = device.newBuffer(bytesLen, RESOURCE_OPTIONS)
    devVals0.setLabel("mlx_arg_sort dev_vals_0")
    val devVals1 = device.newBuffer(bytesLen, RESOURCE_OPTIONS)
    devVals1.setLabel("mlx_arg_sort dev_vals_1")
    val devIdxs0 = device.newBuffer(elCount * 4L, RESOURCE_OPTIONS)
    devIdxs0.setLabel("mlx_arg_sort dev_idxs_0")
    val devIdxs1 = device.newBuffer(elCount * 4L, RESOURCE_OPTIONS)
    devIdxs1.setLabel("mlx_arg_sort dev_idxs_1")
    val blockPartitions = device.newBuffer((nrows * (nblocks + 1L)) * 4L, RESOURCE_OPTIONS)
    blockPartitions.setLabel("mlx_arg_sort block_partitions")

    val encoder: ComputeCommandEncoder = ep.encoder().asRef()

    // Do blockwise sort
    run {
        val name = "sort_mbsort_${dtypeStr}_uint32_bn${bn}_tn${tn}"
        val pipeline = kernels.loadPipeline(device, Source.MlxSort, name)
        encoder.setComputePipelineState(pipeline)
        encoder.debugGroup("mlx_arg_sort multi_block block_sort $dtypeStr nrows=$nrows ncols=$ncols blocks=$nblocks")
        setParam(encoder, 0L, src)
        setParam(encoder, 1L, devVals0)
        setParam(encoder, 2L, devIdxs0)
        setParam(encoder, 3L, ncols.toInt())
        setParam(encoder, 4L, 1)
        setParam(encoder, 5L, 1)
        setParam(encoder, 6L, nrows.toInt())
        setParam(encoder, 7L, ncols)

        val threadGroupCount = MTLSize(nblocks, nrows, 1L)
        val threadGroupSize = MTLSize(bn, 1L, 1L)
        encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
    }

    // Do merges
    var ping = false
    var mergeTiles = 2L
    val nThrPerGroup = minOf(nblocks + 1L, 1024L)
    val partitionName = "partition_mbsort_${dtypeStr}_uint32_bn${bn}_tn${tn}"
    val mergeName = "merge_mbsort_float32_uint32_bn${bn}_tn${tn}"
    while (mergeTiles / 2L < nblocks) {
        val (devValsIn, devValsOut) = if (ping) {
            Pair(devVals1, devVals0)
        } else {
            Pair(devVals0, devVals1)
        }
        val (devIdxsIn, devIdxsOut) = if (ping) {
            Pair(devIdxs1, devIdxs0)
        } else {
            Pair(devIdxs0, devIdxs1)
        }
        ping = !ping

        // Do partition
        run {
            val pipeline = kernels.loadPipeline(device, Source.MlxSort, partitionName)
            encoder.setComputePipelineState(pipeline)
            encoder.debugGroup("mlx_arg_sort multi_block partition $dtypeStr merge_tiles=$mergeTiles blocks=$nblocks")
            setParam(encoder, 0L, blockPartitions)
            setParam(encoder, 1L, devValsIn)
            setParam(encoder, 2L, devIdxsIn)
            setParam(encoder, 3L, ncols.toInt())
            setParam(encoder, 4L, mergeTiles.toInt())
            setParam(encoder, 5L, nblocks.toInt())

            val threadGroupCount = MTLSize(1L, nrows, 1L)
            val threadGroupSize = MTLSize(nThrPerGroup, 1L, 1L)
            encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
        }

        // Do merge
        run {
            val pipeline = kernels.loadPipeline(device, Source.MlxSort, mergeName)
            encoder.setComputePipelineState(pipeline)
            encoder.debugGroup("mlx_arg_sort multi_block merge $dtypeStr merge_tiles=$mergeTiles blocks=$nblocks")
            setParam(encoder, 0L, blockPartitions)
            setParam(encoder, 1L, devValsIn)
            setParam(encoder, 2L, devIdxsIn)
            setParam(encoder, 3L, Output.new(devValsOut))
            setParam(encoder, 4L, Output.new(devIdxsOut))
            setParam(encoder, 5L, ncols.toInt())
            setParam(encoder, 6L, mergeTiles.toInt())
            setParam(encoder, 7L, nblocks.toInt())

            val threadGroupCount = MTLSize(nblocks, nrows, 1L)
            val threadGroupSize = MTLSize(bn, 1L, 1L)
            encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
        }

        mergeTiles *= 2L
    }

    val devIdxsOut = if (ping) devIdxs1 else devIdxs0
    val copyKernel = when (dtype) {
        DType.U8 -> Copy2d.U8
        DType.U32 -> Copy2d.U32
        DType.I64 -> Copy2d.I64
        DType.BF16 -> Copy2d.BFLOAT
        DType.F16 -> Copy2d.HALF
        DType.F32 -> Copy2d.FLOAT
    }
    callCopy2d(
        device,
        encoder,
        kernels,
        copyKernel,
        devIdxsOut,
        dst,
        nrows,
        ncols,
        ncols,
        ncols,
        0L,
        0L,
    )
}

private fun blockSort(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    dtype: DType,
    bn: Long,
    tn: Long,
    nrows: Long,
    ncols: Long,
    src: BufferOffset,
    dst: Buffer,
) {
    val dtypeStr = mlxDTypeStr(dtype)
    val name = "carg_block_sort_${dtypeStr}_uint32_bn${bn}_tn${tn}"
    val pipeline = kernels.loadPipeline(device, Source.MlxSort, name)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("mlx_arg_sort block $name nrows=$nrows ncols=$ncols")
    setParam(encoder, 0L, src)
    setParam(encoder, 1L, Output.new(dst))
    setParam(encoder, 2L, ncols.toInt())
    setParam(encoder, 3L, 1)
    setParam(encoder, 4L, 1)
    setParam(encoder, 5L, ncols.toInt())
    setParam(encoder, 6L, ncols.toInt())

    val threadGroupCount = MTLSize(1L, nrows, 1L)
    val threadGroupSize = MTLSize(bn, 1L, 1L)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callMlxArgSort(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    dtype: DType,
    nrows: Long,
    ncols: Long,
    src: BufferOffset,
    dst: Buffer,
) {
    val tn = 8L
    val divCeil = (ncols + tn - 1L) / tn
    val bn = when {
        divCeil >= 257L && dtype.sizeInBytes() <= 4L -> 512L
        divCeil >= 129L -> 256L
        else -> 128L
    }
    val nPerBlock = bn * tn
    val nBlocks = (ncols + nPerBlock - 1L) / nPerBlock
    if (nBlocks > 1L) {
        multiBlockSort(
            device,
            ep,
            kernels,
            dtype,
            bn,
            tn,
            nBlocks,
            nrows,
            ncols,
            src,
            dst,
        )
    } else {
        blockSort(
            device,
            ep,
            kernels,
            dtype,
            bn,
            tn,
            nrows,
            ncols,
            src,
            dst,
        )
    }
}
