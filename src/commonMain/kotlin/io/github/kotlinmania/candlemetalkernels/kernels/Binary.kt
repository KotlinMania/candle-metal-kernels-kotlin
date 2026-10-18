// port-lint: source kernels/binary.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.getTileSize
import io.github.kotlinmania.candlemetalkernels.linearSplit
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.setParam

object BinaryContiguous {
    val badd: OpKernelGroup = OpKernelGroup("badd")
    val bsub: OpKernelGroup = OpKernelGroup("bsub")
    val bmul: OpKernelGroup = OpKernelGroup("bmul")
    val bdiv: OpKernelGroup = OpKernelGroup("bdiv")
    val bminimum: OpKernelGroup = OpKernelGroup("bminimum")
    val bmaximum: OpKernelGroup = OpKernelGroup("bmaximum")
    val eq: OpKernelGroup = OpKernelGroup("eq")
    val ne: OpKernelGroup = OpKernelGroup("ne")
    val le: OpKernelGroup = OpKernelGroup("le")
    val lt: OpKernelGroup = OpKernelGroup("lt")
    val ge: OpKernelGroup = OpKernelGroup("ge")
    val gt: OpKernelGroup = OpKernelGroup("gt")
    val copy: OpKernelGroup = OpKernelGroup("copy")
}

object BinaryStrided {
    val badd: OpKernelGroup = OpKernelGroup("badd", "_strided")
    val bsub: OpKernelGroup = OpKernelGroup("bsub", "_strided")
    val bmul: OpKernelGroup = OpKernelGroup("bmul", "_strided")
    val bdiv: OpKernelGroup = OpKernelGroup("bdiv", "_strided")
    val bminimum: OpKernelGroup = OpKernelGroup("bminimum", "_strided")
    val bmaximum: OpKernelGroup = OpKernelGroup("bmaximum", "_strided")
    val eq: OpKernelGroup = OpKernelGroup("eq", "_strided")
    val ne: OpKernelGroup = OpKernelGroup("ne", "_strided")
    val le: OpKernelGroup = OpKernelGroup("le", "_strided")
    val lt: OpKernelGroup = OpKernelGroup("lt", "_strided")
    val ge: OpKernelGroup = OpKernelGroup("ge", "_strided")
    val gt: OpKernelGroup = OpKernelGroup("gt", "_strided")
    val copy: OpKernelGroup = OpKernelGroup("copy", "_strided")
}

fun callBinaryContiguous(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    dtypeSize: Long,
    length: Long,
    left: BufferOffset,
    right: BufferOffset,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Binary, kernelName)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, length)
    setParam(encoder, 1L, left)
    setParam(encoder, 2L, right)
    setParam(encoder, 3L, Output.new(output))

    val tileSize = getTileSize(dtypeSize)
    val tiles = if (tileSize > 0L) (length + tileSize - 1L) / tileSize else length
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, tiles)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callBinaryStrided(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    dtypeSize: Long,
    shape: List<Long>,
    leftInput: BufferOffset,
    leftStrides: List<Long>,
    rightInput: BufferOffset,
    rightStrides: List<Long>,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Binary, kernelName)
    val numDims = shape.size.toLong()
    val encoder = ep.encoder()
    val length = shape.fold(1L) { acc, d -> acc * d }
    val tileSize = getTileSize(dtypeSize)
    val tiles = if (tileSize > 0L) (length + tileSize - 1L) / tileSize else length
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, tiles)

    encoder.setComputePipelineState(pipeline)
    setParam(encoder, 0L, length)
    setParam(encoder, 1L, numDims)
    setParam(encoder, 2L, shape)
    setParam(encoder, 3L, leftStrides)
    setParam(encoder, 4L, rightStrides)
    setParam(encoder, 5L, leftInput)
    setParam(encoder, 6L, rightInput)
    setParam(encoder, 7L, Output.new(output))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}
