// port-lint: source kernels/affine.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.KernelName
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.getBlockDims
import io.github.kotlinmania.candlemetalkernels.getTileSize
import io.github.kotlinmania.candlemetalkernels.linearSplit
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.ComputeCommandEncoder
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.setParam

fun callAffine(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    dtypeSize: Long,
    size: Long,
    input: BufferOffset,
    output: Buffer,
    mul: Float,
    add: Float,
) {
    val pipeline = kernels.loadPipeline(device, Source.Affine, name)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, size)
    setParam(encoder, 1L, mul)
    setParam(encoder, 2L, add)
    setParam(encoder, 3L, input)
    setParam(encoder, 4L, Output.new(output))

    val tileSize = getTileSize(dtypeSize)
    val tiles = if (tileSize > 0L) (size + tileSize - 1L) / tileSize else size
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, tiles)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callAffineStrided(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    input: BufferOffset,
    inputStride: List<Long>,
    output: Buffer,
    mul: Float,
    add: Float,
) {
    val pipeline = kernels.loadPipeline(device, Source.Affine, name)
    val size = shape.fold(1L) { acc, d -> acc * d }

    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, size)
    setParam(encoder, 1L, shape.size.toLong())
    setParam(encoder, 2L, shape)
    setParam(encoder, 3L, inputStride)
    setParam(encoder, 4L, mul)
    setParam(encoder, 5L, add)
    setParam(encoder, 6L, input)
    setParam(encoder, 7L, Output.new(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, size)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callPowf(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    dtypeSize: Long,
    size: Long,
    input: BufferOffset,
    output: Buffer,
    mul: Float,
) {
    val pipeline = kernels.loadPipeline(device, Source.Affine, name)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, size)
    setParam(encoder, 1L, mul)
    setParam(encoder, 2L, input)
    setParam(encoder, 3L, Output.new(output))

    val tileSize = getTileSize(dtypeSize)
    val tiles = if (tileSize > 0L) (size + tileSize - 1L) / tileSize else size
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, tiles)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callPowfStrided(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    input: BufferOffset,
    inputStride: List<Long>,
    output: Buffer,
    mul: Float,
) {
    val pipeline = kernels.loadPipeline(device, Source.Affine, name)
    val size = shape.fold(1L) { acc, d -> acc * d }

    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, size)
    setParam(encoder, 1L, shape.size.toLong())
    setParam(encoder, 2L, shape)
    setParam(encoder, 3L, inputStride)
    setParam(encoder, 4L, mul)
    setParam(encoder, 5L, input)
    setParam(encoder, 6L, Output.new(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, size)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callElu(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    dtypeSize: Long,
    size: Long,
    input: BufferOffset,
    output: Buffer,
    mul: Float,
) {
    val pipeline = kernels.loadPipeline(device, Source.Affine, name)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, size)
    setParam(encoder, 1L, mul)
    setParam(encoder, 2L, input)
    setParam(encoder, 3L, Output.new(output))

    val tileSize = getTileSize(dtypeSize)
    val tiles = if (tileSize > 0L) (size + tileSize - 1L) / tileSize else size
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, tiles)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callEluStrided(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    input: BufferOffset,
    inputStride: List<Long>,
    output: Buffer,
    mul: Float,
) {
    val pipeline = kernels.loadPipeline(device, Source.Affine, name)
    val size = shape.fold(1L) { acc, d -> acc * d }

    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, size)
    setParam(encoder, 1L, shape.size.toLong())
    setParam(encoder, 2L, shape)
    setParam(encoder, 3L, inputStride)
    setParam(encoder, 4L, mul)
    setParam(encoder, 5L, input)
    setParam(encoder, 6L, Output.new(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, size)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}
