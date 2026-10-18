// port-lint: source kernels/cast.rs
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

fun callCastContiguous(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    dtypeSize: Long,
    length: Long,
    input: BufferOffset,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Cast, kernelName)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, length)
    setParam(encoder, 1L, input)
    setParam(encoder, 2L, Output.new(output))

    val tileSize = getTileSize(dtypeSize)
    val tiles = if (tileSize > 0L) (length + tileSize - 1L) / tileSize else length
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, tiles)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callCastStrided(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    shape: List<Long>,
    input: BufferOffset,
    inputStrides: List<Long>,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Cast, kernelName)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    val length = shape.fold(1L) { acc, d -> acc * d }
    setParam(encoder, 0L, length)
    setParam(encoder, 1L, shape.size.toLong())
    setParam(encoder, 2L, shape)
    setParam(encoder, 3L, inputStrides)
    setParam(encoder, 4L, input)
    setParam(encoder, 5L, Output.new(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, length)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}
