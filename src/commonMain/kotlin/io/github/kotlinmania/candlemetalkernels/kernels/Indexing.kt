// port-lint: source kernels/indexing.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.linearSplit
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.setParam

fun callIndexSelect(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    idsSize: Long,
    dim: Long,
    contiguous: Boolean,
    srcDims: List<Long>,
    srcStrides: List<Long>,
    input: BufferOffset,
    ids: BufferOffset,
    output: Buffer,
) {
    val d = dim.toInt()
    val leftSize = shape.subList(0, d).fold(1L) { acc, v -> acc * v }
    val rightSize = shape.subList(d + 1, shape.size).fold(1L) { acc, v -> acc * v }
    val srcDimSize = shape[d]
    val dstEl = idsSize * leftSize * rightSize

    val pipeline = kernels.loadPipeline(device, Source.Indexing, name)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, dstEl)
    setParam(encoder, 1L, leftSize)
    setParam(encoder, 2L, srcDimSize)
    setParam(encoder, 3L, rightSize)
    setParam(encoder, 4L, idsSize)
    setParam(encoder, 5L, contiguous)
    setParam(encoder, 6L, srcDims)
    setParam(encoder, 7L, srcStrides)
    setParam(encoder, 8L, input)
    setParam(encoder, 9L, ids)
    setParam(encoder, 10L, Output.new(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callGather(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    idsSize: Long,
    dim: Long,
    input: BufferOffset,
    ids: BufferOffset,
    output: Buffer,
) {
    val d = dim.toInt()
    val leftSize = shape.subList(0, d).fold(1L) { acc, v -> acc * v }
    val rightSize = shape.subList(d + 1, shape.size).fold(1L) { acc, v -> acc * v }
    val srcDimSize = shape[d]
    val dstEl = idsSize * leftSize * rightSize

    val pipeline = kernels.loadPipeline(device, Source.Indexing, name)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, dstEl)
    setParam(encoder, 1L, leftSize)
    setParam(encoder, 2L, srcDimSize)
    setParam(encoder, 3L, rightSize)
    setParam(encoder, 4L, idsSize)
    setParam(encoder, 5L, input)
    setParam(encoder, 6L, ids)
    setParam(encoder, 7L, Output.new(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callScatter(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    srcShape: List<Long>,
    dstShape: List<Long>,
    dim: Long,
    input: BufferOffset,
    ids: BufferOffset,
    output: BufferOffset,
) {
    val d = dim.toInt()
    val leftSize = srcShape.subList(0, d).fold(1L) { acc, v -> acc * v }
    val rightSize = srcShape.subList(d + 1, srcShape.size).fold(1L) { acc, v -> acc * v }
    val srcDimSize = srcShape[d]
    val dstEl = leftSize * rightSize
    val dstDimSize = dstShape[d]

    val pipeline = kernels.loadPipeline(device, Source.Indexing, name)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, dstEl)
    setParam(encoder, 1L, leftSize)
    setParam(encoder, 2L, srcDimSize)
    setParam(encoder, 3L, rightSize)
    setParam(encoder, 4L, dstDimSize)
    setParam(encoder, 5L, input)
    setParam(encoder, 6L, ids)
    setParam(encoder, 7L, Output.fromBufferOffset(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callIndexAdd(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    srcShape: List<Long>,
    dstShape: List<Long>,
    idsShape: List<Long>,
    dim: Long,
    input: BufferOffset,
    ids: BufferOffset,
    output: Buffer,
) {
    val d = dim.toInt()
    val leftSize = srcShape.subList(0, d).fold(1L) { acc, v -> acc * v }
    val rightSize = srcShape.subList(d + 1, srcShape.size).fold(1L) { acc, v -> acc * v }
    val srcDimSize = srcShape[d]
    val dstEl = leftSize * rightSize
    val dstDimSize = dstShape[d]
    val idsDimSize = idsShape[0]

    val pipeline = kernels.loadPipeline(device, Source.Indexing, name)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, dstEl)
    setParam(encoder, 1L, leftSize)
    setParam(encoder, 2L, srcDimSize)
    setParam(encoder, 3L, rightSize)
    setParam(encoder, 4L, dstDimSize)
    setParam(encoder, 5L, idsDimSize)
    setParam(encoder, 6L, input)
    setParam(encoder, 7L, ids)
    setParam(encoder, 8L, Output.new(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}
