// port-lint: source kernels/random.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.MetalKernelError
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.linearSplit
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.ComputeCommandEncoder
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.setParam

fun callRandomUniform(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    min: Float,
    max: Float,
    length: Long,
    seed: Buffer,
    buffer: Buffer,
) {
    if (min >= max) {
        MetalKernelError.LoadLibraryError("min must be less than max").raise()
    }
    val pipeline = kernels.loadPipeline(device, Source.Random, name)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()

    val odd = if (length % 2L != 0L) 1L else 0L
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, length / 2L + odd)

    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("rand_uniform $name elems=$length")

    setParam(encoder, 0L, length)
    setParam(encoder, 1L, min)
    setParam(encoder, 2L, max)
    setParam(encoder, 3L, Output.new(seed))
    setParam(encoder, 4L, Output.new(buffer))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callRandomNormal(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    mean: Float,
    stddev: Float,
    length: Long,
    seed: Buffer,
    buffer: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Random, name)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()

    val odd = if (length % 2L != 0L) 1L else 0L
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, length / 2L + odd)

    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("rand_normal $name elems=$length")

    setParam(encoder, 0L, length)
    setParam(encoder, 1L, mean)
    setParam(encoder, 2L, stddev)
    setParam(encoder, 3L, Output.new(seed))
    setParam(encoder, 4L, Output.new(buffer))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}
