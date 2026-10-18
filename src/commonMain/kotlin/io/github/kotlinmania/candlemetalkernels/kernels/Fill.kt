// port-lint: source kernels/fill.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.linearSplit
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.setParam

fun callConstFill(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    length: Long,
    output: Buffer,
    v: Any?,
) {
    val pipeline = kernels.loadPipeline(device, Source.Fill, name)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, Output.new(output))
    setParam(encoder, 1L, v)
    setParam(encoder, 2L, length)

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, length)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}
