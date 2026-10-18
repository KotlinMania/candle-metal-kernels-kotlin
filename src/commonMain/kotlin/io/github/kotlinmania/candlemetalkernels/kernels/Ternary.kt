// port-lint: source kernels/ternary.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.getTileSize
import io.github.kotlinmania.candlemetalkernels.linearSplit
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.ComputeCommandEncoder
import io.github.kotlinmania.candlemetalkernels.metal.ConstantValues
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.metal.Value
import io.github.kotlinmania.candlemetalkernels.setParam

fun callWhereCond(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    dtypeSize: Long,
    shape: List<Long>,
    cond: BufferOffset,
    condStride: List<Long>,
    condIsContiguous: Boolean,
    left: BufferOffset,
    leftStride: List<Long>,
    leftIsContiguous: Boolean,
    right: BufferOffset,
    rightStride: List<Long>,
    rightIsContiguous: Boolean,
    output: Buffer,
) {
    val constants = ConstantValues.new(
        listOf(
            Pair(0L, Value.Bool(condIsContiguous)),
            Pair(1L, Value.Bool(leftIsContiguous)),
            Pair(2L, Value.Bool(rightIsContiguous)),
        )
    )
    val pipeline = kernels.loadPipelineWithConstants(device, Source.Ternary, name, constants)

    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)

    val size: Long = shape.fold(1L) { acc, d -> acc * d }
    val rank = shape.size.toLong()
    encoder.debugGroup("where $name elems=$size")

    setParam(encoder, 0L, size)
    setParam(encoder, 1L, rank)
    setParam(encoder, 2L, shape)
    setParam(encoder, 3L, condStride)
    setParam(encoder, 4L, leftStride)
    setParam(encoder, 5L, rightStride)
    setParam(encoder, 6L, cond)
    setParam(encoder, 7L, left)
    setParam(encoder, 8L, right)
    setParam(encoder, 9L, Output.new(output))

    val tileSize = getTileSize(dtypeSize)
    val tiles = if (tileSize > 0L) (size + tileSize - 1L) / tileSize else size
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, tiles)

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}
