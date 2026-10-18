// port-lint: source kernels/unary.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.ComputeCommandEncoderProvider
import io.github.kotlinmania.candlemetalkernels.EncoderParam
import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.getBlockDims
import io.github.kotlinmania.candlemetalkernels.getTileSize
import io.github.kotlinmania.candlemetalkernels.linearSplit
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.ComputeCommandEncoder
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.metal.MTLSize
import io.github.kotlinmania.candlemetalkernels.setParam

object UnaryContiguous {
    val cos: OpKernelGroup = OpKernelGroup("cos")
    val sin: OpKernelGroup = OpKernelGroup("sin")
    val exp: OpKernelGroup = OpKernelGroup("exp")
    val sqr: OpKernelGroup = OpKernelGroup("sqr")
    val sqrt: OpKernelGroup = OpKernelGroup("sqrt")
    val neg: OpKernelGroup = OpKernelGroup("neg")
    val log: OpKernelGroup = OpKernelGroup("log")
    val gelu: OpKernelGroup = OpKernelGroup("gelu")
    val abs: OpKernelGroup = OpKernelGroup("abs")
    val ceil: OpKernelGroup = OpKernelGroup("ceil")
    val floor: OpKernelGroup = OpKernelGroup("floor")
    val relu: OpKernelGroup = OpKernelGroup("relu")
    val round: OpKernelGroup = OpKernelGroup("round")
    val erf: OpKernelGroup = OpKernelGroup("erf")
    val geluErf: OpKernelGroup = OpKernelGroup("gelu_erf")
    val tanh: OpKernelGroup = OpKernelGroup("tanh")
    val recip: OpKernelGroup = OpKernelGroup("recip")
    val silu: OpKernelGroup = OpKernelGroup("silu")
    val sign: OpKernelGroup = OpKernelGroup("sign")
    val sigmoid: OpKernelGroup = OpKernelGroup("sigmoid")
    val constSet: OpKernelGroup = OpKernelGroup("const_set")
    val copy: OpKernelGroup = OpKernelGroup("copy")
}

object UnaryStrided {
    val cos: OpKernelGroup = OpKernelGroup("cos", "_strided")
    val sin: OpKernelGroup = OpKernelGroup("sin", "_strided")
    val exp: OpKernelGroup = OpKernelGroup("exp", "_strided")
    val sqr: OpKernelGroup = OpKernelGroup("sqr", "_strided")
    val sqrt: OpKernelGroup = OpKernelGroup("sqrt", "_strided")
    val neg: OpKernelGroup = OpKernelGroup("neg", "_strided")
    val log: OpKernelGroup = OpKernelGroup("log", "_strided")
    val gelu: OpKernelGroup = OpKernelGroup("gelu", "_strided")
    val abs: OpKernelGroup = OpKernelGroup("abs", "_strided")
    val ceil: OpKernelGroup = OpKernelGroup("ceil", "_strided")
    val floor: OpKernelGroup = OpKernelGroup("floor", "_strided")
    val relu: OpKernelGroup = OpKernelGroup("relu", "_strided")
    val round: OpKernelGroup = OpKernelGroup("round", "_strided")
    val erf: OpKernelGroup = OpKernelGroup("erf", "_strided")
    val geluErf: OpKernelGroup = OpKernelGroup("gelu_erf", "_strided")
    val tanh: OpKernelGroup = OpKernelGroup("tanh", "_strided")
    val recip: OpKernelGroup = OpKernelGroup("recip", "_strided")
    val silu: OpKernelGroup = OpKernelGroup("silu", "_strided")
    val sign: OpKernelGroup = OpKernelGroup("sign", "_strided")
    val sigmoid: OpKernelGroup = OpKernelGroup("sigmoid", "_strided")
    val constSet: OpKernelGroup = OpKernelGroup("const_set", "_strided")
    val copy: OpKernelGroup = OpKernelGroup("copy", "_strided")
}

object Copy2d {
    val FLOAT: Kernel = Kernel("copy2d_f32")
    val HALF: Kernel = Kernel("copy2d_f16")
    val BFLOAT: Kernel = Kernel("copy2d_bf16")
    val I64: Kernel = Kernel("copy2d_i64")
    val I32: Kernel = Kernel("copy2d_i32")
    val I16: Kernel = Kernel("copy2d_i16")
    val U32: Kernel = Kernel("copy2d_u32")
    val U8: Kernel = Kernel("copy2d_u8")
}

fun callUnaryContiguous(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: Kernel,
    dtypeSize: Long,
    length: Long,
    input: BufferOffset,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Unary, kernelName.name)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()

    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("unary ${kernelName.name} elems=$length")

    setParam(encoder, 0L, length)
    setParam(encoder, 1L, input)
    setParam(encoder, 2L, Output.new(output))

    val tileSize = getTileSize(dtypeSize)
    val tiles = if (tileSize > 0L) (length + tileSize - 1L) / tileSize else length
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, tiles)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callUnaryStrided(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: Kernel,
    shape: List<Long>,
    input: BufferOffset,
    strides: List<Long>,
    output: BufferOffset,
) {
    val pipeline = kernels.loadPipeline(device, Source.Unary, name.name)

    val length: Long = shape.fold(1L) { acc, d -> acc * d }
    val numDims = shape.size.toLong()
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, length)

    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("unary_strided ${name.name} elems=$length")
    setParam(encoder, 0L, length)
    setParam(encoder, 1L, numDims)
    setParam(encoder, 2L, shape)
    setParam(encoder, 3L, strides)
    setParam(encoder, 4L, input)
    setParam(encoder, 5L, Output.fromBufferOffset(output))
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callConstSetContiguous(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: Kernel,
    dtypeSize: Long,
    length: Long,
    input: Any?,
    output: BufferOffset,
) {
    val pipeline = kernels.loadPipeline(device, Source.Unary, kernelName.name)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()

    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("const_set ${kernelName.name} elems=$length")
    setParam(encoder, 0L, length)
    setParam(encoder, 1L, input)
    setParam(encoder, 2L, Output.fromBufferOffset(output))

    val tileSize = getTileSize(dtypeSize)
    val tiles = if (tileSize > 0L) (length + tileSize - 1L) / tileSize else length
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, tiles)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callConstSetStrided(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: Kernel,
    shape: List<Long>,
    input: Any?,
    strides: List<Long>,
    output: BufferOffset,
) {
    val pipeline = kernels.loadPipeline(device, Source.Unary, name.name)

    val length: Long = shape.fold(1L) { acc, d -> acc * d }
    val numDims = shape.size.toLong()
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, length)

    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("const_set_strided ${name.name} elems=$length")
    setParam(encoder, 0L, length)
    setParam(encoder, 1L, numDims)
    setParam(encoder, 2L, shape)
    setParam(encoder, 3L, strides)
    setParam(encoder, 4L, input)
    setParam(encoder, 5L, Output.fromBufferOffset(output))
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callCopy2d(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: Kernel,
    input: Buffer,
    output: Buffer,
    d1: Long,
    d2: Long,
    srcS: Long,
    dstS: Long,
    srcOInBytes: Long,
    dstOInBytes: Long,
) {
    val pipeline = kernels.loadPipeline(device, Source.Unary, name.name)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("copy2d ${name.name} d1=$d1 d2=$d2")
    setParam(encoder, 0L, d1)
    setParam(encoder, 1L, d2)
    setParam(encoder, 2L, srcS)
    setParam(encoder, 3L, dstS)
    setParam(encoder, 4L, BufferOffset(input, srcOInBytes))
    setParam(encoder, 5L, Output.withOffset(output, dstOInBytes))

    val gridDims = MTLSize(d1, d2, 1L)
    val groupDims = getBlockDims(d1, d2, 1L)
    encoder.dispatchThreads(gridDims, groupDims)
}

fun callCopy2d(
    device: Device,
    encoder: ComputeCommandEncoder,
    kernels: Kernels,
    name: Kernel,
    input: Buffer,
    output: Buffer,
    d1: Long,
    d2: Long,
    srcS: Long,
    dstS: Long,
    srcOInBytes: Long,
    dstOInBytes: Long,
) = callCopy2d(
    device,
    ComputeCommandEncoderProvider(encoder),
    kernels,
    name,
    input,
    output,
    d1,
    d2,
    srcS,
    dstS,
    srcOInBytes,
    dstOInBytes,
)
