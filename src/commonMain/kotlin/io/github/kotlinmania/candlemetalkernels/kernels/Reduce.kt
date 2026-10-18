// port-lint: source kernels/reduce.rs
package io.github.kotlinmania.candlemetalkernels.kernels

import io.github.kotlinmania.candlemetalkernels.BufferOffset
import io.github.kotlinmania.candlemetalkernels.EncoderProvider
import io.github.kotlinmania.candlemetalkernels.Kernels
import io.github.kotlinmania.candlemetalkernels.Output
import io.github.kotlinmania.candlemetalkernels.Source
import io.github.kotlinmania.candlemetalkernels.linearSplit
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.candlemetalkernels.metal.ComputeCommandEncoder
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.metal.MTLSize
import io.github.kotlinmania.candlemetalkernels.setParam

private fun nextPowerOfTwo(n: Long): Long {
    if (n <= 1L) return 1L
    var v = n - 1L
    v = v or (v ushr 1)
    v = v or (v ushr 2)
    v = v or (v ushr 4)
    v = v or (v ushr 8)
    v = v or (v ushr 16)
    v = v or (v ushr 32)
    return v + 1L
}

fun callReduceContiguous(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    shape: List<Long>,
    outLength: Long,
    input: BufferOffset,
    output: Buffer,
) {
    val length: Long = shape.fold(1L) { acc, d -> acc * d }
    val numDims = shape.size.toLong()
    val workPerThreadgroup = length / outLength

    val pipeline = kernels.loadPipeline(device, Source.Reduce, kernelName)

    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("reduce $kernelName length=$length out_length=$outLength")

    val shapeU32 = shape.map { it.toUInt() }
    setParam(encoder, 0L, length.toUInt())
    setParam(encoder, 1L, numDims.toUInt())
    setParam(encoder, 2L, shapeU32)
    setParam(encoder, 3L, workPerThreadgroup.toUInt())
    setParam(encoder, 4L, input)
    setParam(encoder, 5L, Output.new(output))

    val width =
        minOf(
            pipeline.maxTotalThreadsPerThreadgroup(),
            nextPowerOfTwo(workPerThreadgroup / 2L),
        )
    encoder.dispatchThreadGroups(
        MTLSize(outLength, 1L, 1L),
        MTLSize(width, 1L, 1L),
    )
}

fun callReduceStrided(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    shape: List<Long>,
    strides: List<Long>,
    outLength: Long,
    input: BufferOffset,
    output: Buffer,
) {
    val length: Long = shape.fold(1L) { acc, d -> acc * d }
    val numDims = shape.size.toLong()
    val workPerThreadgroup = length / outLength

    val pipeline = kernels.loadPipeline(device, Source.Reduce, kernelName)

    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("reduce_strided $kernelName length=$length out_length=$outLength")

    val shapeU32 = shape.map { it.toUInt() }
    val stridesU32 = strides.map { it.toUInt() }
    setParam(encoder, 0L, length.toUInt())
    setParam(encoder, 1L, numDims.toUInt())
    setParam(encoder, 2L, shapeU32)
    setParam(encoder, 3L, stridesU32)
    setParam(encoder, 4L, workPerThreadgroup.toUInt())
    setParam(encoder, 5L, input)
    setParam(encoder, 6L, Output.new(output))

    val width =
        minOf(
            pipeline.maxTotalThreadsPerThreadgroup(),
            nextPowerOfTwo(workPerThreadgroup / 2L),
        )
    encoder.dispatchThreadGroups(
        MTLSize(outLength, 1L, 1L),
        MTLSize(width, 1L, 1L),
    )
}

fun callLastSoftmax(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    length: Long,
    elements: Long,
    input: Buffer,
    inputOffset: Long,
    output: Buffer,
) {
    val workPerThreadgroup = elements
    val pipeline = kernels.loadPipeline(device, Source.Reduce, kernelName)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("softmax $kernelName length=$length elements=$elements")

    setParam(encoder, 0L, length)
    setParam(encoder, 1L, workPerThreadgroup)
    setParam(encoder, 2L, BufferOffset(input, inputOffset))
    setParam(encoder, 3L, Output.new(output))

    val outLength = length / workPerThreadgroup
    val threadGroupCount = MTLSize(outLength, 1L, 1L)
    val width =
        minOf(
            pipeline.maxTotalThreadsPerThreadgroup(),
            nextPowerOfTwo(workPerThreadgroup / 2L),
        )
    val threadGroupSize = MTLSize(width, 1L, 1L)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callRmsNorm(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    length: Long,
    elementsToSum: Long,
    eps: Float,
    input: Buffer,
    inputOffset: Long,
    alpha: Buffer,
    alphaOffset: Long,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Reduce, kernelName)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("rms_norm $kernelName length=$length elements_to_sum=$elementsToSum")

    setParam(encoder, 0L, length)
    setParam(encoder, 1L, elementsToSum)
    setParam(encoder, 2L, BufferOffset(input, inputOffset))
    setParam(encoder, 3L, Output.new(output))
    setParam(encoder, 4L, BufferOffset(alpha, alphaOffset))
    setParam(encoder, 5L, eps)

    val workPerThreadgroup = elementsToSum
    val outLength = length / workPerThreadgroup
    val threadGroupCount = MTLSize(outLength, 1L, 1L)
    val width =
        minOf(
            pipeline.maxTotalThreadsPerThreadgroup(),
            nextPowerOfTwo(workPerThreadgroup / 2L),
        )
    val threadGroupSize = MTLSize(width, 1L, 1L)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callLayerNorm(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    length: Long,
    elementsToSum: Long,
    eps: Float,
    input: Buffer,
    inputOffset: Long,
    alpha: Buffer,
    alphaOffset: Long,
    beta: Buffer,
    betaOffset: Long,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Reduce, kernelName)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("layer_norm $kernelName length=$length elements_to_sum=$elementsToSum")

    setParam(encoder, 0L, length)
    setParam(encoder, 1L, elementsToSum)
    setParam(encoder, 2L, BufferOffset(input, inputOffset))
    setParam(encoder, 3L, Output.new(output))
    setParam(encoder, 4L, BufferOffset(alpha, alphaOffset))
    setParam(encoder, 5L, BufferOffset(beta, betaOffset))
    setParam(encoder, 6L, eps)

    val workPerThreadgroup = elementsToSum
    val outLength = length / workPerThreadgroup
    val threadGroupCount = MTLSize(outLength, 1L, 1L)
    val width =
        minOf(
            pipeline.maxTotalThreadsPerThreadgroup(),
            nextPowerOfTwo(workPerThreadgroup / 2L),
        )
    val threadGroupSize = MTLSize(width, 1L, 1L)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callRopeI(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    bh: Long,
    td: Long,
    strideB: Long,
    src: Buffer,
    srcOffset: Long,
    cos: Buffer,
    cosOffset: Long,
    sin: Buffer,
    sinOffset: Long,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Reduce, kernelName)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("rope_i $kernelName bh=$bh td=$td")

    setParam(encoder, 0L, bh)
    setParam(encoder, 1L, td)
    setParam(encoder, 2L, strideB)
    setParam(encoder, 3L, BufferOffset(src, srcOffset))
    setParam(encoder, 4L, BufferOffset(cos, cosOffset))
    setParam(encoder, 5L, BufferOffset(sin, sinOffset))
    setParam(encoder, 6L, Output.new(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, (bh * td) / 2L)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callRopeThd(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    b: Long,
    t: Long,
    h: Long,
    d: Long,
    strideB: Long,
    src: Buffer,
    srcOffset: Long,
    cos: Buffer,
    cosOffset: Long,
    sin: Buffer,
    sinOffset: Long,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Reduce, kernelName)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("rope_thd $kernelName b=$b t=$t h=$h d=$d")

    setParam(encoder, 0L, b)
    setParam(encoder, 1L, t)
    setParam(encoder, 2L, h)
    setParam(encoder, 3L, d)
    setParam(encoder, 4L, strideB)
    setParam(encoder, 5L, BufferOffset(src, srcOffset))
    setParam(encoder, 6L, BufferOffset(cos, cosOffset))
    setParam(encoder, 7L, BufferOffset(sin, sinOffset))
    setParam(encoder, 8L, Output.new(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, (b * t * h * d) / 2L)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callRope(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    kernelName: String,
    bh: Long,
    td: Long,
    d: Long,
    strideB: Long,
    src: Buffer,
    srcOffset: Long,
    cos: Buffer,
    cosOffset: Long,
    sin: Buffer,
    sinOffset: Long,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Reduce, kernelName)
    val encoder: ComputeCommandEncoder = ep.encoder().asRef()
    encoder.setComputePipelineState(pipeline)
    encoder.debugGroup("rope $kernelName bh=$bh td=$td d=$d")

    setParam(encoder, 0L, bh)
    setParam(encoder, 1L, td)
    setParam(encoder, 2L, d)
    setParam(encoder, 3L, strideB)
    setParam(encoder, 4L, BufferOffset(src, srcOffset))
    setParam(encoder, 5L, BufferOffset(cos, cosOffset))
    setParam(encoder, 6L, BufferOffset(sin, sinOffset))
    setParam(encoder, 7L, Output.new(output))

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, (bh * td) / 2L)
    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}
