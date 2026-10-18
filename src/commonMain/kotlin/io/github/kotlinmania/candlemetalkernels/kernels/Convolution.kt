// port-lint: source kernels/convolution.rs
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

fun callIm2col1dStrided(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    strides: List<Long>,
    kSize: Long,
    stride: Long,
    padding: Long,
    dilation: Long,
    input: BufferOffset,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Conv, name)
    val lOut = (shape[2] + 2L * padding - dilation * (kSize - 1L) - 1L) / stride + 1L
    val dstEl = shape[0] * lOut * shape[1] * kSize

    val encoder = ep.encoder()
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, dstEl)
    setParam(encoder, 1L, lOut)
    setParam(encoder, 2L, kSize)
    setParam(encoder, 3L, stride)
    setParam(encoder, 4L, padding)
    setParam(encoder, 5L, dilation)
    setParam(encoder, 6L, shape)
    setParam(encoder, 7L, strides)
    setParam(encoder, 8L, input)
    setParam(encoder, 9L, Output.new(output))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callCol2im1d(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    kSize: Long,
    stride: Long,
    input: BufferOffset,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Conv, name)
    val lIn = shape[1]
    val cOut = shape[2]
    val lOut = (lIn - 1L) * stride + kSize
    val dstEl = shape[0] * cOut * lOut

    val encoder = ep.encoder()
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, dstEl)
    setParam(encoder, 1L, lOut)
    setParam(encoder, 2L, lIn)
    setParam(encoder, 3L, cOut)
    setParam(encoder, 4L, kSize)
    setParam(encoder, 5L, stride)
    setParam(encoder, 6L, input)
    setParam(encoder, 7L, Output.new(output))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callIm2colStrided(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    strides: List<Long>,
    hK: Long,
    wK: Long,
    stride: Long,
    padding: Long,
    dilation: Long,
    input: BufferOffset,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Conv, name)

    val h = shape[2]
    val w = shape[3]
    val hOut = (h + 2L * padding - dilation * (hK - 1L) - 1L) / stride + 1L
    val wOut = (w + 2L * padding - dilation * (wK - 1L) - 1L) / stride + 1L

    val dstEl = shape[0] * hOut * wOut * shape[1] * hK * wK

    val encoder = ep.encoder()
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, dstEl)
    setParam(encoder, 1L, hOut)
    setParam(encoder, 2L, wOut)
    setParam(encoder, 3L, hK)
    setParam(encoder, 4L, wK)
    setParam(encoder, 5L, stride)
    setParam(encoder, 6L, padding)
    setParam(encoder, 7L, dilation)
    setParam(encoder, 8L, shape)
    setParam(encoder, 9L, strides)
    setParam(encoder, 10L, input)
    setParam(encoder, 11L, Output.new(output))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callUpsampleNearest2d(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    strides: List<Long>,
    outW: Long,
    outH: Long,
    input: BufferOffset,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Conv, name)
    val dstEl = outW * outH * shape[0] * shape[1]
    val scaleW = shape[2].toFloat() / outW.toFloat()
    val scaleH = shape[3].toFloat() / outH.toFloat()
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, outW)
    setParam(encoder, 1L, outH)
    setParam(encoder, 2L, scaleW)
    setParam(encoder, 3L, scaleH)
    setParam(encoder, 4L, shape)
    setParam(encoder, 5L, strides)
    setParam(encoder, 6L, input)
    setParam(encoder, 7L, Output.new(output))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callUpsampleBilinear2d(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    strides: List<Long>,
    outW: Long,
    outH: Long,
    alignCorners: Boolean,
    scaleH: Double?,
    scaleW: Double?,
    input: BufferOffset,
    output: Buffer,
) {
    val pipeline = kernels.loadPipeline(device, Source.Conv, name)
    val dstEl = outW * outH * shape[0] * shape[1]

    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, outW)
    setParam(encoder, 1L, outH)
    setParam(encoder, 2L, alignCorners)
    setParam(encoder, 3L, scaleH != null)
    setParam(encoder, 4L, scaleH?.toFloat() ?: 0.0f)
    setParam(encoder, 5L, scaleW != null)
    setParam(encoder, 6L, scaleW?.toFloat() ?: 0.0f)
    setParam(encoder, 7L, shape)
    setParam(encoder, 8L, strides)
    setParam(encoder, 9L, input)
    setParam(encoder, 10L, Output.new(output))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callPool2d(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    shape: List<Long>,
    strides: List<Long>,
    outW: Long,
    outH: Long,
    wK: Long,
    hK: Long,
    wStride: Long,
    hStride: Long,
    input: Buffer,
    output: Buffer,
) {
    val dstEl = outW * outH * shape[0] * shape[1]
    val pipeline = kernels.loadPipeline(device, Source.Conv, name)
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, wK)
    setParam(encoder, 1L, hK)
    setParam(encoder, 2L, wStride)
    setParam(encoder, 3L, hStride)
    setParam(encoder, 4L, shape)
    setParam(encoder, 5L, strides)
    setParam(encoder, 6L, input)
    setParam(encoder, 7L, Output.new(output))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

fun callConvTranspose1d(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    dilation: Long,
    stride: Long,
    padding: Long,
    outPadding: Long,
    cOut: Long,
    lOut: Long,
    bSize: Long,
    srcShape: List<Long>,
    srcStrides: List<Long>,
    kernelShape: List<Long>,
    kernelStrides: List<Long>,
    input: Buffer,
    inputOffset: Long,
    kernel: Buffer,
    kernelOffset: Long,
    output: Buffer,
) {
    val dstEl = cOut * lOut * bSize
    val pipeline = kernels.loadPipeline(device, Source.Conv, name)
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, lOut)
    setParam(encoder, 1L, stride)
    setParam(encoder, 2L, padding)
    setParam(encoder, 3L, outPadding)
    setParam(encoder, 4L, dilation)
    setParam(encoder, 5L, srcShape)
    setParam(encoder, 6L, srcStrides)
    setParam(encoder, 7L, kernelShape)
    setParam(encoder, 8L, kernelStrides)
    setParam(encoder, 9L, BufferOffset(input, inputOffset))
    setParam(encoder, 10L, BufferOffset(kernel, kernelOffset))
    setParam(encoder, 11L, Output.new(output))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}

class CallConvTranspose2dCfg(
    val dilation: Long,
    val stride: Long,
    val padding: Long,
    val outputPadding: Long,
    val cOut: Long,
    val outW: Long,
    val outH: Long,
    val bSize: Long,
    val inputDims: List<Long>,
    val inputStride: List<Long>,
    val kernelDims: List<Long>,
    val kernelStride: List<Long>,
    val inputOffset: Long,
    val kernelOffset: Long,
)

fun callConvTranspose2d(
    device: Device,
    ep: EncoderProvider,
    kernels: Kernels,
    name: String,
    cfg: CallConvTranspose2dCfg,
    input: Buffer,
    kernel: Buffer,
    output: Buffer,
) {
    val dstEl = cfg.cOut * cfg.outW * cfg.outH * cfg.bSize
    val pipeline = kernels.loadPipeline(device, Source.Conv, name)
    val (threadGroupCount, threadGroupSize) = linearSplit(pipeline, dstEl)
    val encoder = ep.encoder()
    encoder.setComputePipelineState(pipeline)

    setParam(encoder, 0L, cfg.outW)
    setParam(encoder, 1L, cfg.outH)
    setParam(encoder, 2L, cfg.stride)
    setParam(encoder, 3L, cfg.padding)
    setParam(encoder, 4L, cfg.outputPadding)
    setParam(encoder, 5L, cfg.dilation)
    setParam(encoder, 6L, cfg.inputDims)
    setParam(encoder, 7L, cfg.inputStride)
    setParam(encoder, 8L, cfg.kernelDims)
    setParam(encoder, 9L, cfg.kernelStride)
    setParam(encoder, 10L, BufferOffset(input, cfg.inputOffset))
    setParam(encoder, 11L, BufferOffset(kernel, cfg.kernelOffset))
    setParam(encoder, 12L, Output.new(output))

    encoder.dispatchThreadGroups(threadGroupCount, threadGroupSize)
}
