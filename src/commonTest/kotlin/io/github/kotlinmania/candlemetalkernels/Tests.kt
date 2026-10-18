// port-lint: tests tests.rs
package io.github.kotlinmania.candlemetalkernels

import io.github.kotlinmania.candlemetalkernels.kernels.CallConvTranspose2dCfg
import io.github.kotlinmania.candlemetalkernels.kernels.Copy2d
import io.github.kotlinmania.candlemetalkernels.kernels.GemmDType
import io.github.kotlinmania.candlemetalkernels.kernels.GgmlDType
import io.github.kotlinmania.candlemetalkernels.kernels.SdpaDType
import io.github.kotlinmania.candlemetalkernels.kernels.UnaryContiguous
import io.github.kotlinmania.candlemetalkernels.kernels.UnaryStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callAffine
import io.github.kotlinmania.candlemetalkernels.kernels.callAffineStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callArgSort
import io.github.kotlinmania.candlemetalkernels.kernels.callBinaryContiguous
import io.github.kotlinmania.candlemetalkernels.kernels.callBinaryStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callCastContiguous
import io.github.kotlinmania.candlemetalkernels.kernels.callCastStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callCol2im1d
import io.github.kotlinmania.candlemetalkernels.kernels.callConstFill
import io.github.kotlinmania.candlemetalkernels.kernels.callConstSetContiguous
import io.github.kotlinmania.candlemetalkernels.kernels.callConstSetStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callConvTranspose1d
import io.github.kotlinmania.candlemetalkernels.kernels.callConvTranspose2d
import io.github.kotlinmania.candlemetalkernels.kernels.callCopy2d
import io.github.kotlinmania.candlemetalkernels.kernels.callElu
import io.github.kotlinmania.candlemetalkernels.kernels.callEluStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callGather
import io.github.kotlinmania.candlemetalkernels.kernels.callIm2col1dStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callIm2colStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callIndexAdd
import io.github.kotlinmania.candlemetalkernels.kernels.callIndexSelect
import io.github.kotlinmania.candlemetalkernels.kernels.callLastSoftmax
import io.github.kotlinmania.candlemetalkernels.kernels.callLayerNorm
import io.github.kotlinmania.candlemetalkernels.kernels.callMlxArgSort
import io.github.kotlinmania.candlemetalkernels.kernels.callMlxGemm
import io.github.kotlinmania.candlemetalkernels.kernels.callPool2d
import io.github.kotlinmania.candlemetalkernels.kernels.callPowf
import io.github.kotlinmania.candlemetalkernels.kernels.callPowfStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callQuantizedMatmulMmT
import io.github.kotlinmania.candlemetalkernels.kernels.callQuantizedMatmulMvT
import io.github.kotlinmania.candlemetalkernels.kernels.callRandomNormal
import io.github.kotlinmania.candlemetalkernels.kernels.callRandomUniform
import io.github.kotlinmania.candlemetalkernels.kernels.callReduceContiguous
import io.github.kotlinmania.candlemetalkernels.kernels.callReduceStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callRmsNorm
import io.github.kotlinmania.candlemetalkernels.kernels.callRope
import io.github.kotlinmania.candlemetalkernels.kernels.callRopeI
import io.github.kotlinmania.candlemetalkernels.kernels.callRopeThd
import io.github.kotlinmania.candlemetalkernels.kernels.callScatter
import io.github.kotlinmania.candlemetalkernels.kernels.callSdpaFull
import io.github.kotlinmania.candlemetalkernels.kernels.callSdpaVector
import io.github.kotlinmania.candlemetalkernels.kernels.callSdpaVector2pass
import io.github.kotlinmania.candlemetalkernels.kernels.callUnaryContiguous
import io.github.kotlinmania.candlemetalkernels.kernels.callUnaryStrided
import io.github.kotlinmania.candlemetalkernels.kernels.callUpsampleBilinear2d
import io.github.kotlinmania.candlemetalkernels.kernels.callUpsampleNearest2d
import io.github.kotlinmania.candlemetalkernels.kernels.callWhereCond
import io.github.kotlinmania.candlemetalkernels.kernels.checkBatchCollapse
import io.github.kotlinmania.candlemetalkernels.kernels.selectTileConfig
import io.github.kotlinmania.candlemetalkernels.kernels.shouldUseSplitK
import io.github.kotlinmania.candlemetalkernels.metal.Commands
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.metal.MetalDeviceType
import io.github.kotlinmania.candlemetalkernels.metal.ResidencySet
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class Tests {
    private fun testDevice(): Device = Device.systemDefault() ?: Device.new()

    private fun testCommands(device: Device): Commands {
        val queue = device.newCommandQueue()
        val residencySet = ResidencySet.new(device)
        return Commands.new(queue, residencySet)
    }

    @Test
    fun commandsCreationAndEncoder() {
        val device = testDevice()
        val queue = device.newCommandQueue()
        val residencySet = ResidencySet.new(device)
        val commands = Commands.new(queue, residencySet)

        val encoder = commands.commandEncoder()
        assertNotNull(encoder)
        commands.drop()
    }

    @Test
    fun commandsConcurrentAcquisition() {
        val device = testDevice()
        val queue = device.newCommandQueue()
        val residencySet = ResidencySet.new(device)
        val commands = Commands.new(queue, residencySet)

        repeat(16) {
            val encoder = commands.commandEncoder()
            assertNotNull(encoder)
        }
        commands.waitUntilCompleted()
    }

    @Test
    fun dtypesAndSizes() {
        assertEquals(1L, DType.U8.sizeInBytes())
        assertEquals(4L, DType.U32.sizeInBytes())
        assertEquals(8L, DType.I64.sizeInBytes())
        assertEquals(2L, DType.BF16.sizeInBytes())
        assertEquals(2L, DType.F16.sizeInBytes())
        assertEquals(4L, DType.F32.sizeInBytes())
    }

    @Test
    fun unaryExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val length = 100L
        val inputBuf = device.newBuffer(length * 4L, RESOURCE_OPTIONS)
        val outputBuf = device.newBuffer(length * 4L, RESOURCE_OPTIONS)

        callUnaryContiguous(
            device,
            encoder,
            kernels,
            UnaryContiguous.cos.FLOAT,
            4L,
            length,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
        )

        callUnaryStrided(
            device,
            encoder,
            kernels,
            UnaryStrided.cos.FLOAT,
            listOf(10L, 10L),
            BufferOffset.zeroOffset(inputBuf),
            listOf(10L, 1L),
            BufferOffset.zeroOffset(outputBuf),
        )

        callConstSetContiguous(
            device,
            encoder,
            kernels,
            UnaryContiguous.constSet.FLOAT,
            4L,
            length,
            1.5f,
            BufferOffset.zeroOffset(outputBuf),
        )

        callConstSetStrided(
            device,
            encoder,
            kernels,
            UnaryStrided.constSet.FLOAT,
            listOf(10L, 10L),
            1.5f,
            listOf(10L, 1L),
            BufferOffset.zeroOffset(outputBuf),
        )

        callCopy2d(
            device,
            encoder,
            kernels,
            Copy2d.FLOAT,
            inputBuf,
            outputBuf,
            10L,
            10L,
            10L,
            10L,
            0L,
            0L,
        )

        commands.waitUntilCompleted()
    }

    @Test
    fun binaryExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val length = 64L
        val leftBuf = device.newBuffer(length * 4L, RESOURCE_OPTIONS)
        val rightBuf = device.newBuffer(length * 4L, RESOURCE_OPTIONS)
        val outputBuf = device.newBuffer(length * 4L, RESOURCE_OPTIONS)

        callBinaryContiguous(
            device,
            encoder,
            kernels,
            "badd_f32",
            4L,
            length,
            BufferOffset.zeroOffset(leftBuf),
            BufferOffset.zeroOffset(rightBuf),
            outputBuf,
        )

        callBinaryStrided(
            device,
            encoder,
            kernels,
            "badd_f32_strided",
            4L,
            listOf(8L, 8L),
            BufferOffset.zeroOffset(leftBuf),
            listOf(8L, 1L),
            BufferOffset.zeroOffset(rightBuf),
            listOf(8L, 1L),
            outputBuf,
        )

        commands.waitUntilCompleted()
    }

    @Test
    fun affineExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val length = 32L
        val inputBuf = device.newBuffer(length * 4L, RESOURCE_OPTIONS)
        val outputBuf = device.newBuffer(length * 4L, RESOURCE_OPTIONS)

        callAffine(
            device,
            encoder,
            kernels,
            "affine_f32",
            4L,
            length,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
            2.0f,
            1.0f,
        )

        callAffineStrided(
            device,
            encoder,
            kernels,
            "affine_f32_strided",
            listOf(4L, 8L),
            BufferOffset.zeroOffset(inputBuf),
            listOf(8L, 1L),
            outputBuf,
            2.0f,
            1.0f,
        )

        callPowf(
            device,
            encoder,
            kernels,
            "powf_f32",
            4L,
            length,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
            2.0f,
        )

        callPowfStrided(
            device,
            encoder,
            kernels,
            "powf_f32_strided",
            listOf(4L, 8L),
            BufferOffset.zeroOffset(inputBuf),
            listOf(8L, 1L),
            outputBuf,
            2.0f,
        )

        callElu(
            device,
            encoder,
            kernels,
            "elu_f32",
            4L,
            length,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
            1.0f,
        )

        callEluStrided(
            device,
            encoder,
            kernels,
            "elu_f32_strided",
            listOf(4L, 8L),
            BufferOffset.zeroOffset(inputBuf),
            listOf(8L, 1L),
            outputBuf,
            1.0f,
        )

        commands.waitUntilCompleted()
    }

    @Test
    fun castExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val length = 16L
        val inputBuf = device.newBuffer(length * 4L, RESOURCE_OPTIONS)
        val outputBuf = device.newBuffer(length * 2L, RESOURCE_OPTIONS)

        callCastContiguous(
            device,
            encoder,
            kernels,
            "cast_f32_f16",
            4L,
            length,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
        )

        callCastStrided(
            device,
            encoder,
            kernels,
            "cast_f32_f16_strided",
            listOf(4L, 4L),
            BufferOffset.zeroOffset(inputBuf),
            listOf(4L, 1L),
            outputBuf,
        )

        commands.waitUntilCompleted()
    }

    @Test
    fun fillExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val length = 16L
        val outputBuf = device.newBuffer(length * 4L, RESOURCE_OPTIONS)

        callConstFill(
            device,
            encoder,
            kernels,
            "fill_f32",
            length,
            outputBuf,
            3.14f,
        )

        commands.waitUntilCompleted()
    }

    @Test
    fun indexingExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val inputBuf = device.newBuffer(64L * 4L, RESOURCE_OPTIONS)
        val idsBuf = device.newBuffer(16L * 4L, RESOURCE_OPTIONS)
        val outputBuf = device.newBuffer(16L * 4L, RESOURCE_OPTIONS)

        callIndexSelect(
            device,
            encoder,
            kernels,
            "is_f32_u32",
            listOf(4L, 4L),
            4L,
            0L,
            true,
            listOf(4L, 4L),
            listOf(4L, 1L),
            BufferOffset.zeroOffset(inputBuf),
            BufferOffset.zeroOffset(idsBuf),
            outputBuf,
        )

        callGather(
            device,
            encoder,
            kernels,
            "gather_f32_u32",
            listOf(4L, 4L),
            4L,
            0L,
            BufferOffset.zeroOffset(inputBuf),
            BufferOffset.zeroOffset(idsBuf),
            outputBuf,
        )

        callScatter(
            device,
            encoder,
            kernels,
            "sa_f32_u32",
            listOf(4L, 4L),
            listOf(4L, 4L),
            0L,
            BufferOffset.zeroOffset(inputBuf),
            BufferOffset.zeroOffset(idsBuf),
            BufferOffset.zeroOffset(outputBuf),
        )

        callIndexAdd(
            device,
            encoder,
            kernels,
            "ia_f32_u32",
            listOf(4L, 4L),
            listOf(4L, 4L),
            listOf(4L),
            0L,
            BufferOffset.zeroOffset(inputBuf),
            BufferOffset.zeroOffset(idsBuf),
            outputBuf,
        )

        commands.waitUntilCompleted()
    }

    @Test
    fun convolutionExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val inputBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)
        val kernelBuf = device.newBuffer(256L, RESOURCE_OPTIONS)
        val outputBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)

        callIm2col1dStrided(
            device,
            encoder,
            kernels,
            "im2col1d_f32",
            listOf(1L, 1L, 8L),
            listOf(8L, 8L, 1L),
            3L,
            1L,
            0L,
            1L,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
        )

        callCol2im1d(
            device,
            encoder,
            kernels,
            "col2im1d_f32",
            listOf(1L, 1L, 8L),
            3L,
            1L,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
        )

        callIm2colStrided(
            device,
            encoder,
            kernels,
            "im2col_f32",
            listOf(1L, 1L, 8L, 8L),
            listOf(64L, 64L, 8L, 1L),
            3L,
            3L,
            1L,
            0L,
            1L,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
        )

        callUpsampleNearest2d(
            device,
            encoder,
            kernels,
            "upsample_nearest2d_f32",
            listOf(1L, 1L, 8L, 8L),
            listOf(64L, 64L, 8L, 1L),
            16L,
            16L,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
        )

        callUpsampleBilinear2d(
            device,
            encoder,
            kernels,
            "upsample_bilinear2d_f32",
            listOf(1L, 1L, 8L, 8L),
            listOf(64L, 64L, 8L, 1L),
            16L,
            16L,
            false,
            2.0,
            2.0,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
        )

        callPool2d(
            device,
            encoder,
            kernels,
            "max_pool2d_f32",
            listOf(1L, 1L, 8L, 8L),
            listOf(64L, 64L, 8L, 1L),
            4L,
            4L,
            2L,
            2L,
            2L,
            2L,
            inputBuf,
            outputBuf,
        )

        callConvTranspose1d(
            device,
            encoder,
            kernels,
            "conv_transpose1d_f32",
            1L,
            1L,
            0L,
            0L,
            1L,
            8L,
            1L,
            listOf(1L, 1L, 8L),
            listOf(8L, 8L, 1L),
            listOf(1L, 1L, 3L),
            listOf(3L, 3L, 1L),
            inputBuf,
            0L,
            kernelBuf,
            0L,
            outputBuf,
        )

        val cfg =
            CallConvTranspose2dCfg(
                dilation = 1L,
                stride = 1L,
                padding = 0L,
                outputPadding = 0L,
                cOut = 1L,
                outW = 8L,
                outH = 8L,
                bSize = 1L,
                inputDims = listOf(1L, 1L, 8L, 8L),
                inputStride = listOf(64L, 64L, 8L, 1L),
                kernelDims = listOf(1L, 1L, 3L, 3L),
                kernelStride = listOf(9L, 9L, 3L, 1L),
                inputOffset = 0L,
                kernelOffset = 0L,
            )
        callConvTranspose2d(
            device,
            encoder,
            kernels,
            "conv_transpose2d_f32",
            cfg,
            inputBuf,
            kernelBuf,
            outputBuf,
        )

        commands.waitUntilCompleted()
    }

    @Test
    fun randomExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val seedBuf = device.newBuffer(16L, RESOURCE_OPTIONS)
        val outputBuf = device.newBuffer(64L * 4L, RESOURCE_OPTIONS)

        callRandomUniform(
            device,
            encoder,
            kernels,
            "rand_uniform_f32",
            0.0f,
            1.0f,
            64L,
            seedBuf,
            outputBuf,
        )

        callRandomNormal(
            device,
            encoder,
            kernels,
            "rand_normal_f32",
            0.0f,
            1.0f,
            64L,
            seedBuf,
            outputBuf,
        )

        assertFailsWith<MetalKernelError.LoadLibraryError> {
            callRandomUniform(
                device,
                encoder,
                kernels,
                "rand_uniform_f32",
                2.0f,
                1.0f,
                64L,
                seedBuf,
                outputBuf,
            )
        }

        commands.waitUntilCompleted()
    }

    @Test
    fun ternaryExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val condBuf = device.newBuffer(16L, RESOURCE_OPTIONS)
        val leftBuf = device.newBuffer(64L, RESOURCE_OPTIONS)
        val rightBuf = device.newBuffer(64L, RESOURCE_OPTIONS)
        val outputBuf = device.newBuffer(64L, RESOURCE_OPTIONS)

        callWhereCond(
            device,
            encoder,
            kernels,
            "where_f32",
            4L,
            listOf(4L, 4L),
            BufferOffset.zeroOffset(condBuf),
            listOf(4L, 1L),
            true,
            BufferOffset.zeroOffset(leftBuf),
            listOf(4L, 1L),
            true,
            BufferOffset.zeroOffset(rightBuf),
            listOf(4L, 1L),
            true,
            outputBuf,
        )

        commands.waitUntilCompleted()
    }

    @Test
    fun sortExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val srcBuf = device.newBuffer(64L * 4L, RESOURCE_OPTIONS)
        val dstBuf = device.newBuffer(64L * 4L, RESOURCE_OPTIONS)

        callArgSort(
            device,
            encoder,
            kernels,
            "arg_sort_f32",
            4L,
            16L,
            16L,
            BufferOffset.zeroOffset(srcBuf),
            dstBuf,
        )

        callMlxArgSort(
            device,
            encoder,
            kernels,
            DType.F32,
            4L,
            16L,
            BufferOffset.zeroOffset(srcBuf),
            dstBuf,
        )

        commands.waitUntilCompleted()
    }

    @Test
    fun reduceExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val inputBuf = device.newBuffer(64L * 4L, RESOURCE_OPTIONS)
        val outputBuf = device.newBuffer(16L * 4L, RESOURCE_OPTIONS)
        val alphaBuf = device.newBuffer(16L * 4L, RESOURCE_OPTIONS)
        val betaBuf = device.newBuffer(16L * 4L, RESOURCE_OPTIONS)
        val cosBuf = device.newBuffer(64L * 4L, RESOURCE_OPTIONS)
        val sinBuf = device.newBuffer(64L * 4L, RESOURCE_OPTIONS)

        callReduceContiguous(
            device,
            encoder,
            kernels,
            "fast_sum_f32",
            listOf(4L, 16L),
            4L,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
        )

        callReduceStrided(
            device,
            encoder,
            kernels,
            "fast_sum_f32_strided",
            listOf(4L, 16L),
            listOf(16L, 1L),
            4L,
            BufferOffset.zeroOffset(inputBuf),
            outputBuf,
        )

        callLastSoftmax(
            device,
            encoder,
            kernels,
            "last_softmax_f32",
            64L,
            16L,
            inputBuf,
            0L,
            outputBuf,
        )

        callRmsNorm(
            device,
            encoder,
            kernels,
            "rms_norm_f32",
            64L,
            16L,
            1e-5f,
            inputBuf,
            0L,
            alphaBuf,
            0L,
            outputBuf,
        )

        callLayerNorm(
            device,
            encoder,
            kernels,
            "layer_norm_f32",
            64L,
            16L,
            1e-5f,
            inputBuf,
            0L,
            alphaBuf,
            0L,
            betaBuf,
            0L,
            outputBuf,
        )

        callRopeI(
            device,
            encoder,
            kernels,
            "rope_i_f32",
            2L,
            8L,
            8L,
            inputBuf,
            0L,
            cosBuf,
            0L,
            sinBuf,
            0L,
            outputBuf,
        )

        callRopeThd(
            device,
            encoder,
            kernels,
            "rope_thd_f32",
            1L,
            2L,
            4L,
            8L,
            64L,
            inputBuf,
            0L,
            cosBuf,
            0L,
            sinBuf,
            0L,
            outputBuf,
        )

        callRope(
            device,
            encoder,
            kernels,
            "rope_f32",
            2L,
            8L,
            4L,
            8L,
            inputBuf,
            0L,
            cosBuf,
            0L,
            sinBuf,
            0L,
            outputBuf,
        )

        commands.waitUntilCompleted()
    }

    @Test
    fun quantizedExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val lhsBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)
        val rhsBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)
        val dstBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)

        callQuantizedMatmulMvT(
            device,
            encoder,
            kernels,
            GgmlDType.Q4_0,
            1L,
            1L,
            32L,
            32L,
            lhsBuf,
            0L,
            rhsBuf,
            0L,
            dstBuf,
        )

        callQuantizedMatmulMmT(
            device,
            encoder,
            kernels,
            GgmlDType.Q4_0,
            listOf(1L, 1L, 32L, 32L),
            listOf(1024L, 1024L, 32L, 1L),
            rhsBuf,
            listOf(1L, 1L, 32L, 32L),
            listOf(1024L, 1024L, 32L, 1L),
            lhsBuf,
            0L,
            listOf(1L, 1L, 32L, 32L),
            0L,
            dstBuf,
        )

        assertFailsWith<MetalKernelError.UnsupportedDTypeForOp> {
            callQuantizedMatmulMmT(
                device,
                encoder,
                kernels,
                GgmlDType.Q8_1,
                listOf(1L, 1L, 32L, 32L),
                listOf(1024L, 1024L, 32L, 1L),
                rhsBuf,
                listOf(1L, 1L, 32L, 32L),
                listOf(1024L, 1024L, 32L, 1L),
                lhsBuf,
                0L,
                listOf(1L, 1L, 32L, 32L),
                0L,
                dstBuf,
            )
        }

        commands.waitUntilCompleted()
    }

    @Test
    fun mlxGemmExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val lhsBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)
        val rhsBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)
        val dstBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)

        // GEMV: m = 1
        callMlxGemm(
            device,
            encoder,
            kernels,
            GemmDType.F32,
            1L,
            1L,
            16L,
            16L,
            listOf(16L, 1L),
            0L,
            lhsBuf,
            listOf(16L, 1L),
            0L,
            rhsBuf,
            dstBuf,
        )

        // GEMM: m = 16, n = 16, k = 16
        callMlxGemm(
            device,
            encoder,
            kernels,
            GemmDType.F32,
            1L,
            16L,
            16L,
            16L,
            listOf(16L, 1L),
            0L,
            lhsBuf,
            listOf(16L, 1L),
            0L,
            rhsBuf,
            dstBuf,
        )

        val tile = selectTileConfig(GemmDType.F32, 16L, 16L, 16L, 1L, false, false, MetalDeviceType.Medium)
        assertNotNull(tile)

        val (effB, effM, collapsed) = checkBatchCollapse(2L, 16L, 16L, false, listOf(256L, 16L, 1L), listOf(0L, 16L, 1L))
        assertEquals(1L, effB)
        assertEquals(32L, effM)
        assertTrue(collapsed)

        assertFalse(shouldUseSplitK(2L, 16L, 16L, 16L))

        commands.waitUntilCompleted()
    }

    @Test
    fun sdpaExecution() {
        val device = testDevice()
        val commands = testCommands(device)
        val kernels = Kernels.new()
        val encoder = commands.commandEncoder()

        val qBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)
        val kBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)
        val vBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)
        val outputBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)
        val intermediateBuf = device.newBuffer(1024L, RESOURCE_OPTIONS)
        val sumsBuf = device.newBuffer(256L, RESOURCE_OPTIONS)
        val maxsBuf = device.newBuffer(256L, RESOURCE_OPTIONS)

        callSdpaFull(
            device,
            encoder,
            kernels,
            0L,
            listOf(1L, 4L, 16L, 64L),
            listOf(4096L, 1024L, 64L, 1L),
            qBuf,
            0L,
            listOf(1L, 4L, 16L, 64L),
            listOf(4096L, 1024L, 64L, 1L),
            kBuf,
            0L,
            vBuf,
            listOf(4096L, 1024L, 64L, 1L),
            null,
            null,
            null,
            outputBuf,
            listOf(4096L, 1024L, 64L, 1L),
            0.125f,
            false,
            SdpaDType.F32,
        )

        callSdpaVector(
            device,
            encoder,
            kernels,
            0L,
            listOf(1L, 4L, 1L, 64L),
            qBuf,
            0L,
            listOf(1L, 4L, 16L, 64L),
            listOf(1024L, 64L),
            kBuf,
            0L,
            listOf(1024L, 64L),
            vBuf,
            outputBuf,
            0.125f,
            1.0f,
            SdpaDType.F32,
        )

        callSdpaVector2pass(
            device,
            encoder,
            kernels,
            0L,
            listOf(1L, 4L, 1L, 64L),
            qBuf,
            0L,
            listOf(1L, 4L, 16L, 64L),
            listOf(1024L, 64L),
            kBuf,
            0L,
            listOf(1024L, 64L),
            vBuf,
            outputBuf,
            intermediateBuf,
            sumsBuf,
            maxsBuf,
            0.125f,
            1.0f,
            SdpaDType.F32,
        )

        assertFailsWith<MetalKernelError.SdpaHeadSizeMismatch> {
            callSdpaFull(
                device,
                encoder,
                kernels,
                0L,
                listOf(1L, 4L, 16L, 48L),
                listOf(4096L, 1024L, 48L, 1L),
                qBuf,
                0L,
                listOf(1L, 4L, 16L, 48L),
                listOf(4096L, 1024L, 48L, 1L),
                kBuf,
                0L,
                vBuf,
                listOf(4096L, 1024L, 48L, 1L),
                null,
                null,
                null,
                outputBuf,
                listOf(4096L, 1024L, 48L, 1L),
                0.125f,
                false,
                SdpaDType.F32,
            )
        }

        commands.waitUntilCompleted()
    }
}
