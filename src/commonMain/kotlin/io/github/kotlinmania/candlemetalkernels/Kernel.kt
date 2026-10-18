// port-lint: source kernel.rs
package io.github.kotlinmania.candlemetalkernels

import io.github.kotlinmania.candlemetalkernels.metal.ComputePipeline
import io.github.kotlinmania.candlemetalkernels.metal.ConstantValues
import io.github.kotlinmania.candlemetalkernels.metal.Device
import io.github.kotlinmania.candlemetalkernels.metal.Function
import io.github.kotlinmania.candlemetalkernels.metal.Library
import io.github.kotlinmania.candlemetalkernels.metal.MTLCompileOptions
import io.github.kotlinmania.candlemetalkernels.metal.MTLMathFloatingPointFunctions
import io.github.kotlinmania.candlemetalkernels.metal.MTLMathMode

sealed class KernelName {
    data class Ref(
        val name: String,
    ) : KernelName()

    data class Value(
        val name: String,
    ) : KernelName()

    fun asRef(): String =
        when (this) {
            is Ref -> name
            is Value -> name
        }

    fun hash(): Int = hashCode()

    fun eq(other: KernelName): Boolean = asRef() == other.asRef()

    companion object {
        fun from(value: String): KernelName = Value(value)
    }
}

internal typealias Libraries = MutableMap<Source, Library>
internal typealias Pipelines = MutableMap<Pair<KernelName, ConstantValues?>, ComputePipeline>

class Kernels {
    private val libraries: Libraries = mutableMapOf()
    private val pipelines: Pipelines = mutableMapOf()

    private fun getLibrarySource(source: Source): String =
        when (source) {
            Source.Affine -> AFFINE
            Source.Binary -> BINARY
            Source.Cast -> CAST
            Source.Conv -> CONV
            Source.Fill -> FILL
            Source.Gemm -> MLX_GEMM
            Source.Gemv -> GEMV
            Source.Indexing -> INDEXING
            Source.MlxSort -> MLX_SORT
            Source.Quantized -> QUANTIZED
            Source.Random -> RANDOM
            Source.Reduce -> REDUCE
            Source.Sort -> SORT
            Source.Ternary -> TERNARY
            Source.Unary -> UNARY
            Source.Sdpa -> SDPA
        }

    fun loadLibrary(device: Device, source: Source): Library {
        val existing = libraries[source]
        if (existing != null) return existing
        val sourceContent = getLibrarySource(source)
        val compileOptions = getCompileOptions()
        val lib = device.newLibraryWithSource(sourceContent, compileOptions)
        libraries[source] = lib
        return lib
    }

    private fun loadFunction(
        device: Device,
        source: Source,
        name: String,
        constants: ConstantValues? = null,
    ): Function {
        val lib = loadLibrary(device, source)
        return lib.getFunction(name, constants)
    }

    fun loadPipelineWithConstants(
        device: Device,
        source: Source,
        name: KernelName,
        constants: ConstantValues? = null,
    ): ComputePipeline {
        val key = Pair(name, constants)
        val existing = pipelines[key]
        if (existing != null) return existing
        val func = loadFunction(device, source, name.asRef(), constants)
        val pipeline = device.newComputePipelineStateWithFunction(func)
        pipelines[key] = pipeline
        return pipeline
    }

    fun loadPipelineWithConstants(
        device: Device,
        source: Source,
        name: String,
        constants: ConstantValues? = null,
    ): ComputePipeline = loadPipelineWithConstants(device, source, KernelName.from(name), constants)

    fun loadPipeline(
        device: Device,
        source: Source,
        name: String,
    ): ComputePipeline = loadPipelineWithConstants(device, source, KernelName.from(name), null)

    fun loadPipeline(
        device: Device,
        source: Source,
        name: KernelName,
    ): ComputePipeline = loadPipelineWithConstants(device, source, name, null)

    companion object {
        fun default(): Kernels = Kernels()

        fun new(): Kernels = Kernels()
    }
}

private fun getCompileOptions(): MTLCompileOptions {
    val compileOptions = MTLCompileOptions.new()
    val fastMathEnabled = getEnvBool("CANDLE_METAL_ENABLE_FAST_MATH", true)
    if (fastMathEnabled) {
        compileOptions.mathMode = MTLMathMode.Fast
        compileOptions.mathFloatingPointFunctions = MTLMathFloatingPointFunctions.Fast
    } else {
        compileOptions.mathMode = MTLMathMode.Relaxed
        compileOptions.mathFloatingPointFunctions = MTLMathFloatingPointFunctions.Precise
    }
    compileOptions.fastMathEnabled = fastMathEnabled
    return compileOptions
}
