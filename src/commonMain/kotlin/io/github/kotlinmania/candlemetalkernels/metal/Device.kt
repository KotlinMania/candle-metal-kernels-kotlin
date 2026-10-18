// port-lint: source metal/device.rs
package io.github.kotlinmania.candlemetalkernels.metal

enum class MetalDeviceType {
    Phone,
    BasePro,
    Max,
    Ultra,
    Medium,
}

enum class MTLMathMode {
    Default,
    Fast,
    Relaxed,
}

enum class MTLMathFloatingPointFunctions {
    Fast,
    Precise,
}

class MTLCompileOptions(
    var fastMathEnabled: Boolean = true,
    var mathMode: MTLMathMode = MTLMathMode.Fast,
    var mathFloatingPointFunctions: MTLMathFloatingPointFunctions = MTLMathFloatingPointFunctions.Fast,
) {
    companion object {
        fun new(): MTLCompileOptions = MTLCompileOptions()
    }
}

class Device(
    val rawVal: Any? = null,
    val registryIdVal: Long = 1L,
) {
    fun asRef(): Device = this

    fun registryId(): Long = registryIdVal

    fun newBuffer(length: Long, options: MTLResourceOptions): Buffer =
        Buffer.new(length)

    fun newBufferWithData(pointer: Any?, length: Long, options: MTLResourceOptions): Buffer {
        val data = if (pointer is ByteArray) pointer.copyOf() else ByteArray(length.toInt())
        return Buffer.new(length, data)
    }

    fun newLibraryWithSource(source: String, options: MTLCompileOptions? = null): Library =
        Library(sourceVal = source)

    fun newComputePipelineStateWithFunction(function: Function): ComputePipeline =
        ComputePipeline.new(function)

    fun newCommandQueue(): CommandQueue =
        CommandQueue.new()

    fun recommendedMaxWorkingSetSize(): Long = 32L * 1024L * 1024L * 1024L

    fun currentAllocatedSize(): Long = 0L

    fun architectureName(): String = "applegpu_g14g"

    fun deviceType(): MetalDeviceType =
        when (architectureName().lastOrNull()) {
            'p' -> MetalDeviceType.Phone
            'g' -> MetalDeviceType.BasePro
            's' -> MetalDeviceType.Max
            'd' -> MetalDeviceType.Ultra
            else -> MetalDeviceType.Medium
        }

    companion object {
        fun new(raw: Any? = null): Device = Device(raw)

        fun all(): List<Device> = listOf(Device())

        fun systemDefault(): Device? = Device()
    }
}
