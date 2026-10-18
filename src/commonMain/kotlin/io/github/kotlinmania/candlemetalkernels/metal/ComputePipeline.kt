// port-lint: source metal/compute_pipeline.rs
package io.github.kotlinmania.candlemetalkernels.metal

class ComputePipeline(
    val rawVal: Any? = null,
    val maxThreadsVal: Long = 1024L,
) {
    fun maxTotalThreadsPerThreadgroup(): Long = maxThreadsVal

    fun asRef(): ComputePipeline = this

    companion object {
        fun new(raw: Any? = null, maxThreads: Long = 1024L): ComputePipeline =
            ComputePipeline(raw, maxThreads)
    }
}
