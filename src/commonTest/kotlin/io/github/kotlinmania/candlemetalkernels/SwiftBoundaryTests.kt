package io.github.kotlinmania.candlemetalkernels

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertIs
import kotlin.test.assertNull
import kotlin.test.assertSame

class SwiftBoundaryTests {
    @Test
    fun errorValueRetainsItsCauseWhenRaised() {
        val cause = MetalKernelError.InvalidInput("missing dimensions")
        val error = MetalKernelError.WithBacktrace(cause, "kernel invocation")
        val failure = assertFailsWith<MetalKernelException> { error.raise() }
        assertSame(error, MetalKernelError.fromException(failure))
        assertNull(MetalKernelError.fromException(IllegalArgumentException("unrelated")))
        assertEquals("Invalid input: missing dimensions\nkernel invocation", failure.message)
        assertSame(cause, assertIs<MetalKernelException>(failure.cause).error)
    }

    @Test
    fun readOnlyGuardPreservesItsValueWhenWidened() {
        val guard = RwLockGuard("kernel")
        val widened: RwLockGuard<Any> = guard
        assertEquals("kernel", widened.deref())
    }
}
