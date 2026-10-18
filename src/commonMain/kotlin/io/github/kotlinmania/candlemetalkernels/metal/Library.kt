// port-lint: source metal/library.rs
package io.github.kotlinmania.candlemetalkernels.metal

enum class MTLDataType {
    ULong,
    Float,
    UShort,
    Bool,
}

class FunctionConstantValues {
    private val values: MutableMap<Long, Pair<Any, MTLDataType>> = mutableMapOf()

    fun <T : Any> setConstantValueAtIndex(value: T, dtype: MTLDataType, index: Long) {
        values[index] = Pair(value, dtype)
    }

    companion object {
        fun new(): FunctionConstantValues = FunctionConstantValues()

        fun default(): FunctionConstantValues = FunctionConstantValues()
    }
}

sealed class Value {
    data class USize(
        val value: Long,
    ) : Value()

    data class Bool(
        val value: Boolean,
    ) : Value()

    data class F32(
        val value: Float,
    ) : Value()

    data class U16(
        val value: UShort,
    ) : Value()

    fun dataType(): MTLDataType =
        when (this) {
            is USize -> MTLDataType.ULong
            is F32 -> MTLDataType.Float
            is U16 -> MTLDataType.UShort
            is Bool -> MTLDataType.Bool
        }

    fun hash(): Int = hashCode()
}

data class ConstantValue(
    val index: Long,
    val value: Value,
)

class ConstantValues(
    val values: List<ConstantValue>,
) {
    fun functionConstantValues(): FunctionConstantValues {
        val f = FunctionConstantValues.new()
        for (item in values) {
            val ty = item.value.dataType()
            when (val value = item.value) {
                is Value.USize -> f.setConstantValueAtIndex(value.value, ty, item.index)
                is Value.F32 -> f.setConstantValueAtIndex(value.value, ty, item.index)
                is Value.U16 -> f.setConstantValueAtIndex(value.value, ty, item.index)
                is Value.Bool -> f.setConstantValueAtIndex(value.value, ty, item.index)
            }
        }
        return f
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is ConstantValues) return false
        return values == other.values
    }

    override fun hashCode(): Int = values.hashCode()

    companion object {
        fun new(values: List<ConstantValue>): ConstantValues = ConstantValues(values)
    }
}

class Function(
    val nameVal: String = "",
    val rawVal: Any? = null,
) {
    fun asRef(): Function = this
}

class Library(
    val rawVal: Any? = null,
    val sourceVal: String = "",
) {
    fun getFunction(name: String, constantValues: ConstantValues? = null): Function =
        Function(name)

    companion object {
        fun new(raw: Any? = null): Library = Library(raw)
    }
}
