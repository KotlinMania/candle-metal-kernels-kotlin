package io.github.kotlinmania.candlemetalkernels

import io.github.kotlinmania.candlemetalkernels.metal.MTLResourceOptions

val RESOURCE_OPTIONS: MTLResourceOptions =
    MTLResourceOptions(
        MTLResourceOptions.StorageModeShared.value or MTLResourceOptions.HazardTrackingModeUntracked.value,
    )
