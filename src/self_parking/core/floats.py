"""Binary floating-point decoding helpers ported from TypeScript version."""

from __future__ import annotations

from dataclasses import dataclass

from self_parking.core.types import Gene

Bits = list[Gene]


@dataclass(frozen=True)
class PrecisionConfig:
    sign_bits_count: int
    exponent_bits_count: int
    fraction_bits_count: int

    @property
    def total_bits_count(self) -> int:
        return self.sign_bits_count + self.exponent_bits_count + self.fraction_bits_count


@dataclass(frozen=True)
class PrecisionConfigs:
    custom: PrecisionConfig
    half: PrecisionConfig
    single: PrecisionConfig
    double: PrecisionConfig


precision_configs = PrecisionConfigs(
    custom=PrecisionConfig(sign_bits_count=1, exponent_bits_count=4, fraction_bits_count=5),
    half=PrecisionConfig(sign_bits_count=1, exponent_bits_count=5, fraction_bits_count=10),
    single=PrecisionConfig(sign_bits_count=1, exponent_bits_count=8, fraction_bits_count=23),
    double=PrecisionConfig(sign_bits_count=1, exponent_bits_count=11, fraction_bits_count=52),
)


def bits_to_float(bits: Bits, precision_config: PrecisionConfig) -> float:
    """Decode a sign/exponent/fraction bit layout into a float.

    Note: this intentionally mirrors the simplified JS implementation.
    """
    if len(bits) != precision_config.total_bits_count:
        raise ValueError(
            f"Expected {precision_config.total_bits_count} bits, got {len(bits)}"
        )

    sign = (-1) ** bits[0]

    exponent_bias = 2 ** (precision_config.exponent_bits_count - 1) - 1
    exponent_bits = bits[
        precision_config.sign_bits_count : precision_config.sign_bits_count
        + precision_config.exponent_bits_count
    ]

    exponent_unbiased = 0
    for bit_index, current_bit in enumerate(exponent_bits):
        bit_power_of_two = 2 ** (precision_config.exponent_bits_count - bit_index - 1)
        exponent_unbiased += current_bit * bit_power_of_two
    exponent = exponent_unbiased - exponent_bias

    fraction_bits = bits[
        precision_config.sign_bits_count + precision_config.exponent_bits_count :
    ]
    fraction = 0.0
    for bit_index, current_bit in enumerate(fraction_bits):
        bit_power_of_two = 2 ** (-(bit_index + 1))
        fraction += current_bit * bit_power_of_two

    return sign * (2**exponent) * (1 + fraction)


def bits_to_float16(bits: Bits) -> float:
    if len(bits) != precision_configs.half.total_bits_count:
        raise ValueError(
            f"Expected {precision_configs.half.total_bits_count} bits, got {len(bits)}"
        )

    sign = (-1) ** bits[0]
    exponent_bits_count = precision_configs.half.exponent_bits_count
    fraction_bits_count = precision_configs.half.fraction_bits_count

    exponent_bits = bits[1 : 1 + exponent_bits_count]
    fraction_bits = bits[1 + exponent_bits_count :]

    exponent_unbiased = 0
    for bit_index, current_bit in enumerate(exponent_bits):
        exponent_unbiased += current_bit * 2 ** (exponent_bits_count - bit_index - 1)

    fraction_unbiased = 0
    for bit_index, current_bit in enumerate(fraction_bits):
        fraction_unbiased += current_bit * 2 ** (fraction_bits_count - bit_index - 1)

    exponent_bias = 2 ** (exponent_bits_count - 1) - 1
    max_exponent = 2**exponent_bits_count - 1

    if exponent_unbiased == 0:
        if fraction_unbiased == 0:
            return 0.0 * sign
        # Subnormal.
        exponent = 1 - exponent_bias
        fraction = fraction_unbiased / (2**fraction_bits_count)
        return sign * (2**exponent) * fraction

    if exponent_unbiased == max_exponent:
        if fraction_unbiased == 0:
            return float("inf") * sign
        return float("nan")

    exponent = exponent_unbiased - exponent_bias
    fraction = fraction_unbiased / (2**fraction_bits_count)
    return sign * (2**exponent) * (1 + fraction)


def bits_to_float10(bits: Bits) -> float:
    return bits_to_float(bits, precision_configs.custom)
