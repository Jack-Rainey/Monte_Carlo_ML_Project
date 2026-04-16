from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def residual_conv_block(
    x: keras.KerasTensor,
    *,
    filters: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float,
    name: str,
) -> keras.KerasTensor:
    shortcut = x

    y = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        name=f"{name}_conv1",
    )(x)
    y = layers.LayerNormalization(axis=-1, name=f"{name}_ln1")(y)
    y = layers.Activation("relu", name=f"{name}_relu1")(y)

    if dropout_rate > 0.0:
        y = layers.Dropout(dropout_rate, name=f"{name}_dropout")(y)

    y = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        name=f"{name}_conv2",
    )(y)
    y = layers.LayerNormalization(axis=-1, name=f"{name}_ln2")(y)

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(
            filters=filters,
            kernel_size=1,
            padding="same",
            name=f"{name}_shortcut",
        )(shortcut)

    out = layers.Add(name=f"{name}_add")([shortcut, y])
    out = layers.Activation("relu", name=f"{name}_relu_out")(out)
    return out


def build_hoa_cnn_v1(
    *,
    sequence_length: int,
    num_channels: int,
    base_width: int = 32,
    kernel_size: int = 9,
    dilation_schedule: tuple[int, ...] = (1, 1, 2, 2, 4, 4),
    width_schedule: tuple[int, ...] | None = None,
    dropout_rate: float = 0.0,
) -> keras.Model:
    if width_schedule is None:
        width_schedule = (base_width, base_width, base_width * 2, base_width * 2, base_width, base_width)

    if len(width_schedule) != len(dilation_schedule):
        raise ValueError("width_schedule and dilation_schedule must have the same length")

    inputs = keras.Input(shape=(sequence_length, num_channels), name="low_hoa")
    x = layers.Conv1D(
        filters=base_width,
        kernel_size=kernel_size,
        padding="same",
        name="stem_conv",
    )(inputs)
    x = layers.LayerNormalization(axis=-1, name="stem_ln")(x)
    x = layers.Activation("relu", name="stem_relu")(x)

    for block_idx, (filters, dilation_rate) in enumerate(zip(width_schedule, dilation_schedule), start=1):
        x = residual_conv_block(
            x,
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate,
            name=f"res_block_{block_idx}",
        )

    x = layers.Conv1D(
        filters=base_width,
        kernel_size=kernel_size,
        padding="same",
        name="head_conv",
    )(x)
    x = layers.LayerNormalization(axis=-1, name="head_ln")(x)
    x = layers.Activation("relu", name="head_relu")(x)

    outputs = layers.Conv1D(
        filters=num_channels,
        kernel_size=1,
        padding="same",
        activation="linear",
        name="predicted_high_hoa",
    )(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="hoa_cnn_v1")
