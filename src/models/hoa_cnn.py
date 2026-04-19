from __future__ import annotations

from typing import Sequence

from tensorflow import keras
from tensorflow.keras import layers


def _conv_block(
    x,
    *,
    width: int,
    kernel_size: int,
    dilation_rate: int,
    dropout_rate: float,
    name: str,
):
    x = layers.Conv1D(
        filters=width,
        kernel_size=kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        name=f"{name}_conv",
    )(x)
    x = layers.LayerNormalization(name=f"{name}_ln")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)

    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)

    return x


def build_hoa_cnn_v1(
    *,
    sequence_length: int,
    num_channels: int,
    base_width: int = 32,
    kernel_size: int = 9,
    dilation_schedule: Sequence[int] = (1, 1, 2, 2, 4, 4),
    width_schedule: Sequence[int] = (32, 32, 64, 64, 32, 32),
    dropout_rate: float = 0.0,
    residual_prediction: bool = False,
) -> keras.Model:
    if len(dilation_schedule) != len(width_schedule):
        raise ValueError("dilation_schedule and width_schedule must have the same length")

    inputs = keras.Input(shape=(sequence_length, num_channels), name="low_hoa_input")

    x = inputs

    # Small input projection so the network has some capacity before the main stack.
    x = layers.Conv1D(
        filters=base_width,
        kernel_size=1,
        padding="same",
        name="input_projection",
    )(x)
    x = layers.LayerNormalization(name="input_projection_ln")(x)
    x = layers.Activation("relu", name="input_projection_relu")(x)

    for block_idx, (width, dilation) in enumerate(zip(width_schedule, dilation_schedule)):
        shortcut = x

        x = _conv_block(
            x,
            width=width,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout_rate=dropout_rate,
            name=f"block{block_idx + 1}_a",
        )
        x = _conv_block(
            x,
            width=width,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout_rate=0.0,
            name=f"block{block_idx + 1}_b",
        )

        # Match shortcut width if needed.
        shortcut_width = shortcut.shape[-1]
        if shortcut_width != width:
            shortcut = layers.Conv1D(
                filters=width,
                kernel_size=1,
                padding="same",
                name=f"block{block_idx + 1}_shortcut_projection",
            )(shortcut)

        x = layers.Add(name=f"block{block_idx + 1}_residual_add")([x, shortcut])
        x = layers.Activation("relu", name=f"block{block_idx + 1}_out_relu")(x)

    # Predict either the full target or the correction term.
    correction_or_target = layers.Conv1D(
        filters=num_channels,
        kernel_size=1,
        padding="same",
        name="output_projection",
    )(x)

    if residual_prediction:
        outputs = layers.Add(name="predicted_high_hoa")([inputs, correction_or_target])
    else:
        outputs = layers.Activation("linear", name="predicted_high_hoa")(correction_or_target)

    model = keras.Model(inputs=inputs, outputs=outputs, name="hoa_cnn_v1")
    return model