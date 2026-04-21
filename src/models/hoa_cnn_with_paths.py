from __future__ import annotations
from typing import Sequence
from tensorflow import keras
from tensorflow.keras import layers

def _conv_block(x, *, width: int, kernel_size: int, dilation_rate: int, dropout_rate: float, name: str):
    x = layers.Conv1D(filters=width, kernel_size=kernel_size, padding="same", dilation_rate=dilation_rate, name=f"{name}_conv")(x)
    x = layers.LayerNormalization(name=f"{name}_ln")(x)
    x = layers.Activation("relu", name=f"{name}_relu")(x)
    if dropout_rate > 0.0:
        x = layers.Dropout(dropout_rate, name=f"{name}_dropout")(x)
    return x


def build_hoa_cnn_with_paths_v1(*, sequence_length: int, num_channels: int, path_top_k: int, path_num_features: int, base_width: int = 32, kernel_size: int = 9, dilation_schedule: Sequence[int] = (1,1,2,2,4,4), width_schedule: Sequence[int] = (32,32,64,64,32,32), dropout_rate: float = 0.0, residual_prediction: bool = False, path_branch_width: int = 64, path_embedding_width: int = 32) -> keras.Model:
    low_inputs = keras.Input(shape=(sequence_length, num_channels), name="low_hoa_input")
    path_inputs = keras.Input(shape=(path_top_k, path_num_features), name="path_features_input")
    x = layers.Conv1D(filters=base_width, kernel_size=1, padding="same", name="input_projection")(low_inputs)
    x = layers.LayerNormalization(name="input_projection_ln")(x)
    x = layers.Activation("relu", name="input_projection_relu")(x)
    p = layers.Flatten(name="path_flatten")(path_inputs)
    p = layers.Dense(path_branch_width, name="path_dense_1")(p)
    p = layers.LayerNormalization(name="path_dense_1_ln")(p)
    p = layers.Activation("relu", name="path_dense_1_relu")(p)
    p = layers.Dense(path_embedding_width, name="path_dense_2")(p)
    p = layers.LayerNormalization(name="path_dense_2_ln")(p)
    p = layers.Activation("relu", name="path_dense_2_relu")(p)
    p = layers.RepeatVector(sequence_length, name="path_repeat")(p)
    x = layers.Concatenate(name="feature_fusion_concat")([x, p])
    for block_idx, (width, dilation) in enumerate(zip(width_schedule, dilation_schedule)):
        shortcut = x
        x = _conv_block(x, width=width, kernel_size=kernel_size, dilation_rate=dilation, dropout_rate=dropout_rate, name=f"block{block_idx+1}_a")
        x = _conv_block(x, width=width, kernel_size=kernel_size, dilation_rate=dilation, dropout_rate=0.0, name=f"block{block_idx+1}_b")
        if shortcut.shape[-1] != width:
            shortcut = layers.Conv1D(filters=width, kernel_size=1, padding="same", name=f"block{block_idx+1}_shortcut_projection")(shortcut)
        x = layers.Add(name=f"block{block_idx+1}_residual_add")([x, shortcut])
        x = layers.Activation("relu", name=f"block{block_idx+1}_out_relu")(x)
    correction_or_target = layers.Conv1D(filters=num_channels, kernel_size=1, padding="same", name="output_projection")(x)
    outputs = layers.Add(name="predicted_high_hoa")([low_inputs, correction_or_target]) if residual_prediction else layers.Activation("linear", name="predicted_high_hoa")(correction_or_target)
    return keras.Model(inputs={"low_hoa_input": low_inputs, "path_features_input": path_inputs}, outputs=outputs, name="hoa_cnn_with_paths_v1")
