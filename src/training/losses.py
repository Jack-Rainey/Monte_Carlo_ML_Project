from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from tensorflow import keras


def _to_float_tensor(x, dtype=tf.float32) -> tf.Tensor:
    return tf.cast(tf.convert_to_tensor(x), dtype)


@keras.utils.register_keras_serializable(package="tail_aware")
class TailAwareRIRLoss(keras.losses.Loss):
    """
    Tail-aware raw-space loss for HOA RIR prediction.

    The training pipeline currently feeds normalized targets [B, T, C].
    This loss denormalizes y_true and y_pred back into raw HOA amplitude
    using target-channel statistics, then applies:

    1. Piecewise time-weighted Huber loss over the full IR.
    2. Late-window Huber loss.
    3. Multi-resolution STFT loss over the full IR.
    4. Multi-resolution STFT loss over the late window.
    5. Energy decay curve (EDC) loss over the full IR.
    6. Energy decay curve (EDC) loss over the late window.

    All weights can be set to 0.0 for ablations.
    """

    def __init__(
        self,
        *,
        target_mean,
        target_std,
        sample_rate_hz: int,
        huber_delta: float = 1.0,
        late_start_ms: float = 80.0,
        early_ms: float = 50.0,
        mid_ms: float = 200.0,
        early_weight: float = 1.0,
        mid_weight: float = 3.0,
        late_weight: float = 6.0,
        wave_weight: float = 1.0,
        late_wave_weight: float = 0.5,
        mrstft_weight: float = 0.0,
        late_mrstft_weight: float = 0.0,
        edc_weight: float = 0.05,
        late_edc_weight: float = 0.05,
        stft_resolutions=((512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)),
        edc_floor_db: float = -60.0,
        eps: float = 1e-8,
        name: str = "tail_aware_rir_loss",
    ) -> None:
        super().__init__(name=name, reduction=keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.sample_rate_hz = int(sample_rate_hz)
        self.huber_delta = float(huber_delta)
        self.late_start_ms = float(late_start_ms)
        self.early_ms = float(early_ms)
        self.mid_ms = float(mid_ms)
        self.early_weight = float(early_weight)
        self.mid_weight = float(mid_weight)
        self.late_weight = float(late_weight)
        self.wave_weight = float(wave_weight)
        self.late_wave_weight = float(late_wave_weight)
        self.mrstft_weight = float(mrstft_weight)
        self.late_mrstft_weight = float(late_mrstft_weight)
        self.edc_weight = float(edc_weight)
        self.late_edc_weight = float(late_edc_weight)
        self.stft_resolutions = tuple(tuple(int(v) for v in triple) for triple in stft_resolutions)
        self.edc_floor_db = float(edc_floor_db)
        self.eps = float(eps)

        target_mean = np.asarray(target_mean, dtype=np.float32)
        target_std = np.asarray(target_std, dtype=np.float32)
        if target_mean.ndim != 1 or target_std.ndim != 1:
            raise ValueError("target_mean and target_std must be 1D channel vectors")
        if target_mean.shape != target_std.shape:
            raise ValueError("target_mean and target_std must have matching shapes")

        # Stored in [1, 1, C] format to match [B, T, C].
        self._target_mean_np = target_mean
        self._target_std_np = target_std
        self.target_mean = _to_float_tensor(target_mean.reshape(1, 1, -1))
        self.target_std = _to_float_tensor(target_std.reshape(1, 1, -1))

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "target_mean": self._target_mean_np.tolist(),
                "target_std": self._target_std_np.tolist(),
                "sample_rate_hz": self.sample_rate_hz,
                "huber_delta": self.huber_delta,
                "late_start_ms": self.late_start_ms,
                "early_ms": self.early_ms,
                "mid_ms": self.mid_ms,
                "early_weight": self.early_weight,
                "mid_weight": self.mid_weight,
                "late_weight": self.late_weight,
                "wave_weight": self.wave_weight,
                "late_wave_weight": self.late_wave_weight,
                "mrstft_weight": self.mrstft_weight,
                "late_mrstft_weight": self.late_mrstft_weight,
                "edc_weight": self.edc_weight,
                "late_edc_weight": self.late_edc_weight,
                "stft_resolutions": [list(r) for r in self.stft_resolutions],
                "edc_floor_db": self.edc_floor_db,
                "eps": self.eps,
                "name": self.name,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def _denormalize(self, x_norm: tf.Tensor) -> tf.Tensor:
        x_norm = tf.cast(x_norm, tf.float32)
        return x_norm * self.target_std + self.target_mean

    def _time_weights(self, num_samples: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
        t_ms = tf.cast(tf.range(num_samples), dtype) * (1000.0 / float(self.sample_rate_hz))
        weights = tf.fill([num_samples], tf.cast(self.late_weight, dtype))
        weights = tf.where(t_ms < self.mid_ms, tf.cast(self.mid_weight, dtype), weights)
        weights = tf.where(t_ms < self.early_ms, tf.cast(self.early_weight, dtype), weights)
        weights = weights / tf.maximum(tf.reduce_mean(weights), tf.cast(self.eps, dtype))
        return weights

    def _late_mask(self, num_samples: tf.Tensor, dtype: tf.dtypes.DType) -> tf.Tensor:
        start_idx = tf.cast(
            tf.round(self.late_start_ms * 1e-3 * float(self.sample_rate_hz)),
            tf.int32,
        )
        start_idx = tf.clip_by_value(start_idx, 0, num_samples)
        mask = tf.sequence_mask(
            lengths=tf.expand_dims(num_samples - start_idx, 0),
            maxlen=num_samples,
            dtype=dtype,
        )[0]
        return tf.concat(
            [tf.zeros([start_idx], dtype=dtype), mask[start_idx:]],
            axis=0,
        )

    def _huber_per_element(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        err = y_pred - y_true
        abs_err = tf.abs(err)
        delta = tf.cast(self.huber_delta, y_true.dtype)
        quadratic = tf.minimum(abs_err, delta)
        linear = abs_err - quadratic
        return 0.5 * tf.square(quadratic) + delta * linear

    def _schroeder_edc_db(self, x: tf.Tensor) -> tf.Tensor:
        # x: [B, T, C]
        energy = tf.square(x)
        energy = tf.reverse(tf.cumsum(tf.reverse(energy, axis=[1]), axis=1), axis=[1])
        first = tf.maximum(energy[:, :1, :], self.eps)
        energy = energy / first
        edc_db = 10.0 * tf.math.log(tf.maximum(energy, self.eps)) / tf.math.log(tf.constant(10.0, dtype=x.dtype))
        edc_db = tf.maximum(edc_db, tf.cast(self.edc_floor_db, x.dtype))
        return edc_db

    def _edc_l1(self, pred_raw: tf.Tensor, true_raw: tf.Tensor, late_only: bool) -> tf.Tensor:
        pred_edc = self._schroeder_edc_db(pred_raw)
        true_edc = self._schroeder_edc_db(true_raw)

        valid = tf.cast(true_edc > (self.edc_floor_db + 1e-6), tf.float32)
        if late_only:
            num_samples = tf.shape(true_raw)[1]
            late_mask = self._late_mask(num_samples, tf.float32)[tf.newaxis, :, tf.newaxis]
            valid = valid * late_mask

        valid_sum = tf.reduce_sum(valid, axis=[1, 2], keepdims=True)
        valid = valid / tf.maximum(valid_sum, 1.0)
        return tf.reduce_mean(tf.reduce_sum(tf.abs(pred_edc - true_edc) * valid, axis=[1, 2]))

    def _mrstft_single(self, pred_flat: tf.Tensor, true_flat: tf.Tensor, frame_length: int, frame_step: int, fft_length: int) -> tf.Tensor:
        if frame_length > pred_flat.shape[-1]:
            return tf.constant(0.0, dtype=tf.float32)

        window_fn = tf.signal.hann_window
        pred_stft = tf.signal.stft(
            pred_flat,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=window_fn,
            pad_end=False,
        )
        true_stft = tf.signal.stft(
            true_flat,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=fft_length,
            window_fn=window_fn,
            pad_end=False,
        )

        pred_mag = tf.maximum(tf.abs(pred_stft), self.eps)
        true_mag = tf.maximum(tf.abs(true_stft), self.eps)

        diff = true_mag - pred_mag
        sc_num = tf.sqrt(tf.reduce_sum(tf.square(diff), axis=[1, 2]))
        sc_den = tf.sqrt(tf.reduce_sum(tf.square(true_mag), axis=[1, 2]))
        spectral_convergence = tf.reduce_mean(sc_num / tf.maximum(sc_den, self.eps))

        log_mag = tf.reduce_mean(tf.abs(tf.math.log(true_mag) - tf.math.log(pred_mag)))
        return spectral_convergence + log_mag

    def _mrstft(self, pred_raw: tf.Tensor, true_raw: tf.Tensor) -> tf.Tensor:
        # [B, T, C] -> [B*C, T]
        pred_flat = tf.reshape(tf.transpose(pred_raw, perm=[0, 2, 1]), [-1, tf.shape(pred_raw)[1]])
        true_flat = tf.reshape(tf.transpose(true_raw, perm=[0, 2, 1]), [-1, tf.shape(true_raw)[1]])

        pieces = []
        for frame_length, frame_step, fft_length in self.stft_resolutions:
            if frame_length > pred_raw.shape[1]:
                continue
            pieces.append(self._mrstft_single(pred_flat, true_flat, frame_length, frame_step, fft_length))

        if not pieces:
            return tf.constant(0.0, dtype=tf.float32)
        return tf.add_n(pieces) / float(len(pieces))

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_raw = self._denormalize(y_true)
        y_pred_raw = self._denormalize(y_pred)

        num_samples = tf.shape(y_true_raw)[1]
        dtype = y_true_raw.dtype

        time_weights = self._time_weights(num_samples, dtype)[tf.newaxis, :, tf.newaxis]
        late_mask = self._late_mask(num_samples, dtype)[tf.newaxis, :, tf.newaxis]

        huber_full = self._huber_per_element(y_true_raw, y_pred_raw)
        weighted_wave = tf.reduce_mean(huber_full * time_weights)

        late_huber = huber_full * late_mask
        late_huber = tf.reduce_sum(late_huber, axis=[1, 2]) / tf.maximum(tf.reduce_sum(late_mask, axis=[1, 2]), 1.0)
        late_huber = tf.reduce_mean(late_huber)

        total = (
            self.wave_weight * weighted_wave
            + self.late_wave_weight * late_huber
        )

        if self.mrstft_weight != 0.0:
            total += self.mrstft_weight * self._mrstft(y_pred_raw, y_true_raw)

        if self.late_mrstft_weight != 0.0:
            late_start_idx = int(round(self.late_start_ms * 1e-3 * self.sample_rate_hz))
            late_start_idx = max(0, late_start_idx)
            total += self.late_mrstft_weight * self._mrstft(y_pred_raw[:, late_start_idx:, :], y_true_raw[:, late_start_idx:, :])

        if self.edc_weight != 0.0:
            total += self.edc_weight * self._edc_l1(y_pred_raw, y_true_raw, late_only=False)

        if self.late_edc_weight != 0.0:
            total += self.late_edc_weight * self._edc_l1(y_pred_raw, y_true_raw, late_only=True)

        return tf.cast(total, y_pred.dtype)
