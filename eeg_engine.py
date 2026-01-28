# eeg_engine.py
# 修复版 v8：
# 1. 引入“延迟窗口 + 眨眼剔除”策略，提升指标稳定性。
# 2. 仅使用最长的一段纯净数据进行 FFT，避免眼电干扰。
# 3. 指标计算降频（每 4 帧一次），并在中间进行插值（由 EMA 自然平滑）。

import numpy as np
import time
from scipy import signal
import sys

# 添加路径以确保能找到模块
sys.path.append(".")

try:
    from bluetooth_core import EEGBluetoothReceiver
    BLUETOOTH_AVAILABLE = True
except ImportError:
    print("Warning: 'bluetooth_core.py' not found. Bluetooth mode unavailable.")
    BLUETOOTH_AVAILABLE = False


class EEGAnalyzer:
    """
    核心分析引擎 (适配 1000Hz 采样率 & 高级感官指标计算)
    """

    def __init__(self, fs=1000, buffer_seconds=4.0):
        self.fs = fs
        self.buffer_len = int(fs * buffer_seconds)

        # 1. 数据缓冲区
        self.raw_buffer = np.zeros((2, self.buffer_len))

        # 2. 实时滤波器
        nyquist = 0.5 * fs
        low = 0.5 / nyquist
        high = 82.0 / nyquist
        self.sos = signal.butter(4, [low, high], btype='band', output='sos')
        self.zi_ch1 = signal.sosfilt_zi(self.sos)
        self.zi_ch2 = signal.sosfilt_zi(self.sos)

        # 3. 频段定义
        self.BANDS = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha_low': (8, 10.5),
            'alpha_high': (10.5, 13),
            'alpha_total': (8, 13),
            'beta': (13, 30),
            'gamma_high': (50, 80),
            'full_band': (1, 60)
        }

        # 4. 指标平滑器
        self.metrics_ema = {
            "pos_score": 50.0, "neg_score": 0.0, "slope": -1.0, "alpha_asym": 0.0,
            "sensory_attention": 0.0, "emotional_valence": 0.0,
            "neural_relaxation": 0.0, "cortical_arousal": 0.0,
            "cognitive_depth": 0.0, "immersion_ratio": 0.0, "flavor_purity": 0.0,
            # 分离左右通道的功率记录
            "detailed_powers_L": {k: 0.1 for k in
                                  ['delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta', 'high_beta', 'low_gamma',
                                   'high_gamma']},
            "detailed_powers_R": {k: 0.1 for k in
                                  ['delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta', 'high_beta', 'low_gamma',
                                   'high_gamma']}
        }

        dummy_freqs = np.linspace(1, 60, 30).tolist()
        dummy_psd = np.zeros(30).tolist()
        self.latest_psd_data = {
            'freqs': dummy_freqs,
            'ch1': dummy_psd,
            'ch2': dummy_psd
        }

        self.last_metric_time = 0
        self.METRIC_UPDATE_INTERVAL = 0.1 # 备用，主要使用帧计数
        self.current_quality = 98.0
        
        # === 新增策略参数 ===
        # 延迟 0.35s 以避开最新的眼电（假设眨眼持续 0.2-0.4s）
        self.calc_delay_seconds = 0.35 
        self.calc_interval_frames = 4  # 每 4 帧计算一次指标 (约 130ms)
        self.frame_counter = 0

    def process(self, new_chunk, reliability_mask=None):
        current_time = time.time()
        n_points = new_chunk.shape[1]
        self.frame_counter += 1

        # 计算当前块的“平均可信度”
        if reliability_mask is not None:
            chunk_confidence = np.mean(reliability_mask)
        else:
            chunk_confidence = 1.0

        # 1. 实时滤波
        clean_ch1, self.zi_ch1 = signal.sosfilt(self.sos, new_chunk[0], zi=self.zi_ch1)
        clean_ch2, self.zi_ch2 = signal.sosfilt(self.sos, new_chunk[1], zi=self.zi_ch2)
        clean_ch1 = np.nan_to_num(clean_ch1)
        clean_ch2 = np.nan_to_num(clean_ch2)
        clean_chunk = np.vstack([clean_ch1, clean_ch2])

        # 2. 眨眼检测 (用于质量评分)
        blink_detected = np.any(np.abs(new_chunk) > 300)
        if blink_detected:
            self.current_quality = max(65.0, self.current_quality - 5.0)
        else:
            self.current_quality = min(99.0, self.current_quality + 0.5)

        # 3. 更新 Buffer
        self.raw_buffer = np.roll(self.raw_buffer, -n_points, axis=1)
        self.raw_buffer[:, -n_points:] = clean_chunk

        # 4. 指标计算 (降频 + 延迟策略)
        # 每 calc_interval_frames 帧计算一次，且仅当 buffer 填满时
        should_update_metrics = (self.frame_counter % self.calc_interval_frames == 0)
        
        if should_update_metrics:
            self._compute_metrics_logic(chunk_confidence)
            self.last_metric_time = current_time

        # 5. 打包输出
        step = 4
        vis_chunk_ch1 = clean_ch1[::step].tolist()
        vis_chunk_ch2 = clean_ch2[::step].tolist()

        output = {
            "timestamp": current_time,
            "metrics_updated": should_update_metrics,
            "signals": {
                "fp1": {"chunk_clean": vis_chunk_ch1},
                "fp2": {"chunk_clean": vis_chunk_ch2}
            },
            "status": {
                "blink": bool(blink_detected),
                "quality_val": int(self.current_quality)
            },
            "metrics": {
                "sensory_attention": round(self.metrics_ema['sensory_attention'], 2),
                "emotional_valence": round(self.metrics_ema['emotional_valence'], 2),
                "emotional_arousal": round(self.metrics_ema['emotional_arousal'], 2),

                "neural_relaxation": round(self.metrics_ema['neural_relaxation'], 2),
                "cortical_arousal": round(self.metrics_ema['cortical_arousal'], 2),
                "cognitive_depth": round(self.metrics_ema['cognitive_depth'], 2),

                "immersion_ratio": round(self.metrics_ema['immersion_ratio'], 2),
                "slope": round(self.metrics_ema['slope'], 2),
            },
            "psd_info": self.latest_psd_data,
            "powers": {
                "left": self.metrics_ema['detailed_powers_L'],
                "right": self.metrics_ema['detailed_powers_R']
            }
        }

        return output

    def _get_longest_clean_segment(self, data, threshold=300):
        """
        在给定的数据段中，寻找最长的一段连续无伪迹数据。
        如果整段数据都很脏，返回空数组。
        """
        n_points = data.shape[1]
        # 1. 标记坏点 (任意通道超过阈值即为坏点)
        is_bad = np.any(np.abs(data) > threshold, axis=0) # (N,)
        
        # 2. 如果全干净
        if not np.any(is_bad):
            return data
            
        # 3. 寻找最长连续 False (Clean) 区域
        # 前后补 True (Bad) 以便 diff 检测边缘
        padded = np.concatenate(([True], is_bad, [True]))
        diff = np.diff(padded.astype(int))
        
        # diff == -1 表示从 Bad 变 Clean (Start)
        starts = np.where(diff == -1)[0]
        # diff == 1 表示从 Clean 变 Bad (End)
        ends = np.where(diff == 1)[0]
        
        if len(starts) == 0:
            return np.empty((2, 0))
            
        lengths = ends - starts
        best_idx = np.argmax(lengths)
        
        best_start = starts[best_idx]
        best_end = ends[best_idx]
        
        return data[:, best_start:best_end]

    def _compute_metrics_logic(self, confidence_score=1.0):
        # === 策略：延迟窗口 + 剔除伪迹 ===
        
        # 1. 截取延迟窗口 (排除最近的 delay_seconds 数据)
        delay_samples = int(self.calc_delay_seconds * self.fs)
        if delay_samples < self.buffer_len:
            analysis_window = self.raw_buffer[:, :-delay_samples]
        else:
            analysis_window = self.raw_buffer # Fallback
            
        # 2. 获取最长纯净片段 (去除眼电)
        # 阈值设为 300uV (与眨眼检测一致)
        clean_data = self._get_longest_clean_segment(analysis_window, threshold=300)
        
        # 3. 如果可用数据太短 (< 1.0s)，则放弃本次更新，保持指标不变 (Stability)
        if clean_data.shape[1] < 1.0 * self.fs:
            # print("Artifact too heavy, skipping metric update.")
            return

        # 4. 预处理 (去趋势 + 加窗)
        data = signal.detrend(clean_data, axis=1)
        # 使用与数据长度匹配的窗函数
        curr_len = data.shape[1]
        window = signal.windows.hann(curr_len)
        data_windowed = data * window

        # === 5. FFT 计算 PSD ===
        # 注意：频率分辨率取决于 curr_len
        fft_vals = np.fft.rfft(data_windowed, axis=1)
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(curr_len, d=1 / self.fs)

        # 更新 PSD 数据 (1-60Hz)
        mask_vis = (freqs >= 1) & (freqs <= 85)
        vis_freqs = freqs[mask_vis][::2]
        
        # 仅当有足够频率点时更新 PSD 显示
        if len(vis_freqs) > 5:
            self.latest_psd_data = {
                'freqs': vis_freqs.tolist(),
                'ch1': psd[0, mask_vis][::2].tolist(),
                'ch2': psd[1, mask_vis][::2].tolist()
            }

        avg_psd = np.mean(psd, axis=0)
        total_avg_p = np.sum(avg_psd) + 1e-10

        # === 6. 提取各频段功率 ===
        band_powers = {0: {}, 1: {}}
        channel_total_power = {0: 1e-6, 1: 1e-6}

        for ch in range(2):
            idx_total = (freqs >= 1) & (freqs < 60)
            if np.any(idx_total):
                channel_total_power[ch] = np.sum(psd[ch, idx_total])

            for b_name, (low, high) in self.BANDS.items():
                idx = (freqs >= low) & (freqs < high)
                val = np.sum(psd[ch, idx]) if np.any(idx) else 0.0
                band_powers[ch][b_name] = val

        # === [辅助函数] 鲁棒的不对称性计算 ===
        def calculate_robust_asymmetry(l_val, r_val):
            log_diff  = abs(l_val - r_val) / (l_val +  r_val + 1e-9)
            return log_diff *2- 1

        # === 更新雷达图数据 ===
        keys = ['delta', 'theta', 'low_alpha', 'high_alpha', 'low_beta', 'high_beta', 'low_gamma', 'high_gamma']
        radar_alpha = max(0.01, 0.1 * confidence_score)

        for rb in keys:
            lookup = rb
            if rb == 'low_alpha': lookup = 'alpha_low'
            if rb == 'high_alpha': lookup = 'alpha_high'
            if rb == 'low_beta': lookup = 'beta'
            if rb == 'high_beta': lookup = 'beta'
            if rb == 'low_gamma': lookup = 'gamma_high'
            if rb == 'high_gamma': lookup = 'gamma_high'

            # 左通道
            p_val_L = band_powers[0].get(lookup, 0)
            rel_val_L = p_val_L / (total_avg_p * 0.1 + 1e-6)
            old_L = self.metrics_ema['detailed_powers_L'].get(rb, 0)
            self.metrics_ema['detailed_powers_L'][rb] = old_L * (1 - radar_alpha) + rel_val_L * radar_alpha

            # 右通道
            p_val_R = band_powers[1].get(lookup, 0)
            rel_val_R = p_val_R / (total_avg_p * 0.1 + 1e-6)
            old_R = self.metrics_ema['detailed_powers_R'].get(rb, 0)
            self.metrics_ema['detailed_powers_R'][rb] = old_R * (1 - radar_alpha) + rel_val_R * radar_alpha

        # === 7. 计算核心指标 ===

        # A. 感官注意
        _A_raw_sensory = calculate_robust_asymmetry(band_powers[0]['alpha_low'], band_powers[1]['alpha_low'])

        # B. 情绪愉悦
        l_rel = band_powers[0]['alpha_high'] / (channel_total_power[0] + 1e-20)
        r_rel = band_powers[1]['alpha_high'] / (channel_total_power[1] + 1e-20)
        _B_raw_valence = (l_rel + r_rel  ) /2 * 10

        # C. 综合体验
        _C_raw_comp = calculate_robust_asymmetry(band_powers[0]['alpha_total'], band_powers[1]['alpha_total'])

        # D. 神经松弛
        avg_alpha = (band_powers[0]['alpha_total'] + band_powers[1]['alpha_total']) / 2
        _D_raw_relax = np.log10(avg_alpha + 1e-10)

        # E. 皮层唤醒
        avg_beta = (band_powers[0]['alpha_total'] + band_powers[1]['alpha_total']) / 2
        _E_raw_arousal = np.log10(avg_beta + 1e-10)

        # F. 认知深度
        avg_gamma_h = (band_powers[0]['gamma_high'] + band_powers[1]['gamma_high']) / 2
        _F_raw_arousal = np.log10(avg_gamma_h + 1e-10)

        # G. 沉浸度
        avg_theta = (band_powers[0]['theta'] + band_powers[1]['theta']) / 2
        _G_raw_immersion = avg_beta / (avg_theta + 1e-10)

        # H. 频谱斜率
        idx_slope = (freqs >= 3) & (freqs <= 60)
        if np.sum(idx_slope) > 5:
            x_log = np.log10(freqs[idx_slope] + 1e-10)
            y_log = np.log10(avg_psd[idx_slope] + 1e-10)
            slope, _ = np.polyfit(x_log, y_log, 1)
        else:
            slope = -1.0

        # === 8. EMA 更新 ===
        BASE_ALPHA = 0.15
        effective_alpha = max(0.01, BASE_ALPHA * confidence_score)
        if confidence_score < 0.99:
            effective_alpha = effective_alpha * 0.00001

        def update_ema(key, val, norm=False, scale=1.0):
            d_val = val
            if norm:
                if key in ['sensory_attention',
                           'emotional_arousal',
                           'emotional_valence'
                           ]:
                    d_val = (np.clip(val, -1, 1) + 1) * 50
                else:
                    d_val = val * scale * 10

            old_val = self.metrics_ema.get(key, d_val)
            self.metrics_ema[key] = old_val + effective_alpha * (d_val - old_val)

        update_ema('sensory_attention', _A_raw_sensory, norm=True)
        update_ema('emotional_valence', _B_raw_valence, norm=True)
        update_ema('emotional_arousal', _C_raw_comp, norm=True)

        update_ema('neural_relaxation', _D_raw_relax, norm=True, scale=1)
        update_ema('cortical_arousal', _E_raw_arousal, norm=True, scale=1.0)
        update_ema('cognitive_depth', _F_raw_arousal, norm=True, scale=1.0)

        update_ema('immersion_ratio', _G_raw_immersion)
        update_ema('slope', slope)