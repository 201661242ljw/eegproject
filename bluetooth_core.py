# bluetooth_core.py
# 修复版 v7：稳健混合插值 + 可信度掩码 (Reliability Mask)
# 解决：长丢包导致的滤波器震荡 & 指标跳变问题

import asyncio
import numpy as np
from bleak import BleakClient, BleakScanner
import threading
import queue
import time
from scipy import signal
from scipy.interpolate import CubicSpline

# ==========================================
# [配置]
# ==========================================
DEVICE_NAME = "RFstar_7FEA"
NOTIFY_UUID = "6e400003-b5a3-f393-e0a9-e50e24dcca9e"
WRITE_UUID = "6e400002-b5a3-f393-e0a9-e50e24dcca9e"
START_COMMAND = bytearray([0xA6, 0x35, 0x74, 0x73, 0x74, 0x61, 0x72, 0x74, 0x87, 0x8E])


class EEGBluetoothReceiver:
    def __init__(self):
        self.buffer = np.zeros((2, 1000 * 5))
        self.connected = False
        self.packet_count = 0
        self.latest_sample = [0, 0]

        # 统计信息
        self.bytes_received = 0
        self.packets_this_second = 0
        self.last_print_time = time.time()
        self.drop_count = 0

        self.samples_parsed = 0
        self.samples_per_second = 0
        self.last_sample_count_time = time.time()

        # 队列
        self.data_queue = queue.Queue()
        self.gui_queue = queue.Queue()

        self.stream_buffer = bytearray()

        # 发送缓冲区
        self.batch_buffer = []
        # [新增] 同步记录数据的可信度 (1.0=真实, 0.01=插值)
        self.reliability_buffer = []
        self.BATCH_SIZE = 50

        self.max_retries = 3
        self.retry_count = 0

        # 丢包补偿相关变量
        self.last_packet_counter = -1
        self.history_buffer = []
        self.HISTORY_LEN = 4

        # 初始化滤波器
        self._init_filters()

        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()

    def _init_filters(self):
        """
        初始化滤波器及其状态
        """
        fs = 1000.0

        # 1. 50Hz 带阻滤波器
        self.bs_b, self.bs_a = signal.butter(4, [48.0 / (fs / 2), 52.0 / (fs / 2)], 'bandstop')
        self.zi_bs = list(signal.lfilter_zi(self.bs_b, self.bs_a) for _ in range(2))

        # 2. 0.5Hz 高通滤波器
        self.hp_b, self.hp_a = signal.butter(4, 4 / (fs / 2), 'high')
        self.zi_hp = list(signal.lfilter_zi(self.hp_b, self.hp_a) for _ in range(2))

        # 3. 100Hz 低通滤波器
        self.lp_b, self.lp_a = signal.butter(4, 85.0 / (fs / 2), 'low')
        self.zi_lp = list(signal.lfilter_zi(self.lp_b, self.lp_a) for _ in range(2))

        print("✅ 后端滤波器初始化完成 (Fs=1000Hz, Bandwidth=0.5-100Hz)")

    def _apply_realtime_filter(self, data_chunk):
        """对数据块进行连续滤波"""
        filtered_chunk = np.zeros_like(data_chunk)

        for ch in range(2):
            raw = data_chunk[ch, :]
            # 级联滤波
            out_bs, self.zi_bs[ch] = signal.lfilter(self.bs_b, self.bs_a, raw, zi=self.zi_bs[ch])
            out_hp, self.zi_hp[ch] = signal.lfilter(self.hp_b, self.hp_a, out_bs, zi=self.zi_hp[ch])
            out_lp, self.zi_lp[ch] = signal.lfilter(self.lp_b, self.lp_a, out_hp, zi=self.zi_lp[ch])
            filtered_chunk[ch, :] = out_lp

        return filtered_chunk

    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while self.retry_count < self.max_retries:
            try:
                loop.run_until_complete(self._connect_and_listen())
                break
            except Exception as e:
                print(f"❌ 连接失败 (尝试 {self.retry_count + 1}/{self.max_retries}): {e}")
                self.retry_count += 1
                time.sleep(2)

    async def _connect_and_listen(self):
        print(f"🔍 正在扫描设备: {DEVICE_NAME} ...")
        try:
            device = await BleakScanner.find_device_by_filter(
                lambda d, ad: d.name and DEVICE_NAME in d.name,
                timeout=15.0
            )
        except Exception as e:
            print(f"❌ 扫描失败: {e}")
            device = None

        if not device:
            print("❌ 未找到设备")
            return

        print(f"✅ 找到设备: {device.address}")

        try:
            async with BleakClient(device, timeout=30.0, disconnected_callback=self._on_disconnect) as client:
                self.connected = True
                self.retry_count = 0
                self.last_packet_counter = -1
                self.history_buffer = []  # 重置历史
                print("🎉 蓝牙连接成功！")

                await client.start_notify(NOTIFY_UUID, self._notification_handler)
                await client.write_gatt_char(WRITE_UUID, START_COMMAND)
                print("🚀 启动命令发送成功")

                while client.is_connected:
                    await asyncio.sleep(1.0)
                print("🔌 连接已断开")

        except Exception as e:
            print(f"❌ 连接错误: {e}")
            raise
        finally:
            self.connected = False

    def _on_disconnect(self, client):
        print("⚠️  设备已断开连接")
        self.connected = False

    def bytes_to_int24(self, high, mid, low):
        val = (high << 16) | (mid << 8) | low
        if val & 0x800000:
            val -= 0x1000000
        return val

    def adc_to_uv(self, val):
        return val * 2.0 * 4000.0 / (1 << 24) * 1000

    def _notification_handler(self, sender, data: bytearray):
        self.bytes_received += len(data)
        self.packets_this_second += 1
        self.stream_buffer.extend(data)

        # 统计打印
        current_time = time.time()
        if current_time - self.last_print_time >= 1.0:
            if self.packets_this_second > 0:
                print(
                    f"[📊 统计] 接收: {self.packets_this_second} pkts/s | {self.bytes_received} B/s | 丢弃错位包: {self.drop_count}")
            self.last_print_time = current_time
            self.bytes_received = 0
            self.packets_this_second = 0
            self.drop_count = 0

        # 流式解析
        while True:
            if len(self.stream_buffer) < 39:
                break
            header_index = self.stream_buffer.find(b'\xAA\xBB')

            if header_index == -1:
                break

            if header_index > 0:
                self.stream_buffer = self.stream_buffer[header_index:]

            if len(self.stream_buffer) < 39:
                break
            is_valid_packet = (
                    self.stream_buffer[30] == 0x00 and
                    self.stream_buffer[31] == 0x00 and
                    self.stream_buffer[32] == 0x00 and
                    self.stream_buffer[33] == 0x00
            )

            if not is_valid_packet:
                self.drop_count += 1
                self.stream_buffer = self.stream_buffer[1:]
                continue

            struct_data = self.stream_buffer[:39]

            try:
                self.stream_buffer = self.stream_buffer[39:]

                current_counter = struct_data[2]

                ch_idx_1 = 3
                ch_idx_2 = 5

                ch2_raw = self.bytes_to_int24(struct_data[ch_idx_1 * 3 + 3], struct_data[ch_idx_1 * 3 + 4],
                                              struct_data[ch_idx_1 * 3 + 5])
                ch1_raw = self.bytes_to_int24(struct_data[ch_idx_2 * 3 + 3], struct_data[ch_idx_2 * 3 + 4],
                                              struct_data[ch_idx_2 * 3 + 5])

                ch1_uv = self.adc_to_uv(ch1_raw)
                ch2_uv = self.adc_to_uv(ch2_raw)

                if abs(ch1_uv) > 250000 or abs(ch2_uv) > 250000:
                    self.drop_count += 1
                    continue

                current_sample = [ch1_uv, ch2_uv]
                # print(f"[数据] {current_counter} | {current_sample}")
                current_reliability = 1.0  # 默认真实数据可信度为 1.0

                # ==========================================
                # [核心升级] 智能丢包补偿 - 稳健混合策略 (Hybrid Interpolation)
                # ==========================================
                num_lost = 0
                if self.last_packet_counter != -1:
                    diff = (current_counter - self.last_packet_counter) % 256
                    if diff > 1:
                        num_lost = diff - 1

                        # 策略1: 严重丢包重置 (>1000点, 即1秒)
                        # 数据已断层，插值无意义
                        if num_lost > 1000:
                            print(f"⚠️ 严重丢包 ({num_lost} samples)，重置滤波器状态")
                            self._init_filters()
                            self.history_buffer = []

                        # 策略2: 微小缺口 (< 4点) -> 使用三次样条插值 (Cubic Spline)
                        # 仅处理解析误差导致的极短丢包，保持平滑
                        elif len(self.history_buffer) >= 3 and num_lost < 4:
                            hist_y = np.array(self.history_buffer[-3:])
                            curr_y = np.array(current_sample)

                            x_fit = [0, 1, 2, 3 + num_lost]
                            y_fit_ch1 = hist_y[:, 0].tolist() + [curr_y[0]]
                            y_fit_ch2 = hist_y[:, 1].tolist() + [curr_y[1]]

                            cs_ch1 = CubicSpline(x_fit, y_fit_ch1, bc_type='natural')
                            cs_ch2 = CubicSpline(x_fit, y_fit_ch2, bc_type='natural')

                            x_interp = np.arange(3, 3 + num_lost)
                            interp_ch1 = cs_ch1(x_interp)
                            interp_ch2 = cs_ch2(x_interp)

                            for i in range(len(x_interp)):
                                self.batch_buffer.append([interp_ch1[i], interp_ch2[i]])
                                self.reliability_buffer.append(0.5)  # 样条插值勉强可信

                        # 策略3: 真实蓝牙丢包 (>= 4点) -> 强制线性插值 (Linear)
                        # 蓝牙丢一个包就是6个点，因此所有蓝牙丢包都会走这里
                        # 使用直线连接，防止滤波器产生振铃效应
                        elif self.history_buffer:
                            last_sample = self.history_buffer[-1]
                            start_arr = np.array(last_sample)
                            end_arr = np.array(current_sample)

                            steps = num_lost + 2
                            interpolated = np.linspace(start_arr, end_arr, steps)
                            filling_points = interpolated[1:-1]

                            for pt in filling_points:
                                self.batch_buffer.append(pt.tolist())
                                self.reliability_buffer.append(0.01)  # 线性直线完全不可信

                self.last_packet_counter = current_counter

                # 更新历史缓冲区
                self.history_buffer.append(current_sample)
                if len(self.history_buffer) > 10:
                    self.history_buffer.pop(0)

                self.batch_buffer.append(current_sample)
                self.reliability_buffer.append(current_reliability)  # 放入真实点可信度

                if len(self.batch_buffer) >= self.BATCH_SIZE:
                    raw_batch_data = np.array(self.batch_buffer).T
                    reliability_batch = np.array(self.reliability_buffer)  # 转换为numpy数组

                    filtered_data = self._apply_realtime_filter(raw_batch_data)

                    try:
                        # [关键修改] 发送元组 (数据, 可信度)
                        packet = (filtered_data, reliability_batch)
                        self.gui_queue.put_nowait(packet)
                        self.data_queue.put_nowait(packet)
                    except queue.Full:
                        try:
                            self.gui_queue.get_nowait()
                            self.gui_queue.put_nowait(packet)
                        except queue.Empty:
                            pass

                    self.batch_buffer = []
                    self.reliability_buffer = []

                self.samples_parsed += 1 + num_lost
                self.packet_count += 1

                if time.time() - self.last_sample_count_time >= 1.0:
                    self.samples_per_second = self.samples_parsed
                    self.samples_parsed = 0
                    self.last_sample_count_time = time.time()

            except Exception as e:
                print(f"⚠️ 数据处理异常: {e}")
                pass

    def get_connection_status(self):
        return self.connected

    def get_parsing_rate(self):
        return self.samples_per_second

    def get_gui_queue(self):
        return self.gui_queue


if __name__ == "__main__":
    receiver = EEGBluetoothReceiver()
    while True:
        time.sleep(1)