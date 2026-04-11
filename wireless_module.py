import numpy as np
import tensorflow as tf
import sionna
import sionna.phy

class WirelessLink:
    """无线链路仿真类"""
    def __init__(self, precision=16, coderate=0.5):
        self.precision = precision
        self.k = 6 * precision  # 固定 96 bit（6轴×16位）
        self.coderate = coderate  # 手动设：0.3 / 0.5 / 0.7
        self.num_bits_per_symbol = 2  # QPSK
        
        # 1. 按手动设的码率算 n
        n_raw = round(self.k / self.coderate)
        
        # 2. 保证 n 是 QPSK 整数倍（必须）
        self.n = ((n_raw + self.num_bits_per_symbol - 1) // self.num_bits_per_symbol) * self.num_bits_per_symbol
        
        # 3. 保证 n ≥ k
        self.n = max(self.k, self.n)
        
        # 实际码率
        self.actual_coderate = self.k / self.n
        
        # 调制 / 信道
        self.constellation = sionna.phy.mapping.Constellation("qam", num_bits_per_symbol=self.num_bits_per_symbol)
        self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation)
        self.awgn_channel = sionna.phy.channel.AWGN()
        
        # LDPC 编码器：用手动算的 k 和 n
        self.encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)
    
    def transmit(self, joint_angles, ebno_db=10.0, distance_km=10.0):
        """发送关节角度通过无线链路，返回关节角度、误码率和无线链路时延"""
        # 将关节角度转换为比特流
        bits = self._joints_to_bits(joint_angles)
        
        # 计算噪声功率 - 使用实际码率
        no = sionna.phy.utils.ebnodb2no(
            ebno_db,
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=self.actual_coderate
        )
        
        # 编码
        codewords = self.encoder(bits)
        # 调制
        x = self.mapper(codewords)
        # 通过AWGN信道
        y = self.awgn_channel(x, no)
        # 解调
        llr = self.demapper(y, no)
        # 译码
        bits_hat = self.decoder(llr)
        
        # 将比特流转换回关节角度
        joint_angles_hat = self._bits_to_joints(bits_hat)
        
        # 计算误码率
        ber = self._calculate_ber(bits, bits_hat)
        
        # 计算无线链路时延
        # 1. 物理层处理时延
        # 编码/解码时延 (假设处理速度为100 Mbps)
        processing_speed = 100e6  # 比特/秒
        coding_delay = (self.n * 2) / processing_speed  # 编码+解码
        
        # 调制/解调时延 (假设处理速度为100 Msymbols/s)
        symbol_speed = 100e6  # 符号/秒
        num_symbols = self.n / self.num_bits_per_symbol
        modulation_delay = (num_symbols * 2) / symbol_speed  # 调制+解调
        
        # 2. 传播时延 (光速)
        propagation_delay = distance_km / 300000  # 秒
        
        # 3. 传输时延 (发送时间)
        # 传输时延 (假设传输速率为50 Mbps，符合实际无线链路速率)
        transmission_rate = 50e6  # 比特/秒
        transmission_delay = self.n / transmission_rate
        
        # 总无线链路时延
        total_wireless_delay = coding_delay + modulation_delay + propagation_delay + transmission_delay
        
        return joint_angles_hat, ber, total_wireless_delay
    
    def _calculate_ber(self, bits, bits_hat):
        """计算误码率"""
        # 将TensorFlow张量转换为numpy数组
        bits_np = bits.numpy()[0]
        bits_hat_np = bits_hat.numpy()[0]
        
        # 计算错误比特数
        error_bits = np.sum(bits_np != bits_hat_np)
        # 计算总比特数
        total_bits = len(bits_np)
        # 计算误码率
        ber = error_bits / total_bits if total_bits > 0 else 0.0
        
        return ber
    
    def _joints_to_bits(self, joint_angles):
        """将关节角度转换为比特流"""
        # 取所有6个关节角度
        joint_angles_6 = joint_angles[:6]
        # 将关节角度转换为指定精度的浮点数
        float_array = np.array(joint_angles_6, dtype=f"float{self.precision}")
        # 将浮点数转换为整数
        int_array = float_array.view(f"uint{self.precision}")
        # 将整数转换为二进制字符串
        bit_strings = [np.binary_repr(x, width=self.precision) for x in int_array]
        # 拼接为一个长字符串
        bit_string = ''.join(bit_strings)
        # 转换为浮点数列表 [0.0, 1.0, ...]
        bit_list = [float(bit) for bit in bit_string]
        # 转换为TensorFlow张量并添加batch维度
        bits = tf.convert_to_tensor([bit_list], dtype=tf.float32)
        return bits
    
    def _bits_to_joints(self, bits):
        """将比特流转换回关节角度"""
        # 将TensorFlow张量转换为numpy数组
        bits_np = bits.numpy()
        # 展平张量，处理可能的多维形状
        bits_np = bits_np.flatten()
        # 将浮点数转换为整数（0或1）
        bits_int = bits_np.astype(int)
        # 将整数转换为二进制字符串
        bit_string = ''.join(map(str, bits_int))
        # 每precision位分割为一个浮点数
        joint_angles = []
        for i in range(0, min(len(bit_string), 6 * self.precision), self.precision):
            if i + self.precision <= len(bit_string):
                bit_chunk = bit_string[i:i+self.precision]
                # 确保bit_chunk只包含0和1
                bit_chunk = ''.join(c for c in bit_chunk if c in '01')
                if len(bit_chunk) == self.precision:
                    # 将二进制字符串转换为整数
                    int_val = int(bit_chunk, 2)
                    # 将整数转换为指定精度的浮点数
                    int_val = np.uint16(int_val)
                    float_val = int_val.view(f"float{self.precision}")
                    joint_angles.append(float_val)
        # 确保返回6个关节角度
        while len(joint_angles) < 6:
            joint_angles.append(0.0)
        return joint_angles[:6]

class AdvancedWirelessLink:
    """高级无线链路仿真类，支持3GPP CDL信道模型和OFDM"""
    def __init__(self, precision=16, cdl_model="C", speed=10.0, delay_spread=100e-9, 
                 bs_antennas=4, subcarrier_spacing=30e3, fft_size=76):
        self.precision = precision
        self.k = 6 * precision  # 6个关节角度，每个16位
        
        # 天线配置（固定UT天线为1，简化实现）
        self.NUM_UT = 1
        self.NUM_BS = 1
        self.NUM_UT_ANT = 1
        self.NUM_BS_ANT = bs_antennas
        self.NUM_STREAMS_PER_TX = 1
        
        # 3GPP CDL信道参数
        self.CARRIER_FREQUENCY = 2.6e9
        self.DELAY_SPREAD = delay_spread
        self.DIRECTION = "uplink"
        self.CDL_MODEL = cdl_model
        self.SPEED = speed
        
        # OFDM资源网格参数
        self.RESOURCE_GRID_PARAMS = {
            "num_ofdm_symbols": 14,
            "fft_size": fft_size,
            "subcarrier_spacing": subcarrier_spacing,
            "num_tx": self.NUM_UT,
            "num_streams_per_tx": self.NUM_STREAMS_PER_TX,
            "cyclic_prefix_length": 6,
            "pilot_pattern": "kronecker",
            "pilot_ofdm_symbol_indices": [2, 11]
        }
        
        # 编码调制参数
        self.NUM_BITS_PER_SYMBOL = 2  # QPSK
        self.CODERATE = 0.5  # LDPC码率0.5
        
        # 初始化模块
        self._initialize_modules()
    
    def _initialize_modules(self):
        """初始化Sionna模块"""
        # 流管理
        self.RX_TX_ASSOCIATION = np.array([[1]])
        self.STREAM_MANAGEMENT = sionna.phy.mimo.StreamManagement(
            rx_tx_association=self.RX_TX_ASSOCIATION,
            num_streams_per_tx=self.NUM_STREAMS_PER_TX
        )
        
        # OFDM资源网格
        self.RESOURCE_GRID = sionna.phy.ofdm.ResourceGrid(**self.RESOURCE_GRID_PARAMS)
        
        # 计算码块长度
        self.n = int(self.RESOURCE_GRID.num_data_symbols * self.NUM_BITS_PER_SYMBOL)
        # 确保码率不小于1/5
        min_k = int(self.n * 0.2)  # 最小码率1/5
        max_k = int(self.n * self.CODERATE)
        # 确保k在合理范围内
        self.k = max(min_k, max_k)  # 确保码率不小于1/5
        
        # 天线阵列（固定UT天线为1）
        self.UT_ARRAY = sionna.phy.channel.tr38901.Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=self.CARRIER_FREQUENCY
        )
        
        # BS天线阵列
        self.BS_ARRAY = sionna.phy.channel.tr38901.AntennaArray(
            num_rows=1,
            num_cols=int(self.NUM_BS_ANT / 2),
            polarization="dual",
            polarization_type="cross",
            antenna_pattern="38.901",
            carrier_frequency=self.CARRIER_FREQUENCY
        )
        
        # 3GPP CDL信道
        self.CDL_CHANNEL = sionna.phy.channel.tr38901.CDL(
            model=self.CDL_MODEL,
            delay_spread=self.DELAY_SPREAD,
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=self.UT_ARRAY,
            bs_array=self.BS_ARRAY,
            direction=self.DIRECTION,
            min_speed=self.SPEED
        )
        
        # 发射端模块
        self.encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.mapper = sionna.phy.mapping.Mapper(
            constellation_type="qam",
            num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL
        )
        self.rg_mapper = sionna.phy.ofdm.ResourceGridMapper(self.RESOURCE_GRID)
        
        # 信道模块
        self.channel = sionna.phy.channel.OFDMChannel(
            channel_model=self.CDL_CHANNEL,
            resource_grid=self.RESOURCE_GRID,
            add_awgn=True,
            normalize_channel=True,
            return_channel=True
        )
        
        # 接收端模块
        self.ls_est = sionna.phy.ofdm.LSChannelEstimator(
            resource_grid=self.RESOURCE_GRID,
            interpolation_type="nn"
        )
        self.lmmse_equ = sionna.phy.ofdm.LMMSEEqualizer(
            resource_grid=self.RESOURCE_GRID,
            stream_management=self.STREAM_MANAGEMENT
        )
        self.demapper = sionna.phy.mapping.Demapper(
            demapping_method="app",
            constellation_type="qam",
            num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL
        )
        self.decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(
            encoder=self.encoder,
            hard_out=True
        )
    
    def transmit(self, joint_angles, ebno_db=10.0, distance_km=10.0):
        """发送关节角度通过高级无线链路，返回关节角度、误码率和无线链路时延"""
        # 将关节角度转换为比特流
        bits = self._joints_to_bits(joint_angles)
        
        # 计算噪声功率
        no = sionna.phy.utils.ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=self.NUM_BITS_PER_SYMBOL,
            coderate=self.CODERATE,
            resource_grid=self.RESOURCE_GRID
        )
        
        # 编码
        codewords = self.encoder(bits)
        # 调制
        x = self.mapper(codewords)
        # 调整张量形状以匹配rg_mapper的要求: [batch_size, num_tx, num_streams, num_data_symbols]
        # 当前x形状: [batch_size, num_data_symbols]
        # 需要调整为: [batch_size, num_tx, num_streams_per_tx, num_data_symbols]
        x = tf.expand_dims(x, axis=1)  # 添加num_tx维度
        x = tf.expand_dims(x, axis=2)  # 添加num_streams维度
        # 复制数据到多个流（如果需要）
        if self.NUM_STREAMS_PER_TX > 1:
            # 对于多流情况，我们需要确保数据形状正确
            # 这里简化处理，复制相同的数据到所有流
            x = tf.repeat(x, self.NUM_STREAMS_PER_TX, axis=2)
        # 资源网格映射
        x_rg = self.rg_mapper(x)
        # 信道传输
        y, h_freq = self.channel(x_rg, no)
        # 信道估计
        h_hat, err_var = self.ls_est(y, no)
        # 均衡
        x_hat, no_eff = self.lmmse_equ(y, h_hat, err_var, no)
        # 解调
        llr = self.demapper(x_hat, no_eff)
        # 译码
        bits_hat = self.decoder(llr)
        
        # 将比特流转换回关节角度
        joint_angles_hat = self._bits_to_joints(bits_hat)
        
        # 计算误码率
        ber = self._calculate_ber(bits, bits_hat)
        
        # 计算无线链路时延
        # 1. 物理层处理时延
        # 编码/解码时延 (假设处理速度为100 Mbps)
        processing_speed = 100e6  # 比特/秒
        coding_delay = (self.n * 2) / processing_speed  # 编码+解码
        
        # 调制/解调时延 (假设处理速度为100 Msymbols/s)
        symbol_speed = 100e6  # 符号/秒
        num_symbols = self.n / self.NUM_BITS_PER_SYMBOL
        modulation_delay = (num_symbols * 2) / symbol_speed  # 调制+解调
        
        # OFDM处理时延 (基于FFT大小)
        fft_size = self.RESOURCE_GRID_PARAMS['fft_size']
        # 假设FFT处理速度为1GHz
        fft_speed = 1e9  # 操作/秒
        ofdm_delay = (fft_size * 2) / fft_speed  # FFT/IFFT操作
        
        # 2. 传播时延 (光速)
        propagation_delay = distance_km / 300000  # 秒
        
        # 3. 传输时延 (发送时间) - 基于50 Mbps带宽，符合实际无线链路速率
        transmission_rate = 50e6  # 比特/秒
        transmission_delay = self.n / transmission_rate
        
        # 总无线链路时延
        total_wireless_delay = coding_delay + modulation_delay + ofdm_delay + propagation_delay + transmission_delay
        
        return joint_angles_hat, ber, total_wireless_delay
    
    def _calculate_ber(self, bits, bits_hat):
        """计算误码率"""
        # 将TensorFlow张量转换为numpy数组
        bits_np = bits.numpy()[0]
        bits_hat_np = bits_hat.numpy()[0]
        
        # 计算错误比特数
        error_bits = np.sum(bits_np != bits_hat_np)
        # 计算总比特数
        total_bits = len(bits_np)
        # 计算误码率
        ber = error_bits / total_bits if total_bits > 0 else 0.0
        
        return ber
    
    def _joints_to_bits(self, joint_angles):
        """将关节角度转换为比特流"""
        # 取所有6个关节角度
        joint_angles_6 = joint_angles[:6]
        # 将关节角度转换为指定精度的浮点数
        float_array = np.array(joint_angles_6, dtype=f"float{self.precision}")
        # 将浮点数转换为整数
        int_array = float_array.view(f"uint{self.precision}")
        # 将整数转换为二进制字符串
        bit_strings = [np.binary_repr(x, width=self.precision) for x in int_array]
        # 拼接为一个长字符串
        bit_string = ''.join(bit_strings)
        # 转换为浮点数列表 [0.0, 1.0, ...]
        bit_list = [float(bit) for bit in bit_string]
        # 确保长度不超过k
        if len(bit_list) > self.k:
            bit_list = bit_list[:self.k]
        # 不足k位的补0
        while len(bit_list) < self.k:
            bit_list.append(0.0)
        # 转换为TensorFlow张量并添加batch维度
        bits = tf.convert_to_tensor([bit_list], dtype=tf.float32)
        return bits
    
    def _bits_to_joints(self, bits):
        """将比特流转换回关节角度"""
        # 将TensorFlow张量转换为numpy数组
        bits_np = bits.numpy()
        # 展平张量，处理可能的多维形状
        bits_np = bits_np.flatten()
        # 将浮点数转换为整数（0或1）
        bits_int = bits_np.astype(int)
        # 将整数转换为二进制字符串
        bit_string = ''.join(map(str, bits_int))
        # 每precision位分割为一个浮点数
        joint_angles = []
        for i in range(0, min(len(bit_string), 6 * self.precision), self.precision):
            if i + self.precision <= len(bit_string):
                bit_chunk = bit_string[i:i+self.precision]
                # 确保bit_chunk只包含0和1
                bit_chunk = ''.join(c for c in bit_chunk if c in '01')
                if len(bit_chunk) == self.precision:
                    # 将二进制字符串转换为整数
                    int_val = int(bit_chunk, 2)
                    # 将整数转换为指定精度的浮点数
                    int_val = np.uint16(int_val)
                    float_val = int_val.view(f"float{self.precision}")
                    joint_angles.append(float_val)
        # 确保返回6个关节角度
        while len(joint_angles) < 6:
            joint_angles.append(0.0)
        return joint_angles[:6]