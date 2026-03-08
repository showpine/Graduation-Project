import numpy as np
import tensorflow as tf
import sionna
import sionna.phy

class WirelessLink:
    """无线链路仿真类"""
    def __init__(self, precision=16, n=192, coderate=0.5):
        self.precision = precision
        self.k = 6 * precision  # 6个关节角度，每个16位
        self.n = n
        self.coderate = coderate
        
        # 初始化Sionna模块
        self.constellation = sionna.phy.mapping.Constellation("qam", num_bits_per_symbol=2)
        self.mapper = sionna.phy.mapping.Mapper(constellation=self.constellation)
        self.demapper = sionna.phy.mapping.Demapper("app", constellation=self.constellation)
        self.awgn_channel = sionna.phy.channel.AWGN()
        # 5G LDPC编码器/译码器
        self.encoder = sionna.phy.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.decoder = sionna.phy.fec.ldpc.LDPC5GDecoder(self.encoder, hard_out=True)
    
    def transmit(self, joint_angles, ebno_db=10.0):
        """发送关节角度通过无线链路"""
        # 将关节角度转换为比特流
        bits = self._joints_to_bits(joint_angles)
        
        # 计算噪声功率
        no = sionna.phy.utils.ebnodb2no(
            ebno_db,
            num_bits_per_symbol=2,
            coderate=self.coderate
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
        
        return joint_angles_hat, ber
    
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
        bits_np = bits.numpy()[0]
        # 将浮点数转换为整数（0或1）
        bits_int = bits_np.astype(int)
        # 将整数转换为二进制字符串
        bit_string = ''.join(map(str, bits_int))
        # 每precision位分割为一个浮点数
        joint_angles = []
        for i in range(0, min(len(bit_string), 6 * self.precision), self.precision):
            if i + self.precision <= len(bit_string):
                bit_chunk = bit_string[i:i+self.precision]
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