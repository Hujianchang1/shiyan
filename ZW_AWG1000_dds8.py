import socket
import numpy as np
import struct
import serial
import os
import time

g_ch_status = 0


class tcp_down_cmd:
    head = 0x18EFDC01
    cmd_type = 0x10
    ch = 1
    wave_point_cnt = 0x00000000
    da_default = 0x0000
    yuliu = np.zeros((4,), np.int8)
    wave_data = []
    yuliu1 = np.zeros((8,), np.int8)
    end1 = 0x01DCEF18
    end2 = 0x01DCEF18

    def __init__(self, ch, data):
        self.ch = ch
        if type(data[0]) == np.uint8:
            self.length = len(data)
        elif type(data[0]) == np.uint16:
            self.length = len(data) * 2
        else:
            print('error')
        self.wave_data = data

    def build(self):
        format_str = '!IBBIH4s' + str(self.length) + 's' + '8sII'
        ss = struct.pack(format_str, self.head, self.cmd_type, self.ch, self.wave_point_cnt, self.da_default,
                         self.yuliu.tobytes(), np.asarray(self.wave_data).tobytes(), self.yuliu1.tobytes(), self.end1,
                         self.end2)

        return ss


class dds_param_config_cmd:
    head = 0x18EFDC01
    cmd_type = 0x20
    ch = 1
    reserve = 0x0000
    dds_data = None
    reserve1 = 0x00000000
    end = 0x01DCEF18

    def __init__(self, ch, data):
        self.ch = ch
        self.dds_data = data

    def build(self):
        format_str = '!IBBH' + str(len(self.dds_data)) + 's' + 'I' + 'I'
        ss = struct.pack(format_str, self.head, self.cmd_type, self.ch,self.reserve, self.dds_data, self.reserve1,self.end)
        return ss

class markmode_set_cmd:
    head = 0x18EFDC01
    cmd_type = 0x17
    ch = 1
    mark_mode = 0x00
    trig_en = 0x00
    pulse_cnt = 0x00000000
    end = 0x01DCEF18

    def __init__(self):
        pass

    def build(self):
        format_str = '!IBBBBII'
        ss = struct.pack(format_str, self.head, self.cmd_type, self.ch, self.mark_mode, self.trig_en,
                         self.pulse_cnt, self.end)
        return ss

class tcp_open_cmd:
    head = 0x18EFDC01
    cmd_type = 2
    ch = 0x00
    yuliu = np.zeros((6,), np.int8)
    end = 0x01DCEF18

    def __init__(self, ch_list, operation='open'):
        global g_ch_status
        if operation == 'open':
            for i in ch_list:
                g_ch_status |= (1 << (i - 1))
        else:
            for i in ch_list:
                g_ch_status &= ~(1 << (i - 1))
        self.ch = g_ch_status
#         print(f'open ch = {self.ch}')

    def build(self):
        format_str = '!IBB6sI'
        ss = struct.pack(format_str, self.head, self.cmd_type, self.ch, self.yuliu.tobytes(), self.end)
        return ss


class uart_set_ip_cmd:
    head = 0x55
    cmd_type = 2
    ip = []
    yuliu = 0
    end = 0xaa

    def __init__(self, ip=''):
        ip_list = ip.split('.')
        for i in ip_list:
            self.ip.append(int(i))

    def build(self):
        return struct.pack('!BB4sBB', self.head, self.cmd_type, np.asarray(self.ip, np.uint8).tobytes(),
                           self.yuliu, self.end)


class tcp_data_source_cmd:
    head = 0x18EFDC01
    cmd_type = 3
    mode = 0
    ftw = 0
    yuliu = np.zeros((3,), np.int8)
    end = 0x01DCEF18

    def __init__(self, ch, ftw, mode=0):
        self.ch = ch
        self.mode = mode
        self.ftw = ftw

    def build(self):
        format_str = '!IBBBH3sI'
        ss = struct.pack(format_str, self.head, self.cmd_type, self.ch, self.mode, self.ftw, self.yuliu.tobytes(), self.end)
        return ss


class tcp_ref_switch_cmd:
    head = 0x18EFDC01
    cmd_type = 4
    ref = 0x00
    clk = None
    yuliu = np.zeros((5,), np.int8)
    end = 0x01DCEF18

    def __init__(self, ref: str, freq):
        if ref == 'ext_ref':
            self.ref = 0x01
        elif ref == 'int_ref':
            self.ref = 0x00
        else:
            print(f'input error')
        if freq == 100:
            self.clk = 0
        elif freq == 10:
            self.clk = 1
        else:
            print(f'input error')

    def build(self):
        format_str = '!IBBB5sI'
        ss = struct.pack(format_str, self.head, self.cmd_type, self.ref, self.clk,
                         self.yuliu.tobytes(), self.end)
        return ss


class tcp_attenuation_cmd:
    head = 0x18EFDC01
    cmd_type = 5
    ch = 0x01
    atten = 0x00
    yuliu = np.zeros((5,), np.int8)
    end = 0x01DCEF18

    def __init__(self, ch, db):
        self.ch = ch
        self.atten = db

    def build(self):
        return struct.pack('!IBBB5sI', self.head, self.cmd_type, self.ch, self.atten, self.yuliu.tobytes(), self.end)


class tcp_output_mode_cmd:
    head = 0x18EFDC01
    cmd_type = 6
    ch = 0x01
    mode = 0x00
    yuliu = np.zeros((5,), np.int8)
    end = 0x01DCEF18

    def __init__(self, ch: int, mode: str):
        self.ch = ch
        if mode == "AC":
            self.mode = 0x00
        elif mode == "DC":
            self.mode = 0x01
        else:
            pass

    def build(self):
        return struct.pack('!IBBB5sI', self.head, self.cmd_type, self.ch, self.mode, self.yuliu.tobytes(), self.end)


class tcp_ch_tiggerselect_cmd:
    head = 0x18EFDC01
    cmd_type = 7
    ch = 0x00
    ch_trigger = 0x00
    trigger_delay_cu = 0x00000000
    trigger_delay_xi = 0x00
    markout_delay = 0x00000000
    yuliu = np.zeros((12,), np.int8)
    end = 0x01DCEF18

    def __init__(self):
        pass

    def build(self):
        return struct.pack('!IBBBIBI12sI', self.head, self.cmd_type, self.ch,
                           self.ch_trigger, self.trigger_delay_cu, self.trigger_delay_xi,
                           self.markout_delay, self.yuliu.tobytes(), self.end)


class tcp_fan_ctrl_cmd:
    head = 0x18EFDC01
    cmd_type = 8
    speed = 0x0000
    yuliu = np.zeros((5,), np.int8)
    end = 0x01DCEF18

    def __init__(self):
        pass

    def build(self):
        return struct.pack('!IBH5sI', self.head, self.cmd_type, self.speed, self.yuliu.tobytes(), self.end)


class tcp_get_temp_cmd:
    head = 0x18EFDC01
    cmd_type = 9
    yuliu = np.zeros((7,), np.int8)
    end = 0x01DCEF18

    def __init__(self):
        pass

    def build(self):
        return struct.pack('!IB7sI', self.head, self.cmd_type, self.yuliu.tobytes(), self.end)


class tcp_set_ip_cmd:
    head = 0x18EFDC01
    cmd_type = 0x11
    ip = []
    yuliu = np.zeros(3, np.uint8)
    end = 0x01DCEF18

    def __init__(self, ip=''):
        ip_list = ip.split('.')
        for i in ip_list:
            self.ip.append(int(i))

    def build(self):
        return struct.pack('!IB4s3sI', self.head, self.cmd_type, np.asarray(self.ip, np.uint8).tobytes(),
                           self.yuliu.tobytes(), self.end)


class tcp_da_default_cmd:
    head = 0x18EFDC01
    cmd_type = 0x12
    channel = 0
    default = 0x0000
    yuliu = np.zeros(4, np.uint8)
    end = 0x01DCEF18

    def __init__(self):
        pass

    def build(self):
        return struct.pack('!IBBH4sI', self.head, self.cmd_type, self.channel, self.default,
                           self.yuliu.tobytes(), self.end)


class tcp_dc_verify_query_cmd:
    head = 0x18EFDC01
    cmd_type = 0x14
    yuliu = np.zeros(7, np.uint8)
    end = 0x01DCEF18

    def __init__(self):
        pass

    def build(self):
        return struct.pack('!IB7sI', self.head, self.cmd_type, self.yuliu.tobytes(), self.end)


class tcp_dc_verify_write_cmd:
    head = 0x18EFDC01
    cmd_type = 0x15
    yuliu1 = np.zeros(3, np.uint8)
    dc_coe = []
    da_delay = []
    trig_delay = []
    yuliu2 = 0x00000000
    end = 0x01DCEF18

    def __init__(self):
        pass

    def build(self):
        return struct.pack('!IB3s32s16s16sII', self.head, self.cmd_type, self.yuliu1.tobytes(),
                           np.asarray(self.dc_coe, np.uint16).byteswap().tobytes(), np.asarray(self.da_delay, np.uint16).byteswap().tobytes(),
                           np.asarray(self.trig_delay, np.uint16).byteswap().tobytes(), self.yuliu2, self.end)


class tcp_da_odelay_cmd:
    head = 0x18EFDC01
    cmd_type = 0x16
    channel = 0
    dac_odelay = 0x0000
    trig_odelay = 0x0000
    yuliu = np.zeros(2, np.uint8)
    end = 0x01DCEF18

    def __init__(self):
        pass

    def build(self):
        return struct.pack('!IBBHH2sI', self.head, self.cmd_type, self.channel, self.dac_odelay,
                           self.trig_odelay, self.yuliu.tobytes(), self.end)
 

class AWG1000:
    veriy_dic = {
        1: [1, 0], 2: [1, 0],
        3: [1, 0], 4: [1, 0],
        5: [1, 0], 6: [1, 0],
        7: [1, 0], 8: [1, 0],
    }
    mode = np.zeros(8)

    def __init__(self):
        self.s = None
        self.u = None

    def __del__(self):
        if self.s is not None:
            self.s.close()

    def connect(self, ip: str, port: int):
        """
        以太网连接设备
        :param ip: IP地址
        :param port: 端口号
        :return:
        """
        self.s = socket.socket()
        self.s.connect((ip, port))
        if port == 9001 :
            temptup = self._dc_verify_query()
            temp1 = []
            temp2 = []
            for i in range(len(temptup[5]) // 2):
                temp1.append(temptup[5][2 * i] << 8 | temptup[5][2 * i + 1])
                temp2.append(temptup[6][2 * i] << 8 | temptup[6][2 * i + 1])
            print(f'设备初始化完成...')           

    def disconnect(self):
        """
        以太网断开连接
        :return:
        """
        if self.s is not None:
            self.s.close()

    def open_uart(self, com: str):
        """
        打开串口
        :param com: 串口号
        :return:
        """
        try:
            self.u = serial.Serial(com, 115200, stopbits=1, bytesize=8, parity='N', timeout=5)
        except serial.SerialException:
            print("Is com port being used by other application?")

    def close_uart(self):
        """
        关闭串口
        :return:
        """
        if self.u is not None:
            self.u.close()

    def get_ack_status(self, x=0):
        if x == 0:
            msg = self.s.recv(16)
        else:
            msg = self.u.read(8)
            return True
        format_str = '!IBB6sI'
        head, type, ack, yuliu, end = struct.unpack(format_str, msg)
        if ack == 0xAA:
            return True
        else:
            return False

    def change_dev_ip(self, mode: str, ip: str):
        """
        以太网或着串口方式更改设备IP
        :param mode: "eth":以太网 ”uart“:串口
        :param ip: ip地址，eg:"192.168.1.10"
        :return:
        """
        iplist = ip.split('.')
        intvalue = [int(i) for i in iplist]
        if intvalue[3] == 255:
            print(f'input ip is not 255')
            return False
        if mode == "eth":
            cmd = tcp_set_ip_cmd(ip)
            if self.s is not None:
                self.s.send(cmd.build())
                return self.get_ack_status()
            else:
                print(f"ethernet disconnect...")
        elif mode == "uart":
            cmd = uart_set_ip_cmd(ip)
            if self.u is not None:
                self.u.write(cmd.build())
                return self.get_ack_status(1)
            else:
                print(f"uart not open...")
        else:
            print(f"input param error...")
        return False

    def get_dev_ip(self):
        """
        通过串口获取设备IP
        :return: 返回设备IP
        """
        ip = None
        frame = [0x55, 0x01, 0, 0, 0, 0, 0, 0xAA]
        if self.u is None:
            print(f"uart not open...")
            return
        self.u.write(frame)
        try:
            msg = self.u.read(8)
        except serial.SerialTimeoutException:
            print(f"serial read timeout...")
            return
        if len(msg) == 8:
            ip = f'{msg[2]}.{msg[3]}.{msg[4]}.{msg[5]}'
        return ip

    def open_channel(self, ch: list):
        """
        打开通道
        :param ch:需要打开的通道列表， eg:[1,5]
        :return:返回执行状态（bool）
        """
        cmd = tcp_open_cmd(ch)
        self.s.send(cmd.build())
        return self.get_ack_status()

    def close_channel(self, ch: list):
        """
        关闭通道
        :param ch:需要关闭的通道列表， eg:[1,5]
        :return:返回执行状态
        """
        cmd = tcp_open_cmd(ch, 'close')
        self.s.send(cmd.build())
        return self.get_ack_status()

    def set_refclk(self, ref_config: str, freq_config: int):
        """
        参考钟设置
        :param ref_config: ”int_ref“:内参考 ”ext_ref“:外参考
        :param freq_config: 频率值（MHz）常用10MHz, 100MHz
        :return:
        """
        if ref_config != 'ext_ref' and ref_config != 'int_ref':
            print(f'input error')
            return
        if freq_config != 100 and freq_config != 10:
            print(f'input error')
            return
        cmd = tcp_ref_switch_cmd(ref_config, freq_config)
        self.s.send(cmd.build())
        return self.get_ack_status()
    
#     def send_wave(self, data, ch):
#         cmd = tcp_down_cmd(ch, data)
#         cmd.wave_point_cnt = len(data)
#         self.s.send(cmd.build())
        
    def send_wave(self, data, ch):
        cmd = tcp_down_cmd(ch, data)
        cmd.wave_point_cnt = len(data)
        
#         print(len(cmd.build()))
#         print(type(cmd.build()))
        mdata = cmd.build()
        psize = len(mdata)
        osize = 1024*1024*1024
        integer = psize // osize
        dec = psize % osize
        for i in range(0, integer, 1):
            self.s.send(mdata[i * osize:(i + 1) * osize])
        if dec != 0:
            self.s.send(mdata[integer * osize: integer * osize + dec])

    def send_waveform_file(self, path: str, ch: int):
        """
        发送波形二进制文件
        :param path:文件路径
        :param ch:通道
        :return:返回执行状态
        """
        fd = open(path, 'rb')
        osize = 1024*1024*1024
        fsize = os.path.getsize(path)
        integer = fsize // osize
        dec = fsize % osize
        for i in range(0, integer, 1):
            self.s.send(fd.read(osize))
            time.sleep(1)
        if dec != 0:
            self.s.send(fd.read(dec))
        
#         self.s.send(np.fromfile(fd, np.uint8))
        return self.get_ack_status()

    def send_waveform_data(self, data, ch: int):
        """
        波形数据下发
        :param data:下发数据
        :param ch:通道1-8
        :return:返回执行状态
        """
        temp = (0 - self.veriy_dic[ch][1]) / self.veriy_dic[ch][0]
        default = round((temp * (pow(2, 15) - 1) + pow(2, 16)) % pow(2, 16))
#         print(self.mode)
        yushu = len(data) % 16
        if yushu:
            cha = 16 - yushu
            if self.mode[ch - 1] == 0:
                a = np.zeros(cha, dtype=np.int8)
                data = np.append(data, a)
            else:
                a = np.zeros(cha, dtype=np.int16)
                a += default
#                 print(a)
                data = np.append(data, a)
#         array = np.asarray(data).clip(-1, 1)
#         point = data * (2 ** 15 - 1)
        u16point = np.asarray(data, dtype=np.uint16).byteswap()
        self.send_wave(u16point, ch)
        return self.get_ack_status()

    def set_trigger_param(self, ch: int, trig_ch: int, trig_delay: int, mark_delay: int):
        """
        设置触发参数
        :param ch:通道[1,8]
        :param trig_ch:触发通道[0,8]
        :param trig_delay:触发延时:416ps*trig_delay = ?
        :param mark_delay:3.3ns * mark_delay = ?
        :return: 返回指令执行结果
        """
        assert 1 <= ch <= 8, 'input channel error[1,8]'
        assert 0 <= trig_ch <= 8, 'input triger channel error[0,8]'
        cmd = tcp_ch_tiggerselect_cmd()
        cmd.ch = ch
        if trig_ch == 0:
            cmd.ch_trigger = 9
        else:
            cmd.ch_trigger = trig_ch
        zheng = trig_delay // 16
        yu = trig_delay % 16
        cmd.trigger_delay_cu = zheng
        cmd.trigger_delay_xi = yu
        cmd.markout_delay = mark_delay
        self.s.send(cmd.build())
        return self.get_ack_status()
    
    def set_marktrig(self, ch=9,mark_mode=0, pulse_cnt=0, trig_en = 0):
        """
        设置触发参数
        :param ch:通道[1,8]
        :param mark_mode:mark模式[0,1] 0：mark模式 1:方波输出模式
        :param pulse_cnt:方波高电平时间0~2**32 
        :param trig_en:方波脉冲输出 0 不使能 1使能
        :return: 返回指令执行结果
        """
        assert 1 <= ch <= 9, 'input channel error[1,8]'
        assert 0 <= mark_mode <= 1, 'input triger channel error[0,1]'
        assert 0 <= trig_en <= 1, 'input triger channel error[0,1]'
        cmd = markmode_set_cmd()
        cmd.ch = ch
        cmd.mark_mode = mark_mode
        cmd.pulse_cnt = pulse_cnt
        cmd.trig_en = trig_en
        self.s.send(cmd.build())
        return self.get_ack_status()
    
    def set_fan_speed(self, speed: int):
        """
        设置风扇转速
        :param speed:转速区间0-400
        :return:
        """
        assert 0 <= speed <= 9999, 'input speed error[0, 400]'

        cmd = tcp_fan_ctrl_cmd()
        cmd.speed = speed
        self.s.send(cmd.build())
        return self.get_ack_status()

    def get_temperature(self):
        """
        获取设备温度
        :return: 返回4组温度值
        """
        cmd = tcp_get_temp_cmd()
        self.s.send(cmd.build())
        msg = self.s.recv(24)
        a, b, c, t1, t2, t3, t4, d, e = struct.unpack('IBBHHHH6sI', msg)

        t5 = ((t1 >> 8) & 0xff) | ((t1 & 0xff) << 8)
        t6 = ((t2 >> 8) & 0xff) | ((t2 & 0xff) << 8)
        t7 = ((t3 >> 8) & 0xff) | ((t3 & 0xff) << 8)
        t8 = ((t4 >> 8) & 0xff) | ((t4 & 0xff) << 8)

        temp1 = t5 * 503 / pow(2, 16) - 273
        temp2 = t6 * 503 / pow(2, 16) - 273
        temp3 = t7 * 503 / pow(2, 16) - 273
        temp4 = t8 * 503 / pow(2, 16) - 273
        # print(f'{temp1} {temp2} {temp3} {temp4}')
        list = [temp1, temp2, temp3, temp4]
        return list

    def set_output_mode(self, ch: int, mode: str):
        """
        设置前端输出模式
        :param ch:通道
        :param mode:”AC“:AC模式输出 ”DC“:DC模式输出
        :return:
        """
        if mode == "AC":
            self.mode[ch - 1] = 0
            self.set_default_vbias(ch, self.veriy_dic[ch][1] )
        elif mode == "DC":
            self.mode[ch - 1] = 1
            self.set_default_vbias(ch, 0)
        else:
            print(f"input param error...")
        cmd = tcp_output_mode_cmd(ch, mode)
        self.s.send(cmd.build())
        return self.get_ack_status()

    def set_rf_atten(self, ch: int, attenuation: int):
        """
        设置衰减
        :param ch:通道
        :param attenuation:衰减0-31dB
        :return:
        """
        assert 0 <= attenuation <= 31, 'input power error[0, 31]'
        cmd = tcp_attenuation_cmd(ch, attenuation)
        self.s.send(cmd.build())
        
        return self.get_ack_status()

    def set_data_source(self, ch: int, freq: int, source: int):
        """
        设置数据源
        :param ch : 12 34 56 78 绑定只需设置1357
        :param freq:输出频率MHz
        :param source:0:波形下载，通过trig播放
                      1:无需下载自行产生freq
                      2:无需trig,反复输出下载的波形
                      3:dds模式
        :return:
        """
        if source == 1:
            fs_w = round(freq / 600 * pow(2, 16) / 8)
        else:
            fs_w = 0
        cmd = tcp_data_source_cmd(ch, fs_w, source)
        self.s.send(cmd.build())
        return self.get_ack_status()

    def set_default_vbias(self, ch, vbias):
        """
        设置DA输出默认值
        :param ch: 通道号
        :param vbias:
        :return:
        """
        assert 1 <= ch <= 8, 'input channel error[1, 8]'
        cmd = tcp_da_default_cmd()
        cmd.channel = ch
        temp = (vbias - self.veriy_dic[ch][1]) / self.veriy_dic[ch][0]
#         print(self.veriy_dic[ch][1])
#         print(self.veriy_dic[ch][0])
#        print(round(temp,4))
        cmd.default = round((temp * (pow(2, 15) - 1) + pow(2, 16)) % pow(2, 16))
        self.s.send(cmd.build())
        return self.get_ack_status()

    def _set_odelay(self, ch, dac, trig):
        """
        设置DA-ODELAY值
        :param ch: 通道号
        :param dac:dac_odelay trig:trig_odelay
        :return:
        """
        assert 1 <= ch <= 8, 'input channel error[1, 8]'
        cmd = tcp_da_odelay_cmd()
        cmd.channel = ch
        cmd.dac_odelay = dac
        cmd.trig_odelay = trig
        self.s.send(cmd.build())
        return self.get_ack_status()

    def _dc_verify_query(self):
        cmd = tcp_dc_verify_query_cmd()
        self.s.send(cmd.build())
        recvmsg = self.s.recv(80)
        msgtup = struct.unpack("!IBBH32s16s16sII", recvmsg)

        for i in range(len(msgtup[4]) // 4):
            for j in range(2):
                # self.veriy_dic[i+1][j] = np.int16((msgtup[4][4*i+2*j] << 8 | msgtup[4][4*i+2*j + 1]))/10000
                x = (msgtup[4][4*i+2*j] << 8 | msgtup[4][4*i+2*j + 1])
                y = x.to_bytes(2,byteorder='little')
                z = struct.unpack('<h',y)
                self.veriy_dic[i+1][j] = (z[0] / 10000)
                
        
        print('设备DC通道校准系数：')
        print(self.veriy_dic)
        return msgtup

    def da_verify_write(self, verfiy: dict, da_odelay: list, trig_odelay: list):
        cmd = tcp_dc_verify_write_cmd()
        for i in range(8):
            for j in range(2):
                cmd.dc_coe.append(verfiy[i+1][j] * 10000)
        cmd.da_delay = da_odelay
        cmd.trig_delay = trig_odelay
        self.s.send(cmd.build())
        return self.get_ack_status()

    def sin_wave(self, A, f, phi, t):
        fs = 2.4e+9
        Ts = 1/fs
        n = t / Ts
        y = np.asarray(A*np.sin(2*np.pi*f*(np.asarray(np.arange(n), dtype=np.int32))*Ts + phi*(np.pi/180)) * (2 ** 15 - 1), dtype=np.int16)
        return y

    def pulsewave_gen(self, Vset, t, ch):
        """
        :params Vset:    [-2,2]单位V：
        :params t:    时间长度(秒)[1e-9,0.4]
        :params ch:   通道号[1:8] 注：因为每个通道有单独的校准系数，故需要设置通道号
        """
        fs = 2.4e+9
        A = (Vset - self.veriy_dic[ch][1]) / self.veriy_dic[ch][0]
        A = round((A * (pow(2, 15) - 1) + pow(2, 16)) % pow(2, 16))
        Ts = 1/fs
        n = round(t / Ts)
        list = []
        for i in range(n):
            list.append(A)
        return np.asarray(list)

    def square_wave(self, A, f, phi, t):
        """
        :params A:    振幅[1,-1]
        :params f:    信号频率(Hz)[1,1e9]
        :params phi:   相位[0,360]
        :params t:    时间长度(秒)[1e-9,0.4]
        """
        fs = 2.4e+9
        Ts = 1/fs
        n = t / Ts
        n = np.arange(n)
        x = A*np.sin(2*np.pi*f*n*Ts + phi*(np.pi/180))
        y = []
        for i in x:
            if np.sin(i) > 0:
                y.append(-1)
            else:
                y.append(1)
        return np.array(y)   

    def dds_param_config(self, ch, data: bytes):
        cmd = dds_param_config_cmd(ch, data)
        self.s.send(cmd.build())
        return self.get_ack_status()


    ### 将描述字列表准换为bytes
    def dds_seqword_transf(self, seqword_x8):
        dds_num = len(seqword_x8)
        dds_seqword_len = np.zeros(8, dtype=np.uint16)
        byte_len = 0
        for i in range(dds_num):
            dds_seqword_len[i] = len(seqword_x8[i])
            byte_len += dds_seqword_len[i] * 32
            
        tx_data = np.zeros(byte_len, dtype=np.uint8)
        
        a = 0
        for i in range(dds_num):
            for j in range(dds_seqword_len[i]):         
                for k in range(4):
                    tx_data[a+j*32+k] = (seqword_x8[i][j][0] >> k*8) & 0xff 
                    
                for k in range(8):
                    tx_data[a+j*32+k+4] = (seqword_x8[i][j][1] >> k*8) & 0xff 
                    
                for k in range(2):
                    tx_data[a+j*32+k+12] = (seqword_x8[i][j][2] >> k*8) & 0xff 
                    
                for k in range(1):
                    tx_data[a+j*32+k+14] = (seqword_x8[i][j][3] >> k*8) & 0xff 
                    
                for k in range(4):
                    tx_data[a+j*32+k+15] = (seqword_x8[i][j][4] >> k*8) & 0xff                 
                    
                for k in range(1):
                    tx_data[a+j*32+k+19] = (seqword_x8[i][j][5] >> k*8) & 0xff                 
                    
                for k in range(8):
                    tx_data[a+j*32+k+20] = (seqword_x8[i][j][6] >> k*8) & 0xff                 
                    
                for k in range(4):
                    tx_data[a+j*32+k+28] = (seqword_x8[i][j][7] >> k*8) & 0xff                 
                    
            a += dds_seqword_len[i] * 32
            
        tx_bytes = dds_seqword_len.tobytes() + bytes(tx_data)
        return tx_bytes
    
    def DefWaveGen(self, CH : int, TrigDelay, WaveParameter):
        #将用户给定的参数转化为fpga需要的参数，再转为bytes下发
        '''
        CH: 通道编号，int型, 范围1~8。
    
        TrigDelay: 触发延迟列表，长度为1~8，对应任一通道的八路DD触发延迟样点，int型，范围0~2^40-1，对应时间长度为0~458s
        
        WaveParameter: 波形参数列表，长度为1~8，对应任一通道的八路DDS。即
                      WaveParameter = [Parameter1, Parameter2, ... Parameter8]
           
        Parameter为某一路DDS的参数列表，最多包含1024组参数
                 Parameter = [[Freq, Length, AMP, Phase, WinType, WinLength],
                                       ...
                              [Freq, Length, AMP, Phase, WinType, WinLength]
                             ] 
        
        Freq: 频率，int型，单位Hz，范围0~1.2e9
        Length: 样点长度，int型，范围24~2^40-1，对应时间长度为10ns~458s。当Length为0时表示无限长度
        AMP: 幅度，int型，范围0~65535 
        Phase: 相位偏移，float型，单位角度，范围正负360
        WinType: 窗函数类型。0：不加窗；1：sin；
        WinLength: 窗函数上升沿样点长度（下降沿相同），int型，范围0~2^39-1，对应时间长度为0~229s
        
        '''
        '''
        seqword_x8: 描述字列表，长度1~8，对应8路DDS
        seqword_x8 = [seqword_x1(1), ... seqword_x1(8)]
        
        seqword_x1: 某一路DDS的描述字列表，长度为对应的SEQLen
        seqword_x1 = [[Freq, Length, AMP, PhasePos,PHASE, WinType, WinLength, WinFreq],
                                  ...
                      [Freq, Length, AMP, PhasePos,PHASE, WinType, WinLength, WinFreq]
                    ]
        Freq: uint32
        Length: int64
        AMP: uint16
        PhasePos: int8
        Phase: uint32
        WinType: int8
        WinLength: int64
        WinFreq: uint32
        
        
        '''    
        
        fs = 2.4*1e9
        #8路DDS的描述字列表
        seqword_x8 = []
        usd_dds = len(WaveParameter)
        amp_all = 0
        amp_l = []
        
        
        for i in WaveParameter:
            amp_all += i[0][2]
            amp_l.append(i[0][2])
      
        for i in range(usd_dds):
            #1路DDS的描述字列表
            seqword_x1 = [] 
            #将trig_delay转化为第一个描述字
            try:
                data_len = TrigDelay[i]
                if data_len > 24:
                    new_list = [0, data_len, 0, 0, 0, 0, 0, 0]
                    seqword_x1.append(new_list)
            except:
                pass
            
            #8路DDS幅度归一化
            try:
                amp = int(amp_l[i] / amp_all * 65535)
            except ZeroDivisionError:
                amp = 0
            
            #将用户给定的参数转化为fpga需要的参数
            for param in WaveParameter[i]:
                freq = int(param[0] / fs * 2**32)
                data_len = int(param[1])
                amp = param[2]              
                phase_offest = param[3]
                if phase_offest >= 0:
                    pahse_pos = 0
                else:
                    pahse_pos = 1
                phase_offest = int(abs(phase_offest / 360 * 2**32))
                win_type = int(param[4])
                if data_len:
                    win_freq = int(2**31 / data_len)
                else:  #无限长度模式不支持加窗
                    win_freq = 0
                    win_type = 0
                win_len = param[5]
                if win_len == 0:
                    win_type = 0
                if win_len > data_len / 2:
                    win_len = int(data_len / 2)
                new_list = [freq, data_len, amp, pahse_pos, phase_offest, win_type, win_len, win_freq]
                seqword_x1.append(new_list)
            seqword_x8.append(seqword_x1)
        #转为bytes型
        tx_bytes = self.dds_seqword_transf(seqword_x8)
        self.dds_param_config(CH, tx_bytes)
