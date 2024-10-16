import socket
import numpy as np
import struct
import time
from ctypes import c_int32
from enum import Enum
from qcodes import Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils.validators import Strings, Lists, Arrays, Bool, Numbers, Ints, Dict

class cmd_type(Enum):
    SET_VOL_CMD = 0x01
    SET_OPEN_CMD = 0x02
    SET_SLOPE_CMD = 0x03
    GET_CURRENT_CMD = 0x04
    SET_LED_CMD = 0x05
    SET_CHMODE_CMD = 0x06
    SET_DECID_CMD = 0x07
    SET_IP_CMD = 0x08
    SET_VERIFY_CMD = 0x09
    GET_CHSTAUE_CMD = 0x0A
    SET_HEART_CMD = 0x0B
    GET_TEMP_CMD = 0x0C
    SET_PWD_CMD = 0x0D
    SET_CHDIS_CMD = 0x0E
    GET_VOL_CMD = 0x0F
    SET_RESLOPE_CMD = 0x10
    GET_VOLDAC_CMD = 0x11
    GET_IP_CMD = 0x12
    SET_SEQUENCE_CMD = 0x13
    SET_SEQUENCET_CMD = 0x14
    GET_SEQUENCE_CMD = 0x15
    SET_SEQUENCE_POS_CMD = 0x16
    SET_SOFTTRIG_CMD = 0x17
    SET_GOLBALEN_CMD = 0x18
    SET_PLREG_CMD = 0x19
    GET_PLREG_CMD = 0x1A
    SET_WAVE_FILE = 0x1B
    GET_VERIFY_CMD = 0x1C
    GET_DEVINFO_CMD = 0xFE
    SET_DEVINFO_CMD = 0xFF


class cmd_base:
    head = 0xF7F6F5F4
    cmd = np.uint32(0)
    len = np.uint32(524)
    ad_da_cmd = []
    zero = []

    def __init__(self):
        pass

    def build(self):
        format_str = '!3I' + str(len(self.ad_da_cmd)) + 's'
        self.zero.clear()
        for i in range(524 - len(self.ad_da_cmd) - 12):
            self.zero.append(0)
        send_str = struct.pack(format_str, self.head, self.cmd, self.len,
                               np.asarray(self.ad_da_cmd, np.uint8).tobytes())
        send_str += np.asarray(self.zero, np.uint8).tobytes()
        return send_str

    def set_cmd(self, cmd_list=None):
        self.ad_da_cmd.clear()
        self.ad_da_cmd += cmd_list

class ip_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 11
    type = 0x08
    ip = []
    mask = []
    gateway = []
    crc = 0
    end = 0xaa

    def __init__(self, ip=[], mask=[], gw=[]):
        super().__init__()
        self.ip = ip
        self.mask = mask
        self.gateway = gw

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type]
        buffer += self.ip
        buffer += self.mask
        buffer += self.gateway
        buffer.append(self.crc)
        buffer.append(self.end)

        super().set_cmd(buffer)

        return super().build()
    
class vol_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = cmd_type.SET_VOL_CMD.value
    ch = None
    sequence = 0x00
    vol = []
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.ch, self.sequence]
        buffer += self.vol
        buffer.append(self.crc)
        buffer.append(self.end)
        super().set_cmd(buffer)
        return super().build()


class open_close_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = cmd_type.SET_OPEN_CMD.value
    ch = 0x00
    switch = 0x00
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.ch, self.switch, self.crc, self.end]
        super().set_cmd(buffer)
        return super().build()
    
class led_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = cmd_type.SET_LED_CMD.value
    ch = 0x00
    status = 0
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.ch, self.status, self.crc, self.end]
        super().set_cmd(buffer)
        return super().build()

class get_vol_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = cmd_type.GET_VOL_CMD.value
    ch = 0x00
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.ch, self.crc, self.end]
        super().set_cmd(buffer)
        return super().build()
    
class set_sequence_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = cmd_type.SET_SEQUENCE_CMD.value
    ch = 0x00
    sequence_mode = 0x00
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.ch, self.sequence_mode, self.crc, self.end]
        super().set_cmd(buffer)
        return super().build()
    
class slope_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = cmd_type.SET_SLOPE_CMD.value
    slope = 0x00000000
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.slope & 0xff, (self.slope >> 8) & 0xff,
                  (self.slope >> 16) & 0xff, (self.slope >> 24) & 0xff, self.crc, self.end]
        super().set_cmd(buffer)
        return super().build()
    
class status_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = cmd_type.GET_CHSTAUE_CMD.value
    ch = 0
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.ch, self.crc, self.end]
        super().set_cmd(buffer)
        return super().build()
    
class set_ch_dis(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = cmd_type.SET_CHDIS_CMD.value
    ch = 0
    ty = 0
    st = 0
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = []
        buffer += [self.hd, self.id, self.length, self.type]
        buffer += [self.ch, self.ty, self.st]
        buffer += [self.crc, self.end]
        super().set_cmd(buffer)
        return super().build()
    
class public_cmd:
    head = 0x01020304
    type = cmd_type.SET_WAVE_FILE.value
    data1 = np.uint32(0)
    data2 = np.uint32(0)
    data3 = np.uint32(0)
    data4 = np.uint32(0)

    def __init__(self):
        self.datan = np.zeros(500, dtype=np.uint8)

    def build(self):
        format = "<6I500s"
        remainder = 500 - len(self.datan)
        zeroarray = np.zeros(remainder,np.uint8)
        datan_temp = np.append(self.datan, zeroarray)
        sendstr = struct.pack(format, self.head, self.type, self.data1,self.data2,
                              self.data3,self.data4, datan_temp.tobytes())
        return sendstr
    
class set_sequence_total_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = cmd_type.SET_SEQUENCET_CMD.value
    ch = 0x00
    sequence_total = 0x00000000
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.ch, self.sequence_total & 0xff,
                  (self.sequence_total >> 8) & 0xff,
                  (self.sequence_total >> 16) & 0xff, (self.sequence_total >> 24) & 0xff, self.crc, self.end]
        super().set_cmd(buffer)
        return super().build()
    
class set_reg_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 3
    type = cmd_type.SET_PLREG_CMD.value
    base_address = 0
    offset = 0
    reg = 0

    def __init__(self):
        super().__init__()

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.base_address & 0xff,
                          (self.base_address >> 8) & 0xff, (self.base_address >> 16) & 0xff,
                          (self.base_address >> 24) & 0xff,
                          (self.offset >> 0) & 0xff, (self.offset >> 8) & 0xff, (self.offset >> 16) & 0xff,
                          (self.offset >> 24) & 0xff,
                          (self.reg >> 0) & 0xff, (self.reg >> 8) & 0xff, (self.reg >> 16) & 0xff,
                          (self.reg >> 24) & 0xff]
        super().set_cmd(buffer)
        return super().build()
    

class get_verify_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = cmd_type.GET_VERIFY_CMD.value
    ch = 0x00
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.ch, self.crc, self.end]
        super().set_cmd(buffer)
        return super().build()


class SingleChannel(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, channel_number, **kwargs):
        super().__init__(parent, name, **kwargs)
        self.channel_number = channel_number
        self.k = 1000
        self.verify_k = 1
        self.verify_b = 0
        self._get_verify()


        self.add_parameter('output_state',
                            label=f'Channel {channel_number} open',
                            set_cmd=self._set_output_state,
                            get_cmd=self._get_output_state,
                            vals=Strings(),
                            docstring='open or close this channel')
    
        self.add_parameter('voltage',
                            label=f'Channel {channel_number} Voltage',
                            unit='V',
                            set_cmd=self._set_voltage,
                            get_cmd=self._get_voltage,
                            vals=Numbers(-10,10),
                            docstring='get/set voltage for this channel')
        
        self.add_parameter('slope',
                            label=f'Channel {channel_number} slope',
                            unit='mv/s',
                            set_cmd=self._set_slope,
                            vals=Ints(10,10000000),
                            docstring='set slope for this channel')
        
        self.add_parameter('pwm', # 启用精度扩展
                    label=f'Channel {channel_number} pwm',
                    unit=None,
                    set_cmd=self._set_pwm,
                    vals=Bool(),
                    docstring='set pwm for this channel')
        
        self.add_parameter('sequence',
                    label=f'Channel {channel_number} vol sequence',
                    unit='V',
                    set_cmd=self._set_sequence,
                    vals=Lists(),
                    docstring='set vol sequence for this channel')
        
        self.add_parameter('seq_play_pos',
                    label=f'Channel {channel_number} play pos',
                    set_cmd=self._set_sequence_play_pos,
                    vals=Arrays(shape=(2,)),
                    docstring='set seq play pos for this channel')

        self.add_parameter('wave',
                    label=f'Channel {channel_number} wave',
                    unit='V',
                    set_cmd=self._set_awg_wave,
                    vals=Arrays(),
                    docstring='set wave for this channel')
        
        self.add_parameter('predefined_wave_param', # 实时产生波形数据，配置实时波形参数
                    label=f'Channel {channel_number} wave param',
                    unit='V',
                    set_cmd=self._set_pre_wave_param,
                    vals=Arrays(shape=(6,)),
                    docstring='set dds wave param for this channel')
        
        self.add_parameter('awg_wave_param', # 存储波形数据，配置存储波形参数
                    label=f'Channel {channel_number} awg wave param',
                    unit='V',
                    set_cmd=self._set_awg_wave_param,
                    vals=Arrays(shape=(3,)),
                    docstring='set wave param for this channel')

        self.add_parameter('mode',
                    label=f'Channel {channel_number} mode',
                    unit=None,
                    set_cmd=self._set_mode,
                    get_cmd=self._get_mode,
                    vals=Strings(),
                    val_mapping = {'NORMAL':0, 'SEQUENCE':1, 'AWG':2},
                    docstring='set channel mode for this channel')
        
        self.add_parameter('enable_trig',
                    label=f'Channel {channel_number} enable trig',
                    unit=None,
                    set_cmd=self._eanble_trig,
                    vals=Bool(),
                    docstring='set channel trig enable for this channel')
        
        self.add_parameter('verify',
                    label=f'Channel {channel_number} verify',
                    get_cmd=self._get_verify,
                    docstring='set/get channel verify for this channel')
        

    def _set_output_state(self, state):
        cmd = open_close_cmd()
        cmd.ch = self.channel_number
        if state == 'ON':
            cmd.switch = 1
            self._parent.zwdx_send(cmd.create_pack())
            self._parent._get_status()
            self._parent._ctrl_led(0x01)
        else:
            cmd.switch = 2
            mode = self._get_mode()
            self._set_mode(0)
            temp = self._parent._get_allch_vol()
            vol_list = list(map(abs, temp))
            self._set_voltage(0)
            delay = max(vol_list) / (self.k / 1000)
            time.sleep(delay)
            self._parent.zwdx_send(cmd.create_pack())
            self._parent._get_status()
            self._parent._ctrl_led(0x02)
            self._set_mode(mode)
    


    def _get_output_state(self):
        cmd = status_cmd()
        cmd.ch = self.channel_number
        self._parent.zwdx_send(cmd.create_pack())
        recvmsg = self._parent.zwdx_recv(7)
        temptup = struct.unpack('BBBBBBB', recvmsg)
        return temptup[4]

    def _set_voltage(self, vol:float, sequence_index=0):
        cmd = vol_cmd()
        cmd.ch = self.channel_number
        cmd.sequence = sequence_index
        b = format(vol, '.6f').encode('utf-8')
        length = len(b)
        cmd.vol.clear()
        for cnt in range(8):
            if cnt < length:
                cmd.vol.append(b[cnt])
            else:
                cmd.vol.append(0)
        self._parent.zwdx_send(cmd.create_pack())
        self._parent._get_status()

    def _get_voltage(self):
        i32 = lambda x: c_int32(x).value if x >= 0 else c_int32((1 << 32) + x).value
        cmd = get_vol_cmd()
        cmd.ch = self.channel_number
        self._parent.zwdx_send(cmd.create_pack())
        msg = self._parent.zwdx_recv(15)
        a, b, c, d, e, vol, k, f, g = struct.unpack('!BBBBBIIBB', msg)
        self.k = k
        rtn_vol = i32(vol) / 1000000
        return rtn_vol
    
    def _set_slope(self, k:int):
        cmd = slope_cmd()
        cmd.slope = k
        self._parent.zwdx_send(cmd.create_pack())
        self._parent._get_status()
        self.k = k

    def _set_pwm(self, enable:bool):
        cmd = set_ch_dis()
        cmd.ch = self.channel_number - 1
        cmd.ty = 3
        cmd.st = enable
        self._parent.zwdx_send(cmd.create_pack())
        self._parent._get_status()

    def _set_sequence(self, volseq:list):
        """
        设置各通道电压序列
        :param volseq:电压序列,最多支持16个 eg:[1,2,3,5,6]
        :return:
        """
        assert len(volseq) <= 16, "vol is too many[len<=16]"
        for i in range(len(volseq)):
            self._set_voltage(volseq[i], i+1)

    def _set_sequence_play_pos(self, pos=(0,0)):
        """
        设置单个通道序列播放的起始位置和结束位置
        :param ch: 通道号
        :param start: 起始位置
        :param end: 结束位置
        :return:
        """
        start, end = pos
        cmd = set_sequence_total_cmd()
        cmd.ch = self.channel_number
        seq_start_total = (start << 16) | (end - start + 1)  # 设置起始地址和序列个数
        cmd.sequence_total = seq_start_total
        self._seq_start_total = seq_start_total
        self._parent.zwdx_send(cmd.create_pack())
        self._parent._get_status()
        chtrigcmd = set_sequence_cmd()
        chtrigcmd.ch = self.channel_number
        chtrigcmd.type = cmd_type.SET_SEQUENCE_POS_CMD.value
        chtrigcmd.sequence_mode = 1  # enable ch trigger
        self._parent.zwdx_send(chtrigcmd.create_pack())
        chtrigcmd.type = cmd_type.SET_SEQUENCE_CMD.value
        self._parent._get_status()

    def _set_awg_wave(self, srcwave:np.ndarray): # [-1,1]
        tempwave = (srcwave - self.verify_b) / self.verify_k
        tempwave1 = tempwave + 1
        y = np.round(tempwave1 * (2**19 - 1))
        wave = np.asarray(y, dtype=np.int32)

        cmd = public_cmd()
        wavedata = wave.tobytes()
        interger = int(len(wavedata) / 500)
        remainder = int(len(wavedata) % 500)
        # print(f'{interger} {remainder}')
        cmd.data4 = self.channel_number - 1
        allpack = 0
        if remainder > 0:
            allpack = interger + 1
        else:
            allpack = interger
        cmd.data1 = allpack
        cnt = 0
        for i in range(interger):
            cmd.data2 = i
            cmd.data3 = 500 
            cmd.datan = np.frombuffer(wavedata[i*500:i*500+500], np.uint8)
            self._parent.zwdx_send(cmd.build())
            cnt += 1
            time.sleep(0.001)
        if remainder > 0 :
            cmd.data2 = allpack - 1
            cmd.data3 = remainder
            cmd.datan = np.frombuffer(wavedata[cnt*500:cnt*500+remainder], np.uint8)
            self._parent.zwdx_send(cmd.build())
        self._parent.zwdx_recv(1) # wait finish

    def _set_pre_wave_param(self, param):
        wave_type, plus, Freq, srcAmp, Phase, offset = param
        Amp = (srcAmp-self.verify_b)/self.verify_k

        vol_range = 10000 #输出电压范围单位mV
        sample_freq = 500000#采样频率单位Hz     
        real_ch = self.channel_number - 1
        offset_all = int(offset / vol_range * 0x7FFFF)
        #正弦参数计算
        # 
        # (优先确保所有波形频率一直)
        # sin_constraint = sample_freq / (int(sample_freq / Freq / 4 ) * 4)
        # sin_freq = int(sin_constraint / sample_freq  * pow(2,44))
        #
        sin_freq = int(Freq * pow(2,44) / sample_freq)
        sin_freq_h = sin_freq >>32
        sin_freq_l = sin_freq & 0xFFFFFFFF
        sin_amp = int(Amp / vol_range * 0x7FFFF)
        sin_phase = int(Phase / 360 *  pow(2,44))
        sin_phase_h = sin_phase >> 32
        sin_phase_l = sin_phase & 0xFFFFFFFF


        #方波及脉冲参数计算
        if (wave_type == 0x02):
            plus_vp = int((Amp + vol_range )/ (2 * vol_range) * pow(2,20) - 1)
            plus_vn = int(pow(2,20) - (Amp + vol_range )/ (2 * vol_range) * pow(2,20))
            plus_high_n = int(sample_freq / Freq / 2) 
            plus_low_n  = int(sample_freq / Freq / 2 )
        else:
            plus_vp = int((Amp + vol_range )/ (2 * vol_range) * pow(2,20) - 1)
            plus_vn = pow(2,19)
            plus_high_n = int((sample_freq / Freq)* plus)
            plus_low_n  = int(sample_freq / Freq)  - plus_high_n
        #三角波参数计算
        triangle_high = int(((Amp + vol_range )/ (2 * vol_range) * pow(2,20) - pow(2,19)) / (sample_freq / Freq / 4 + 1))
        triangle_low  = triangle_high
        triangle_points = int(sample_freq / Freq / 4) 
        triangle_phase = 0

        #锯齿波参数计算
        sawtooth_step = int((((Amp + vol_range )/ (2 * vol_range) * pow(2,20) - 1)- pow(2,19)) / (sample_freq / Freq / 2 ))
        sawtooth_points_high = int(sample_freq / Freq / 2 ) 
        sawtooth_points_low = sawtooth_points_high 

        if(wave_type == 0x01):
            wave_type_send = 0x01
            sin_real_freq  = float(sin_freq / pow(2,44) * sample_freq)
            # print('channel : ',self.channel_number,';','wave type : Sine     ; Real Out Freq : ',sin_real_freq,'Hz')
        elif((wave_type == 0x02)):
            wave_type_send = 0x02
            square_real_freq  = float(sample_freq / (plus_high_n * 2)) 
            # print('channel : ',self.channel_number,';','wave type : Square   ; Real Out Freq : ',square_real_freq,'Hz')
        elif((wave_type == 0x05)):
            wave_type_send = 0x02
            square_real_freq  = float(sample_freq / (int(plus_high_n / plus )))
            # print('channel : ',self.channel_number,';','wave type : Plus     ; Real Out Freq : ',square_real_freq,'Hz')
        elif(wave_type == 0x03):
            wave_type_send = 0x03
            triangle_real_freq  = float(sample_freq / (triangle_points * 4))
            # print('channel : ',self.channel_number,';','wave type : Triangle ; Real Out Freq : ',triangle_real_freq,'Hz')
        elif(wave_type == 0x04):
            wave_type_send = 0x04
            sawtooth_real_freq  = float(sample_freq / (sawtooth_points_high * 2))
            # print('channel : ',self.channel_number,';','wave type : Sawtooth ; Real Out Freq : ',sawtooth_real_freq,'Hz')            
        else:
            pass
            # print('wave type err!')

        #配置波形参数
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 11*4, sin_amp)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 12*4, sin_phase_h)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 13*4, sin_phase_l)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 14*4, sin_freq_h)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 15*4, sin_freq_l)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 16*4, plus_vp)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 17*4, plus_vn)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 18*4, plus_high_n)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 19*4, plus_low_n)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 20*4, sawtooth_step)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 21*4, sawtooth_points_high)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 22*4, sawtooth_points_low)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 23*4, triangle_high)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 24*4, triangle_low)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 25*4, triangle_phase)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 26*4, triangle_points)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 27*4, offset_all)

        #设置wave_type
        #read_back = self.get_reg_value(0x43C00400 + real_ch* 32*4)
        self._parent.dev_mem_write(0x43C00400 , real_ch* 32*4, ((wave_type_send << 13) & 0x1E000))

    def _set_awg_wave_param(self, param):
        play_mode, play_cnt, play_length = param
        real_ch = self.channel_number - 1
        wave_type_send = play_mode + 5
        self._parent.dev_mem_write(0x43C00400 , real_ch* 32*4, ((wave_type_send << 13) & 0x1E000)) 
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 10*4, play_length)
        self._parent.dev_mem_write(0x43C00400 , real_ch * 32*4 + 2*4, play_cnt<<16)
        
    def _set_mode(self, mode):
        """
        :param mode:0：正常模式， 1：序列模式 2:AWG模式
        """
        modecmd = set_sequence_cmd()
        modecmd.sequence_mode = mode
        modecmd.ch = self.channel_number
        self._parent.zwdx_send(modecmd.create_pack())
        self._parent._get_status()

    def _get_mode(self):
        mode = self._parent.dev_mem_read(0x43C00000, (self.channel_number-1)*12*4)
        mode_value = (mode >> 13) & 0x03
        return mode_value

    def _eanble_trig(self, enable):
        chtrigcmd = set_sequence_cmd()
        chtrigcmd.ch = self.channel_number
        chtrigcmd.type = cmd_type.SET_SEQUENCE_POS_CMD.value
        chtrigcmd.sequence_mode = enable  # disenable ch trigger
        self._parent.zwdx_send(chtrigcmd.create_pack())
        chtrigcmd.type = cmd_type.SET_SEQUENCE_CMD.value
        self._parent._get_status()

    def _get_verify(self):
        cmd = get_verify_cmd()
        cmd.ch = self.channel_number
        self._parent.zwdx_send(cmd.create_pack())
        kb = self._parent.zwdx_recv(16)
        kbarray = np.frombuffer(kb, np.double)
        self.verify_k = kbarray[0]
        self.verify_b = kbarray[1]


class DC2000MlutiChannelDevice(Instrument):
    def __init__(self, name, ip_addr:str, **kwargs):
        super().__init__(name, **kwargs)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)
        self.s.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 60 * 1000, 10 * 1000))
        self.s.settimeout(5)
        try: 
            self.s.connect((ip_addr, 8080))
        except socket.error as e:
            print(f'Connection error: {e}')

        self.add_submodule('channels', ChannelList(self, 'Channels', SingleChannel, snapshotable=False))

        for i in np.arange(1, 17, 1):
            channel = SingleChannel(self, f'ch{i}', i)
            self.channels.append(channel)
            setattr(self, f'ch{i}', channel)

        # self.connect_message() 没有设备信息，暂时不写

        self.add_parameter('all_ch_status',
                            set_cmd=None,
                            get_cmd=self._allch_status,
                            docstring='get all channel status')
        
        self.add_parameter('enable_global_trig',
                            set_cmd=self._enable_global_trig,
                            vals=Bool(),
                            docstring='set global trig enable or disable')
        
        self.add_parameter('soft_trig',
                            label=f'send soft trig',
                            set_cmd=self._set_soft_trig,
                            vals=Arrays(shape=(2,)),
                            # vals=Dict(trigcnt=Numbers(0, 100), interval=Numbers(0, 10000)),
                            docstring='send soft trig')
        

    def zwdx_send(self, data):
        self.s.send(data)

    def zwdx_recv(self, cnt):
        return self.s.recv(cnt)
    
    def dev_mem_write(self, base, offset, data):
        cmd = set_reg_cmd()
        cmd.type = cmd_type.SET_PLREG_CMD.value
        cmd.base_address = base
        cmd.offset = offset
        cmd.reg = data
        self.zwdx_send(cmd.create_pack())

    def dev_mem_read(self, base, offset):
        cmd = set_reg_cmd()
        cmd.type = cmd_type.GET_PLREG_CMD.value
        cmd.base_address = base
        cmd.offset = offset
        self.zwdx_send(cmd.create_pack())
        msgbytes = self.zwdx_recv(6)
        return int.from_bytes(msgbytes[2:6], 'little')

    def _get_status(self):
        msg = self.zwdx_recv(7)
        format_str = '!BBBBBBB'
        a, b, c, d, status, e, f = struct.unpack(format_str, msg)
        return status
    
    def _get_allch_vol(self):
        vol_list = []
        for i in np.arange(0,16,1):
            vol_list.append(self.channels[i]._get_voltage())
        return vol_list
    
    def _enable_global_trig(self, enable):
        """
        设置所有通道（全局）使能/禁止触发
        :param enable: 0:禁止 1:使能
        :return:
        """
        gtrigcmd = set_sequence_cmd()
        gtrigcmd.type = cmd_type.SET_GOLBALEN_CMD.value
        gtrigcmd.sequence_mode = enable
        self.zwdx_send(gtrigcmd.create_pack())
        gtrigcmd.type = cmd_type.SET_SEQUENCE_CMD.value
        return self._get_status()

    def _set_soft_trig(self, trig):
        """
        给trigcnt个软触发，可以理解为给trigcnt个脉冲，脉冲周期为interval
        :param trigcnt:触发次数，也可以理解为脉冲个数
        :param interval: 触发间隔，可以理解为脉冲周期
        :return:
        """
        trigcnt, interval = trig
        # trigcnt = trig.get('trigcnt')
        # interval = trig.get('interval')

        gsofttrigcmd = set_sequence_cmd()
        gsofttrigcmd.type = cmd_type.SET_SOFTTRIG_CMD.value
        while trigcnt:
            gsofttrigcmd.sequence_mode = 1
            self.zwdx_send(gsofttrigcmd.create_pack())
            self._get_status()
            time.sleep(interval / 2)
            gsofttrigcmd.sequence_mode = 0
            self.zwdx_send(gsofttrigcmd.create_pack())
            self._get_status()
            time.sleep(interval / 2)
            trigcnt -= 1
        gsofttrigcmd.type = cmd_type.SET_SEQUENCE_CMD.value

    def _ctrl_led(self, value):
        cmd = led_cmd()
        cmd.status = value
        self.zwdx_send(cmd.create_pack())

    def _allch_status(self):
        stlist = []
        for i in range(16):
            st = self.channels[i]._get_output_state()
            if st == 1:
                stlist.append('ON')
            else:
                stlist.append('OFF')
        return stlist
    
    def change_ip(self, ip='', mask='255.255.255.0', gw='192.168.1.1'):
        ip_list = []
        mask_list = []
        gw_list = []
        ip_list = ip.split('.')
        int_ip_list = list(map(int, ip_list))
        mask_list = mask.split('.')
        int_mask_list = list(map(int, mask_list))
        gw_list = gw.split('.')
        int_gw_list = list(map(int, gw_list))
        cmd = ip_cmd(int_ip_list, int_mask_list, int_gw_list)
        self.zwdx_send(cmd.create_pack())

    def close(self):
        """关闭 socket 连接"""
        self.s.close()
        super().close()
    