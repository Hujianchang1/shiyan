import socket
import numpy as np
import struct
from qcodes import Instrument
from qcodes.instrument.channel import InstrumentChannel, ChannelList
from qcodes.utils.validators import Strings

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

class gnd_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = 0x06
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

class open_close_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = 0x02
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
    
class vol_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = 0x01
    ch = None
    vol = []
    crc = 0
    end = 0xaa

    def __init__(self):
        super().__init__()
        pass

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type, self.ch]
        buffer += self.vol
        buffer.append(self.crc)
        buffer.append(self.end)
        super().set_cmd(buffer)
        return super().build()
    
class slope_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 8
    type = 0x03
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
    
class ip_cmd(cmd_base):
    hd = 0x55
    id = 0x01
    length = 11
    type = 0x08
    ip = []
    mask = []
    crc = 0
    end = 0xaa

    def __init__(self, ip=[], mask=[]):
        super().__init__()
        self.ip = ip
        self.mask = mask

    def create_pack(self):
        buffer = [self.hd, self.id, self.length, self.type]
        buffer += self.ip
        buffer += self.mask
        buffer.append(self.crc)
        buffer.append(self.end)

        super().set_cmd(buffer)

        return super().build()


class SingleChannel(InstrumentChannel):
    def __init__(self, parent: Instrument, name: str, channel_number, **kwargs):
        super().__init__(parent, name, **kwargs)
        self.channel_number = channel_number
        self.channel_state  = 'OFF'

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
                            set_parser=float,
                            get_parser=float,
                            docstring='get/set voltage for this channel')
        
        self.add_parameter('slope',
                            label=f'Channel {channel_number} slope',
                            unit='mv/s',
                            set_cmd=self._set_slope,
                            set_parser=int,
                            docstring='set slope for this channel')
        

    def _set_output_state(self, state):
        cmd = open_close_cmd()
        cmd.ch = self.channel_number + 8
        if state == 'ON':
            cmd.switch = 0x02
        else:
            cmd.switch = 0x01
        self.channel_state = state
        self._parent.zwdx_send(cmd.create_pack())

    def _get_output_state(self):
        return self.channel_state

    def _set_voltage(self, vol:float):
        assert -10 <= vol <= 10, f'input vol over range[-10,10]'
        cmd = vol_cmd()
        cmd.ch = self.channel_number + 8
        b = format(vol, '.6f').encode('utf-8')
        length = len(b)
        cmd.vol.clear()
        for i in range(8):
            if i < length:
                cmd.vol.append(b[i])
            else:
                cmd.vol.append(0)
        self._parent.zwdx_send(cmd.create_pack())

    def _get_voltage(self):
        return None
    
    def _set_slope(self, k:int):
        """设置斜率，默认1000MV/S"""
        cmd = slope_cmd()
        cmd.slope = k
        self._parent.zwdx_send(cmd.create_pack())
        


class DC1000MlutiChannelDevice(Instrument):
    def __init__(self, name, ip_addr:str, testing=False, **kwargs):
        super().__init__(name, **kwargs)
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, True)
        self.s.ioctl(socket.SIO_KEEPALIVE_VALS, (1, 60 * 1000, 10 * 1000))
        self.s.settimeout(5)
        try: 
            self.s.connect((ip_addr, 7))
        except socket.error as e:
            print(f'Connection error: {e}')

        self.add_submodule('channels', ChannelList(self, 'Channels', SingleChannel, snapshotable=False))

        for i in range(1, 9):
            channel = SingleChannel(self, f'ch{i}', i)
            self.channels.append(channel)
            setattr(self, f'ch{i}', channel)


        # self.connect_message() 没有设备信息，暂时不写

        # 初始化共地模式
        cmd = gnd_cmd()
        cmd.ch = 9
        self.zwdx_send(cmd.create_pack())
    
    def zwdx_send(self, data):
        self.s.send(data)

    def zwdx_recv(self, cnt):
        return self.s.recv(cnt)

    def allch_status(self):
        return [i.channel_state for i in self.channels]
    
    def change_ip(self, ip='', mask='255.255.255.0'):
        ip_list = []
        mask_list = []
        ip_list = ip.split('.')
        int_ip_list = list(map(int, ip_list))
        mask_list = mask.split('.')
        int_mask_list = list(map(int, mask_list))
        cmd = ip_cmd(int_ip_list, int_mask_list)
        self.zwdx_send(cmd.create_pack())
    
    def close(self):
        """关闭 socket 连接"""
        self.s.close()
        super().close()