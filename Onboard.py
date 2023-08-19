import os
import time
import struct

import digi.xbee.exception
from digi.xbee.devices import DigiMeshDevice,RemoteDigiMeshDevice,XBee64BitAddress,ZigBeeDevice

xbee = DigiMeshDevice('COM5', 115200)
remote = RemoteDigiMeshDevice(xbee, XBee64BitAddress.from_hex_string("0013A20040D8DCD5"))
xbee.open(force_settings=True)
print("open")

def pack(msg_id, uav_id, mode, battery, timestamp, lat, lng, alt, heading, speed):
    return struct.pack('<BBBBqiiiii', msg_id, uav_id, mode, battery, timestamp, lat, lng, alt, heading, speed)


b = [98, 95, 95, 90, 88, 85, 82]
hea = [-90.165, -135.5614, -145.15614, 180.156, 160.978, 80.3754, 45.15564, 15.55555]
m = [8, 8, 8, 8, 1, 1, 4]
t = time.time
# uavID = int(xbee.get_node_id())
# print(uavID)
def check(interval, pt):
    d = int(interval*10)
    if int((t()) * 10) % d == 0 and t() - pt > interval/1.01:
        return True
    else:
        return False
p = 0
a = 0
i = 0
t = time.time
start = time.time()
packs = bytearray(bytearray([100, 100]) + os.urandom(98))
print(len(packs))
# xbee.send_data(remote, packs)
# xbee.send_data_broadcast(packs)
print("send")
# data = []
# while not data:
#     try:
#         data = xbee1.read_data(timeout=1e-6)
#     except digi.xbee.exception.TimeoutException:
#         pass
# print(len(data.data))
while t() - start <= 1:
    xbee.send_data_async(remote, packs)
    # xbee.send_data_broadcast(bytearray(256))
    i += 1
print(i)
# while True:
#     for i in range(7):
#         msg = pack(0, uavID, m[i], b[i], int(time.time()*1e3), int(uav1[i][0]*1e7), int(uav1[i][1]*1e7), int(26*1e7),
#                    int(hea[i]*1e3), int(5*1e3))
#         print(len(msg))
#         xbee.send_data(remote, msg)
#         xbee.send_data_async(remote, msg)
#         # xbee.send_data_broadcast(msg)
#         time.sleep(0.3)
    # if check(0.1, p):
    #     a += 1
    #     p = time.time()
    #     xbee.send_data(remote, struct.pack('<q', int(p*1e5)))
    #     print("send", p)
    #
    # if xbee.has_packets():
    #     packet = xbee.read_data()
    #     if packet:
    #         print(packet.data.decode(), packet.timestamp)

# 0013A20042092F3A
# 0013A200420935CA



