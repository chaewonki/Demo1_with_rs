from .protocol_constants import STX, KV, MA, MAS, MS, APR, COL_X, COL_Y
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal
import serial
import threading

class SerialComm(QWidget):
    kv_signal = pyqtSignal(int)
    mA_signal = pyqtSignal(float)
    mAs_signal = pyqtSignal(float)
    ms_signal = pyqtSignal(float)
    projection_signal = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.initSerial()

    def initSerial(self):
        self.serial_port = serial.Serial(
            port='COM3',
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            timeout=None
        )
        
        serial_thread = threading.Thread(target=self.receive_packet)
        serial_thread.daemon = True
        serial_thread.start()
    
    def extract_direction_bits(self, data_bytes):
        last_byte = data_bytes[-1]
        thrid_fourth_bits = (last_byte >> 2) & 0b11
        thrid_fourth_bits = (thrid_fourth_bits<< 2) & 0b1111
        return thrid_fourth_bits

    def receive_packet(self):
        while True:
            byte_read = self.serial_port.read(1)
            if byte_read.startswith(STX):
                packet = byte_read + self.serial_port.read(9)
                id_bytes = packet[1:4] 
                data_bytes = packet[4:7]
                if id_bytes == KV:
                    kV_value = int.from_bytes(data_bytes, byteorder='big')
                    self.kv_signal.emit(kV_value)
                elif id_bytes == MA:
                    mA_value = int.from_bytes(data_bytes, byteorder='big') / 100
                    self.mA_signal.emit(mA_value)
                elif id_bytes == MAS:
                    mAs_ms_value = int.from_bytes(data_bytes, byteorder='big') / 100
                    self.mAs_signal.emit(mAs_ms_value)
                elif id_bytes == MS:
                    ms_value = int.from_bytes(data_bytes, byteorder='big') / 100
                    self.ms_signal.emit(ms_value)
                elif id_bytes == APR:
                    projection_bits = self.extract_direction_bits(data_bytes)
                    self.projection_signal.emit(projection_bits)
                elif id_bytes == COL_X:
                    cox_value = int.from_bytes(data_bytes)
                    print(f"COL_X: {cox_value}")
                elif id_bytes == COL_Y:
                    coy_value = int.from_bytes(data_bytes)
                    print(f"COL_Y: {coy_value}")
    
    def string_to_byte(self, key_string):
        return bytes.fromhex(key_string.replace(" ",""))
