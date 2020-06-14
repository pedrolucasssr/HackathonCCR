#Leitura de giroscopio e acelerometro via interface Raspberry Pi com MPU6050

import smbus # Importa modulo SMBus I2C
from time import sleep

# Registradores para o MPU6050 e seus enderecos
PWR_MGMT_1   = 0x6B
SMPLRT_DIV   = 0x19
CONFIG       = 0x1A
GYRO_CONFIG  = 0x1B
INT_ENABLE   = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H  = 0x43
GYRO_YOUT_H  = 0x45
GYRO_ZOUT_H  = 0x47


def MPU_Init():
    # Escreve em registrador de taxa de amostra
    bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
    
    # Escreve em registrador de gestao de energia
    bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
    
    # Escreve em registrador de configuracao
    bus.write_byte_data(Device_Address, CONFIG, 0)
    
    # Escreve em registrador de configuracao de giroscopio
    bus.write_byte_data(Device_Address, GYRO_CONFIG, 24)
    
    # Escreve em registrador de habilitacao de registro
    bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def read_raw_data(addr):
    # Valore de acelerometro e giroscopio sao em 16-bit
        high = bus.read_byte_data(Device_Address, addr)
        low = bus.read_byte_data(Device_Address, addr+1)
    
        # Concatena valores maior e menor
        value = ((high << 8) | low)
        
        # Adquire valores assinados de mpu6050
        if(value > 32768):
                value = value - 65536
        return value


bus = smbus.SMBus(1)    # define linguagem de comunicação
Device_Address = 0x68   # endereco placa MPU6050

MPU_Init()

print (" Lendo data de giroscopio e acelerometro")

while True:
    
    # Leitura de valor bruto de acelerometro
    acc_x = read_raw_data(ACCEL_XOUT_H)
    acc_y = read_raw_data(ACCEL_YOUT_H)
    acc_z = read_raw_data(ACCEL_ZOUT_H)
    
    # Leitura de valor bruto de Giroscopio
    gyro_x = read_raw_data(GYRO_XOUT_H)
    gyro_y = read_raw_data(GYRO_YOUT_H)
    gyro_z = read_raw_data(GYRO_ZOUT_H)
    
    # Escala completa +/- 250 graus pelo fator de escala de sensibilidade
    Ax = acc_x/16384.0
    Ay = acc_y/16384.0
    Az = acc_z/16384.0
    
    Gx = gyro_x/131.0
    Gy = gyro_y/131.0
    Gz = gyro_z/131.0
    

    print ("Gx=%.2f" %Gx, u'\u00b0'+ "/s", "\tGy=%.2f" %Gy, u'\u00b0'+ "/s", "\tGz=%.2f" %Gz, u'\u00b0'+ "/s", "\tAx=%.2f g" %Ax, "\tAy=%.2f g" %Ay, "\tAz=%.2f g" %Az)     
    sleep(1)
