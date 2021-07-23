# coding: utf-8

# from pynq.pl import PL
# import pynq
import serial
# import RPi.GPIO as GPIO
import cv2
import numpy as np
import time
import dronekit as dk
from pymavlink import mavutil
from datetime import datetime
import os


def arm_and_takeoff(aTargetAltitude):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    vehicle.mode = dk.VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        print(" Waiting for arming...")
        time.sleep(1)

    time.sleep(8)
    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)  # Take off to target altitude
    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)


def send_ned_velocity(velocity_x, velocity_y, velocity_z):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    vehicle.send_mavlink(msg)
    vehicle.flush()


def goto_position_target_local_ned(north, east, down):  # THIS FUNCTION IS NOT NEEDED IN THIS SCRIPT
    # IT IS HERE FOR REFERENCE
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame
        0b0000111111111000,  # type_mask (only positions enabled)
        north, east, down,  # x, y, z positions (or North, East, Down in the MAV_FRAME_BODY_NED frame
        0, 0, 0,  # x, y, z velocity in m/s  (not used)
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    # send command to vehicle
    vehicle.send_mavlink(msg)
    vehicle.flush()


# ----------------------------------------------KEEP IT THERE FOR REFERENCE
# connection_string = '/dev/ttyACM0'	#Establishing Connection With Flight Controller
# vehicle = dk.connect(connection_string, wait_ready=True, baud=115200)
# cmds = vehicle.commands
# cmds.download()
# cmds.wait_ready()
# waypoint1 = dk.LocationGlobalRelative(cmds[0].x, cmds[0].y, 3)  # Destination point 1
# ----------------------------------------------

# END of definitions!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# In[3]:

# In[4]:
# camera (input) configuration
# frame_in_w = 640
# frame_in_h = 480

# do need to set becasue for some reason the analog to digital convertor
# randomily set the size base on the computer and changing it will casue to not
# capture

# In[5]:
now = datetime.now()
time_stamp = now.strftime("%Y-%m-%d_%H_%M_%S")
with open(time_stamp + '_logs.txt', 'a+') as f:
    f.write('FLIGHT TEST: ' + time_stamp)
    f.write('\n')

cap = cv2.VideoCapture(0)
# videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
# videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);
print("capture device is open: " + str(cap.isOpened()))
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'XVID'))
writer = cv2.VideoWriter(time_stamp + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 20, (int(cap.get(3)), int(cap.get(4))))

fps = cap.get(cv2.CAP_PROP_FPS)
w = cap.get(3)
h = cap.get(4)
roistartw = cap.get(3) / 10
roistarth = cap.get(4) / 10

lower_red = np.array([0, 100, 100])
upper_red = np.array([179, 255, 255])
with open(time_stamp + '_logs.txt', 'a+') as f:
    f.write('Lower Red: ' + str(lower_red) + ' Upper Red: ' + str(upper_red))
    f.write('\n')

# In[6]:
# initialize the HOG descriptor/person detector
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# globalimage=np.zeros((640,480,3), np.uint8)
# not need to for fire

# Setting up GPIO
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(18, GPIO.OUT)
# p = GPIO.PWM(18, 50)
# p.start(2.5)
# time.sleep(1)
# ...

# setting up xbee communication
# GPIO.setwarnings(False)
ser = serial.Serial(

    port='/dev/ttyUSB0',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)
# INITIALIZING DRONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
connection_string = '/dev/ttyACM0'  # Establishing Connection With PIXHAWK
vehicle = dk.connect(connection_string, wait_ready=True, baud=115200)  # PIXHAWK is PLUGGED to NUC (RPi too?) VIA USB
cmds = vehicle.commands
cmds.download()
cmds.wait_ready()

altitude = 30
speed = 1.5
waypoint1 = dk.LocationGlobalRelative(cmds[0].x, cmds[0].y, altitude)
arm_and_takeoff(altitude)
vehicle.airspeed = speed  # set drone speed to be used with simple_goto
vehicle.simple_goto(waypoint1)  # trying to reach 1st waypoint
with open(time_stamp + '_logs.txt', 'a+') as f:
    f.write('Altitude: ' + str(altitude))
    f.write('\n')
    f.write('Airspeed: ' + str(speed))
    f.write('\n')
# time.sleep(30)
# ----------------------------------------------
detected = 0
counter = 10
while not detected:
    # read next image
    ret, frame = cap.read()
    writer.write(frame)

    if not ret:
        print('Error Camera not connected')
        with open(time_stamp + '_logs.txt', 'a+') as f:
            f.write('Camera Error : RET returned False')
            f.write('\n')

    # cv2.imshow('original', frame)
    # frame = imutils.resize(frame, width=320)
    # frame3 = cv2.rectangle(frame, (int(roistartw)*3,int(roistarth)*3), (int(w)-int(roistartw)*3,int(h)-int(roistarth)*3), (0,255,0), 2)
    # cv2.imshow('roibox', frame3)
    frame = frame[int(roistarth) * 1:int(h) - int(roistarth) * 1, int(roistartw) * 1:int(w) - int(roistartw) * 1]
    frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame1, lower_red, upper_red)
    res = cv2.bitwise_and(frame1, frame1, mask=mask)
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    if cv2.countNonZero(gray) != 0:
        counter = counter - 1
        if counter == 0:
            now = datetime.now()
            time_stamp1 = now.strftime("%Y-%m-%d_%H_%M_%S")
            cv2.imwrite(time_stamp + '.jpg', frame)
            detected = 1
            print("FIRE")
            with open(time_stamp + '_logs.txt', 'a+') as f:
                f.write('FIRE AT: ' + time_stamp1)
                f.write('\n')
                f.write('Number of Red pixels: ' + str(cv2.countNonZero(gray)))
                f.write('\n')
    if cv2.countNonZero(gray) == 0:
        counter = 10
    # cv2.imshow('ROI', frame)
    # cv2.imshow('mask', mask)
    # cv2.imshow('res', res)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break
    cv2.waitKey(int((1 / int(fps)) * 1000))
# STOP Flying --------------------------------
send_ned_velocity(0, 0, 0)  # stop the vehicle
# sleepNrecord(2)
time.sleep(3)  # for 3 seconds
# READ CURRENT COORDINATES FROM PIXHAWK-------------------
lat = vehicle.location.global_relative_frame.lat  # get the current latitude
lon = vehicle.location.global_relative_frame.lon  # get the current longitude
coords = str(lat) + " " + str(lon)
with open(time_stamp + '_logs.txt', 'a+') as f:
    f.write('Coordinates: ' + coords)
    f.write('\n')
# TRANSMIT CURRENT COORDINATES TO RESCUE DR --------------
ser.write(coords.encode())

# RETURN HOME CODE ----------------------------
vehicle.mode = dk.VehicleMode("RTL")
# ANOTHER LOOP
while vehicle.armed:
    ret, frame = cap.read()
    writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break
    cv2.waitKey(int((1 / int(fps)) * 1000))
# time.sleep(20)

# ---------------------------------------------
vehicle.mode = dk.VehicleMode("LAND")
vehicle.flush()


# add to keep recourding??
# or move code to into look and if statement
# and add another flag
