import numpy as np
import navpy
import math as m
from PIL import Image

def getCameraIntrinsicMatrix():

    # Get focal lengths fx and fy of camera and principle point cx and cy
    # (all values are in pixels)
    # f = 19mm, image size = 320 x 213
    f = 19
    fx = f * 3.7795275591
    fy = f * 3.7795275591
    cx = 160
    cy = 106.5
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

def getCameraPosition():
    # Replace these with actual GPS camera positions
    lat = +(34. + 0./60 + 0.19237/3600) # North
    lon = -(117. + 20./60 + 0.77188/3600)  # West
    alt = 234.052  # [meters]
    return lat, lon, alt

def getCameraHeight():

    #zc = 172 # in inches
    zc = 4.3688 # in meters
    #zc = 53.25 # in inches
    return zc

def getCameraOrientation():
    roll = 0
    pitch = 40
    yaw = 0
    return roll, pitch, yaw

def getGimbalAngles():
    tilt = 40
    pan = 50
    return tilt, pan


def main():

    # TEST
    # u = 1
    # v = 2
    # uv = np.array([u, v, 1])
    # y = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(np.dot(y, uv))
    # print(y)

    A = getCameraIntrinsicMatrix()
    # print(A)

    #test = np.array([[4, 7], [2, 6]])
    #print(np.linalg.inv(test))

    # Get camera's GPS positions and height
    lat, lon, alt = getCameraPosition()
    zc = getCameraHeight()

    # Transform camera position from Geodetic (lla) to NED
    # A point near Los Angeles, CA, given in https://github.com/NavPy/NavPy/blob/master/navpy/core/tests/test_navpy.py
    # (this will change to the location of the experiment later on)
    lat_ref = +(34. + 0./60 + 0.00174/3600) # North
    lon_ref = -(117. + 20./60 + 0.84965/3600) # West
    alt_ref = 251.702 # in meters
    rnnc = navpy.lla2ned(lat, lon, alt, lat_ref, lon_ref, alt_ref, latlon_unit='deg', alt_unit='m', model='wgs84')
    #print(rnnc)
    # TEST
    #lat_ref = 34.004924
    #lon_ref = -117.897842
    #alt_ref = 129.9972
    rnnc = np.array([0, 0, -zc])
    #rnnc = np.array([0, 0, -1.3462])


    # Get the body frame’s orientation angles and gimbal angles from the GPS
    roll, pitch, yaw = getCameraOrientation()
    tilt, pan = getGimbalAngles()
    # print(roll, pitch, yaw, tilt, pan)

    # Compute Rcn from roll, pitch, yaw, tilt, pan angles
    Rnb = np.array([[m.cos(yaw)*m.cos(pitch), -m.sin(yaw)*m.cos(roll) + m.cos(yaw)*m.sin(roll)*m.sin(pitch), m.sin(yaw)*m.sin(roll) + m.cos(yaw)*m.cos(roll)*m.sin(pitch)],
                    [m.sin(yaw)*m.cos(pitch), m.cos(yaw)*m.cos(roll) + m.sin(roll)*m.sin(pitch)*m.sin(yaw), -m.cos(yaw)*m.sin(roll) + m.sin(pitch)*m.sin(yaw)*m.cos(roll)],
                    [-m.sin(pitch), m.cos(pitch)*m.sin(roll), m.cos(pitch)*m.cos(roll)]])
    Rcm = np.array([[0, 1, 0],
                    [-1, 0, 0],
                    [0, 0, 1]])
    Rmb = np.array([[m.cos(pan)*m.cos(tilt), m.sin(pan)*m.cos(tilt), -m.sin(tilt)],
                    [-m.sin(pan), m.cos(pan), 0],
                    [m.cos(pan)*m.sin(tilt), m.sin(pan)*m.sin(tilt), m.cos(tilt)]])
    Rmb = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    # Rmb is simply the identity matrix if camera is strapped directly to airframe
    Rcn = np.linalg.inv(np.dot(Rnb, np.linalg.inv(np.dot(Rcm, Rmb))))
    #print(Rcn)

    # Compute G(NE) from Rcn and rnnc
    Gne = Rcn.copy()
    Gne[:, 2] = -np.dot(Rcn, rnnc)
    #print(Gne)

    # Obtain the pixel positions (u,v) of hot spots from fire detection process
    image = Image.open("location1_fire.jpg")
    pixels = image.load()
    width = image.size[0]
    height = image.size[1]
    # print(width, height)
    for i in range(0, height):
        for j in range(0, width):
            if (pixels[j, i] != 0):
                uv = np.array([j, i, 1])

                # Testing the pinhole camera model
                #xyzc = zc * np.dot(np.linalg.inv(A), uv)
                #print(m.sqrt(xyzc[0] * xyzc[0] + xyzc[1] * xyzc[1] + xyzc[2] * xyzc[2]))

                # Compute the NE coordinates of the hot spots from G(NE)
                NE = zc * np.dot(np.dot(np.linalg.inv(Gne), np.linalg.inv(A)), uv)
                #print(m.sqrt(NE[0] * NE[0] + NE[1] * NE[1]))

                # Converts hot spots' coordinates from NED to lla (GPS) coordinates
                result = navpy.ned2lla([NE[0], NE[1], 0], lat_ref, lon_ref, alt_ref, latlon_unit='deg', alt_unit='m', model='wgs84')
                print(result)

                #print('X', end='')
            #else:
                #print('_', end='')
        #print()

if __name__ == "__main__":
    main()