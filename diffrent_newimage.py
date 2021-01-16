newimage[x][y] = (int(image[x-1][y-1])*int(image[x-1][y])*int(image[x-1][y+1])*int(image[x][y-1]) *
                  int(image[x][y])*int(image[x][y+1])*int(image[x+1][y-1])*int(image[x+1][y]) *
                  int(image[x+1][y+1]))
