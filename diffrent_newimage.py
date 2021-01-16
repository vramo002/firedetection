newimage[x][y] = (int(image[x-1][y-1])*int(image[x-1][y])*int(image[x-1][y+1])*int(image[x][y-1]) *
                  int(image[x][y])*int(image[x][y+1])*int(image[x+1][y-1])*int(image[x+1][y]) *
                  int(image[x+1][y+1]))

newimage[x][y] = (image[x-1][y-1]*image[x-1][y]*image[x-1][y+1]*image[x][y-1]*image[x][y]*image[x][y+1] *
                  image[x+1][y-1]*image[x+1][y]*image[x+1][y+1])
