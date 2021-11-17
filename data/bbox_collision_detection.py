# Adapted from http://www.3dkingdoms.com/weekly/weekly.php?a=3

# Input:  
# The Axis-Aligned bounding box is defined by B1 and B2
# B1:[x,y,z], the smallest values of X, Y, Z
# B2:[x,y,z], the largest values of X, Y, Z
# L1: [x,y,z], point 1 on the line 
# L2: [x,y,z], point 2 on the line 
# 
# Output:
# return True if line (L1, L2) intersects with the box (B1, B2)

import numpy as np

class Point:
    def __init__(self, x,y,z):
        self.x=np.array(x)
        self.y=np.array(y)
        self.z=np.array(z)

# returns True if line (L1, L2) intersects with the box (B1, B2)
# returns intersection point in Hit
class CheckLineBox:

    def __init__(self, B1, B2,  L1,  L2):
        self.B1 = Point(B1[0],B1[1],B1[2])
        self.B2 = Point(B2[0],B2[1],B2[2])
        self.L1 = Point(L1[0],L1[1],L1[2])
        self.L2 = Point(L2[0],L2[1],L2[2])
        self.Hit = None

    def GetIntersection(self, fDst1, fDst2, P1, P2):
        if ( (fDst1 * fDst2) >= 0):
            return 0
        if ( fDst1 == fDst2):
            return 0
        
        p1=np.array([P1.x,P1.y,P1.z])
        p2=np.array([P2.x,P2.y,P2.z])
        # self.Hit = P1 + (P2-P1) * ( -fDst1/(fDst2-fDst1) )
        hit = p1 + (p2-p1) * ( -fDst1/(fDst2-fDst1) )
        self.Hit = Point(hit[0],hit[1],hit[2])
        return 1

    def InBox(self, Axis):
        if ( Axis==1 and self.Hit.z > self.B1.z and self.Hit.z < self.B2.z and self.Hit.y > self.B1.y and self.Hit.y < self.B2.y):
            return 1
        if ( Axis==2 and self.Hit.z > self.B1.z and self.Hit.z < self.B2.z and self.Hit.x > self.B1.x and self.Hit.x < self.B2.x):
            return 1
        if ( Axis==3 and self.Hit.x > self.B1.x and self.Hit.x < self.B2.x and self.Hit.y > self.B1.y and self.Hit.y < self.B2.y):
            return 1
        return 0

    def check(self):
        if (self.L2.x < self.B1.x and self.L1.x < self.B1.x): 
            return False
        if (self.L2.x > self.B2.x and self.L1.x > self.B2.x): 
            return False
        if (self.L2.y < self.B1.y and self.L1.y < self.B1.y): 
            return False
        if (self.L2.y > self.B2.y and self.L1.y > self.B2.y): 
            return False
        if (self.L2.z < self.B1.z and self.L1.z < self.B1.z): 
            return False
        if (self.L2.z > self.B2.z and self.L1.z > self.B2.z): 
            return False
        if (self.L1.x > self.B1.x and self.L1.x < self.B2.x and
            self.L1.y > self.B1.y and self.L1.y < self.B2.y and
            self.L1.z > self.B1.z and self.L1.z < self.B2.z):
            self.Hit = self.L1 
            return True
            
        if ( (self.GetIntersection( self.L1.x-self.B1.x, self.L2.x-self.B1.x, self.L1, self.L2) and self.InBox(1))
        or (self.GetIntersection( self.L1.y-self.B1.y, self.L2.y-self.B1.y, self.L1, self.L2) and self.InBox(2)) 
        or (self.GetIntersection( self.L1.z-self.B1.z, self.L2.z-self.B1.z, self.L1, self.L2) and self.InBox(3)) 
        or (self.GetIntersection( self.L1.x-self.B2.x, self.L2.x-self.B2.x, self.L1, self.L2) and self.InBox(1)) 
        or (self.GetIntersection( self.L1.y-self.B2.y, self.L2.y-self.B2.y, self.L1, self.L2) and self.InBox(2)) 
        or (self.GetIntersection( self.L1.z-self.B2.z, self.L2.z-self.B2.z, self.L1, self.L2) and self.InBox(3))):
            return True

        return False


if __name__ == '__main__':
    
    # Debug
    B1=[0,0,0]
    B2=[2,3,4]
    L1=[0.5,-1,2]
    L2=[3,2,1]
    checker=CheckLineBox( B1, B2,  L1,  L2)
    collision_flag=checker.check()
    print('s')