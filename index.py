import numpy as np
import matplotlib.pyplot as plt
import math

class transformation:
  def __init__(self, i, j,k,scale_factor,x,y,z,F_old,theta_x,theta_y,theta_z):
    self.i = i
    self.j = j
    self.k = k
    self.scale_factor = scale_factor

    self.x = x
    self.y = y
    self.z = z
    self.F_old = F_old

  def vector_to_matrix(i,j,k,scale_factor):
    P = np.array([i,j,k,0])
    P_scaled = np.array([i,j,k,1])*scale_factor
    lambda_factor = math.sqrt(np.sum(np.square(P)))
    unit_vector = P/lambda_factor
    return print('P Original Vector = {0}\nP Scaled Vector = {1}\nLamda Factor = {2}\nUnity Vector = {3}'.format(P,P_scaled,lambda_factor,unit_vector))

  def translation(F_old,x,y,z):
    Trans = [[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]]
    Trans = np.array(Trans,dtype=float)
    F_new = np.matmul(Trans,F_old)
    return print('Trans =\n {0}\n\nF_new = Trans x F_old\n {1}\n'.format(Trans,F_new))
  
  def rot_x(theta_x):
    Rot_x_theta = np.array([[1,0,0,0],[0,math.cos(theta_x*math.pi/180),-math.sin(theta_x*math.pi/180),0],[0,math.sin(theta_x*math.pi/180),math.cos(theta_x*math.pi/180),0],[0,0,0,1]],dtype=float)
    return Rot_x_theta

  def rot_y(theta_y):
    Rot_y_theta = np.array([[math.cos(theta_y*math.pi/180),0,math.sin(theta_y*math.pi/180),0],[0,1,0,0],[-math.sin(theta_y*math.pi/180),0,math.cos(theta_y*math.pi/180),0],[0,0,0,1]],dtype=float)
    return Rot_y_theta

  def rot_z(theta_z):
    Rot_z_theta = np.array([[math.cos(theta_z*math.pi/180),-math.sin(theta_z*math.pi/180),0,0],[math.sin(theta_z*math.pi/180),math.cos(theta_z*math.pi/180),0,0],[0,0,1,0],[0,0,0,1]],dtype=float)
    return Rot_z_theta

  def rotation_transformation_z_y(theta_y,theta_z,Tx,Ty,Tz,n,o,a):

    P = np.array([n,o,a,1],dtype=float)
    Trans = [[1,0,0,Tx],[0,1,0,Ty],[0,0,1,Tz],[0,0,0,1]]
    Trans = np.array(Trans,dtype=float)
    Rot_Y = rot_y(theta_y)
    Rot_Z = rot_z(theta_z)
    R = Rot_Y @ Trans @ Rot_Z @ P
    R = np.around(R,4)
    return print('Trans =\n {0}\n'.format(R))