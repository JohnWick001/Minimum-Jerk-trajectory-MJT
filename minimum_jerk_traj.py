import matplotlib.pyplot as plt

import numpy as np
import math as m



#single line


goal = {'x': 210., 'y': 380., 'l': 20}
goal1 = {'x': 210., 'y': 210., 'l': 20}    
dt = 0.001
m1 = 0.1; m2 = 0.1; 
x2=goal.get('x')-200
y2=goal.get('y')-200

x1=goal1.get('x')-200
y1=goal1.get('y')-200
      
l1=100
l2=100

the2_des=m.acos((x2*x2 + y2*y2 -l1*l1 -l2*l2)/(2*l1*l2))
the1_des=m.atan2(y2, x2)-m.atan2(l2*m.sin(the2_des), l1+l2*m.cos(the2_des))


the2_des1=m.acos((x1*x1 + y1*y1 -l1*l1 -l2*l2)/(2*l1*l2))
the1_des1=m.atan2(y1, x1)-m.atan2(l2*m.sin(the2_des1), l1+l2*m.cos(the2_des1))


duration=len(np.arange(210,381,1))

theta1_des=np.zeros(duration)
theta2_des=np.zeros(duration)

for t in range(0,duration):
    theta1_des[t]=the1_des1+(the1_des-the1_des1)*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5)
    theta2_des[t]=the2_des1+(the2_des-the2_des1)*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5)
    

 
#square


goal2 = {'x': 210., 'y': 380., 'l': 20}
goal1 = {'x': 115., 'y': 295., 'l': 20}    
goal = {'x': 210., 'y': 210., 'l': 20}    
goal3 = {'x': 285., 'y': 295., 'l': 20}  
dt = 0.001
m1 = 0.1; m2 = 0.1; 
x1=goal.get('x')-200
y1=goal.get('y')-200

x2=goal1.get('x')-200
y2=goal1.get('y')-200
x3=goal2.get('x')-200
y3=goal2.get('y')-200
x4=goal3.get('x')-200
y4=goal3.get('y')-200
      
l1=100
l2=100

the2_des1=m.acos((x2*x2 + y2*y2 -l1*l1 -l2*l2)/(2*l1*l2))
the1_des1=m.atan2(y2, x2)-m.atan2(l2*m.sin(the2_des1), l1+l2*m.cos(the2_des1))


the2_des=m.acos((x1*x1 + y1*y1 -l1*l1 -l2*l2)/(2*l1*l2))
the1_des=m.atan2(y1, x1)-m.atan2(l2*m.sin(the2_des), l1+l2*m.cos(the2_des))

the2_des2=m.acos((x3*x3 + y3*y3 -l1*l1 -l2*l2)/(2*l1*l2))
the1_des2=m.atan2(y3, x3)-m.atan2(l2*m.sin(the2_des2), l1+l2*m.cos(the2_des2))


the2_des3=m.acos((x4*x4 + y4*y4 -l1*l1 -l2*l2)/(2*l1*l2))
the1_des3=m.atan2(y4, x4)-m.atan2(l2*m.sin(the2_des3), l1+l2*m.cos(the2_des3))


duration=len(np.arange(210,381,1)/2)
duration=85
theta1_des=np.zeros(duration)
theta2_des=np.zeros(duration)
theta1_des1=np.zeros(duration)
theta2_des1=np.zeros(duration)
theta1_des2=np.zeros(duration)
theta2_des2=np.zeros(duration)
theta1_des3=np.zeros(duration)
theta2_des3=np.zeros(duration)

theta1=np.zeros(4*duration)
theta2=np.zeros(4*duration)

for t in range(0,duration):
    theta1_des[t]=the1_des+(the1_des1-the1_des)*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5)
    theta2_des[t]=the2_des+(the2_des1-the2_des)*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5)
for t in range(0,duration):
    theta1_des1[t]=the1_des1+(the1_des2-the1_des1)*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5)
    theta2_des1[t]=the2_des1+(the2_des2-the2_des1)*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5)
for t in range(0,duration):
    theta1_des2[t]=the1_des2+(the1_des3-the1_des2)*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5)
    theta2_des2[t]=the2_des2+(the2_des3-the2_des2)*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5)
for t in range(0,duration):
    theta1_des3[t]=the1_des3+(the1_des-the1_des3)*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5)
    theta2_des3[t]=the2_des3+(the2_des-the2_des3)*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5)

for i in range(4*duration):
    if(i<duration):
        theta1[i]=theta1_des[i]
        theta2[i]=theta2_des[i]
    elif(i<2*duration):
        theta1[i]=theta1_des1[i-duration]
        theta2[i]=theta2_des1[i-duration]
    elif(i<3*duration):
        theta1[i]=theta1_des2[i-2*duration]
        theta2[i]=theta2_des2[i-2*duration]
    elif(i<4*duration):
        theta1[i]=theta1_des3[i-3*duration]
        theta2[i]=theta2_des3[i-3*duration]


   
#circle




dt = 0.001
m1 = 0.1; m2 = 0.1; 

l1=100
l2=100


duration=10
number_of_points=5800

goalx=np.zeros(number_of_points)
goaly=np.zeros(number_of_points)

for i in range(number_of_points):
    #for j in range(duration):
    goalx[i]=90*(m.cos((np.radians(i*360/number_of_points))))+200#np.radians
    goaly[i]=90*(m.sin((np.radians(i*360/number_of_points))))+200
    
goalx=goalx-200
goaly=goaly-200
theta2=np.arccos((np.square(goalx)+ np.square(goaly) -l1*l1 -l2*l2)/(2*l1*l2))
theta1=np.arctan2(goaly, goalx)-np.arctan2(l2*np.sin(theta2), l1+l2*np.cos(theta2))

end1_eff1=np.zeros(number_of_points)
end1_eff2=np.zeros(number_of_points)

for t in range(0,number_of_points):
    (a1l, a2l) = [100,100]  # radius, arm length
    (a1r, a2r) = [theta1[t] ,theta2[t]]
    a1xy = np.array([200., 200.])    # a1 start (x0, y0)
    a1xy_ = np.array([np.cos(a1r), np.sin(a1r)]) * a1l + a1xy  # a1 end and a2 start (x1, y1)
    finger = np.array([np.cos(a1r + a2r), np.sin(a1r + a2r)]) * a2l + a1xy_
    end1_eff1[t]=(finger[0]/400)
    end1_eff2[t]=(finger[1]/400)

fig, ax = plt.subplots()
#ax.set_title('Trajectory'+' [ Kp='+str(env.kp)+' Kd='+str(env.kd)+' ]')
ax.plot(end1_eff1, end1_eff2)
ax.plot((goalx+200)/400,(goaly+200)/400,linestyle = 'dotted')
plt.plot(end1_eff1[0], end1_eff2[0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
plt.plot(end1_eff1[0], end1_eff2[0], marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
#plt.plot(env.goal1.get('x')/400, env.goal1.get('y')/400, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
#plt.plot(env.goal2.get('x')/400, env.goal2.get('y')/400, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
#plt.plot(env.goal3.get('x')/400, env.goal3.get('y')/400, marker="o", markersize=5, markeredgecolor="red", markerfacecolor="green")
#ax.plot(np.arange(time+2), yd)
#plt.xlim([0, 1])
#plt.ylim([0, 1])
plt.xlabel('x')
plt.ylabel('y')
plt.show()    


"""
for t in range(duration):
    trajectory_theta1[t]=(theta1[int(t)]+(theta1[int(t)+1]-theta1[int(t)])*(10*(t)**3-15*(t)**4+6*(t)**5))
    trajectory_theta2[t]=(theta2[int(t)]+(theta2[int(t)+1]-theta2[int(t)])*(10*(t)**3-15*(t)**4+6*(t)**5))
"""


"""
for t in range((number_of_points-1)*duration):
    trajectory_theta1[t]=(theta1[int(t/duration)]+(theta1[int(t/duration)+1]-theta1[int(t/duration)])*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5))%2*m.pi
    trajectory_theta2[t]=(theta2[int(t/duration)]+(theta2[int(t/duration)+1]-theta2[int(t/duration)])*(10*(t/duration)**3-15*(t/duration)**4+6*(t/duration)**5))%2*m.pi

"""





