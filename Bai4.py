import math 
import matplotlib.pyplot as plt #for plotting
import numpy as np 
from numpy.linalg import inv

def f(y_old_1, y1, y2, h):
    return (y_old_1 + h * (y1 * (1 - y2)) - y1)
def g(y_old_2, y1, y2, h):
    return (y_old_2 + h * (y2 * (y1 - 1)) - y2)
# Defining the Jacobian Function
def jacobian(y_old_1, y_old_2, y1, y2, h):
    J = np.ones((2,2))
    dy = 1e-6

    J[0,0] = (f(y_old_1, y1 + dy, y2, h) - f(y_old_1, y1, y2, h))/dy
    J[0,1] = (f(y_old_1, y1, y2 + dy, h) - f(y_old_1, y1, y2, h))/dy

    J[1,0] = (g(y_old_2, y1 + dy, y2, h) - g(y_old_2, y1, y2, h))/dy
    J[1,1] = (g(y_old_2, y1, y2 + dy, h) - g(y_old_2, y1, y2, h))/dy 
    return J

def newtRhap(y1, y2, y1_guess, y2_guess, h):
	S_old = np.ones((2, 1))
	S_old[0] = y1_guess
	S_old[1] = y2_guess
	F = np.ones((2, 1))
	error = 9e9
	tol = 1e-9
	alpha = 1
	iter = 1

	while error > tol:
		J = jacobian(y1, y2, S_old[0], S_old[1], h)
		F[0] = f(y1,S_old[0], S_old[1], h) 
		F[1] = g(y2,S_old[0], S_old[1], h)
		S_new = S_old - alpha * (np.matmul(inv(J), F))
		error = np.max(np.abs(S_new - S_old))
		S_old = S_new
		iter = iter + 1

	return [S_new[0],S_new[1]] 

def implicit_euler(inty1, inty2, tspan, dt):
	t = np.arange(0, tspan,dt)
	y1 = np.zeros(len(t))
	y2 = np.zeros(len(t))
	y1[0] = inty1
	y2[0] = inty2
	
	for i in range(1, len(t)):
		y1[i] , y2[i]  = newtRhap(y1[i-1], y2[i-1], y1[i-1], y2[i-1], dt)

	return [t,y1,y2]

t,y1,y2 = implicit_euler(2,2,10,0.01)


plt.plot(t, y1,'b', label = 'Romeo')
plt.plot(t, y2,'g', label = 'Juliet')
plt.xlabel('Time')
plt.ylabel('Love for the other')
plt.title('Lotka-Volterra, $R_0$ = 2, $J_0$ = 2')
plt.legend()
plt.show()