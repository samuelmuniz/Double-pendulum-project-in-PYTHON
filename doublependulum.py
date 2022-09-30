from pickle import TRUE
import sympy as smp

g,t,m1,m2,l1,l2 = smp.symbols('g t m1 m2 l1 l2',real = TRUE)
theta1,theta2 = smp.symbols('theta_1 theta_2',real = TRUE,cls = smp.Function)

theta1 = theta1(t)
theta2 = theta2(t)

dtheta1dt = smp.diff(theta1,t)
dtheta2dt = smp.diff(theta2,t)

T = (0.5)*m1*(l1**2)*dtheta1dt**2 + 0.5*m2*((l1 * dtheta1dt)**2 + (l2 * dtheta2dt)**2 + 2*l1*l2*dtheta1dt*dtheta2dt*smp.cos(theta1 - theta2)) 
V = -(m1+m2)*g*l1*smp.cos(theta1) - m2*g*l2*smp.cos(theta2)

L = T - V

dtheta1dt = smp.diff(theta1,t)
dtheta2dt = smp.diff(theta2,t)

F1 = smp.diff(smp.diff(L,dtheta1dt),t) - smp.diff(L,theta1)
F2 = smp.diff(smp.diff(L,dtheta2dt),t) - smp.diff(L,theta2)
F1.simplify()
F2.simplify()

solve = smp.solve([F1,F2],[smp.diff(theta1,t,t),smp.diff(theta2,t,t)])

print(solve[smp.diff(theta1,t,t)])
print('#'*40)
print(F1)