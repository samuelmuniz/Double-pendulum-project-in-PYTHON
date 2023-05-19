import numpy as np
import matplotlib.pyplot as plt
import sympy


class double:

    def __init__(self,m1 = 1,m2 = 1,l1 = 1,l2 = 1):
        # Pendulum Parameters:
        self.m_1 = m1  # kg
        self.m_2 = m2  # kg
        self.l_1 = l1  # m
        self.l_2 = l2  # m
        self.g = 9.8  # m/s^2
        self.th1_vec = [1]
        self.dth1_vec = [0]
        self.th2_vec = [0]
        self.dth2_vec = [1]
        self.x1_vec = []
        self.y1_vec = []
        self.x2_vec = []
        self.y2_vec = []

    def simulation(self,dt):
        # Create Symbols for Time:
        t = sympy.Symbol('t')  # Creates symbolic variable t

        # Create Generalized Coordinates as a function of time: q = [theta_1, theta_2]
        th1 = sympy.Function('th1')(t)
        th2 = sympy.Function('th2')(t)   

        # Position Equation: r = [x, y]
        r1 = np.array([self.l_1 * sympy.sin(th1), -self.l_1 * sympy.cos(th1)])  # Position of first pendulum
        r2 = np.array([self.l_2 * sympy.sin(th2) + r1[0], -self.l_2 * sympy.cos(th2) + r1[1]])  # Position of second pendulum

        # Velocity Equation: d/dt(r) = [dx/dt, dy/dt]
        v1 = np.array([r1[0].diff(t), r1[1].diff(t)])  # Velocity of first pendulum
        v2 = np.array([r2[0].diff(t), r2[1].diff(t)])  # Velocity of second pendulum

        # Energy Equations:
        T = 1/2 * self.m_1 * np.dot(v1, v1) + 1/2 * self.m_2 * np.dot(v2, v2)  # Kinetic Energy
        V = self.m_1 * self.g * r1[1] + self.m_2 * self.g * r2[1] # Potential Energy
        L = T - V  # Lagrangian

        # Lagrange Terms:
        dL_dth1 = L.diff(th1)
        dL_dth1_dt = L.diff(th1.diff(t)).diff(t)
        dL_dth2 = L.diff(th2)
        dL_dth2_dt = L.diff(th2.diff(t)).diff(t)

        # Euler-Lagrange Equations: dL/dq - d/dt(dL/ddq) = 0
        th1_eqn = dL_dth1 - dL_dth1_dt
        th2_eqn = dL_dth2 - dL_dth2_dt

        # Replace Time Derivatives and Functions with Symbolic Variables:
        replacements = [(th1.diff(t).diff(t), sympy.Symbol('ddth1')),
                        (th1.diff(t), sympy.Symbol('dth1')), 
                        (th1, sympy.Symbol('th1')), 
                        (th2.diff(t).diff(t), sympy.Symbol('ddth2')), 
                        (th2.diff(t), sympy.Symbol('dth2')), 
                        (th2, sympy.Symbol('th2'))]

        th1_eqn = th1_eqn.subs(replacements)
        th2_eqn = th2_eqn.subs(replacements)
        r1 = r1[0].subs(replacements), r1[1].subs(replacements)
        r2 = r2[0].subs(replacements), r2[1].subs(replacements)

        # Simplfiy then Force SymPy to Cancel factorization: [Sometimes needed to use .coeff()]
        th1_eqn = sympy.simplify(th1_eqn)
        th2_eqn = sympy.simplify(th2_eqn)
        th1_eqn = th1_eqn.cancel()
        th2_eqn = th2_eqn.cancel()

        # Solve for Coefficients for A * x = B where x = [ddth1 ddth2]
        A1 = th1_eqn.coeff(sympy.Symbol('ddth1'))
        A2 = th1_eqn.coeff(sympy.Symbol('ddth2'))
        A3 = th2_eqn.coeff(sympy.Symbol('ddth1'))
        A4 = th2_eqn.coeff(sympy.Symbol('ddth2'))

        # Multiply remaining terms by -1 to switch to other side of equation: A * x - B = 0 -> A * x = B
        remainder = [(sympy.Symbol('ddth1'), 0), (sympy.Symbol('ddth2'), 0)]
        B1 = -1 * th1_eqn.subs(remainder)
        B2 = -1 * th2_eqn.subs(remainder)

        # Generate Lambda Functions for A and B and Position Equations:
        replacements = (sympy.Symbol('th1'), 
                        sympy.Symbol('dth1'), 
                        sympy.Symbol('th2'), 
                        sympy.Symbol('dth2'))

        A1 = sympy.utilities.lambdify(replacements, A1, "numpy")
        A2 = sympy.utilities.lambdify(replacements, A2, "numpy")
        A3 = sympy.utilities.lambdify(replacements, A3, "numpy")
        A4 = sympy.utilities.lambdify(replacements, A4, "numpy")
        B1 = sympy.utilities.lambdify(replacements, B1, "numpy")
        B2 = sympy.utilities.lambdify(replacements, B2, "numpy")
        r1 = sympy.utilities.lambdify(replacements, r1, "numpy")
        r2 = sympy.utilities.lambdify(replacements, r2, "numpy")

        sim_time = 10
        time = np.arange(0, sim_time, dt)
        sim_length = len(time)


        # Initialize A and B:
        A = np.array([[0, 0], [0, 0]])
        B = np.array([0, 0])

        # Euler Integration:
        for i in range(1, sim_length):
            # Animation States:
            x1_vec, y1_vec = r1(self.th1_vec[i-1], self.dth1_vec[i-1], self.th2_vec[i-1], self.dth2_vec[i-1])
            self.x1_vec.append(x1_vec)
            self.y1_vec.append(y1_vec) 
            x2_vec, y2_vec = r2(self.th1_vec[i-1], self.dth1_vec[i-1], self.th2_vec[i-1], self.dth2_vec[i-1])
            self.x2_vec.append(x2_vec)
            self.y2_vec.append(y2_vec) 
            # Evaluate Dynamics:
            A[0, 0] = A1(self.th1_vec[i-1], self.dth1_vec[i-1], self.th2_vec[i-1], self.dth2_vec[i-1])
            A[0, 1] = A2(self.th1_vec[i-1], self.dth1_vec[i-1], self.th2_vec[i-1], self.dth2_vec[i-1])
            A[1, 0] = A3(self.th1_vec[i-1], self.dth1_vec[i-1], self.th2_vec[i-1], self.dth2_vec[i-1])
            A[1, 1] = A4(self.th1_vec[i-1], self.dth1_vec[i-1], self.th2_vec[i-1], self.dth2_vec[i-1])
            B[0] = B1(self.th1_vec[i-1], self.dth1_vec[i-1], self.th2_vec[i-1], self.dth2_vec[i-1])
            B[1] = B2(self.th1_vec[i-1], self.dth1_vec[i-1], self.th2_vec[i-1], self.dth2_vec[i-1])
            [ddth1, ddth2] = np.linalg.solve(A, B)
            # Euler Step Integration:
            self.th1_vec.append(self.th1_vec[i-1] + self.dth1_vec[i-1] * dt)
            self.dth1_vec.append(self.dth1_vec[i-1] + ddth1 * dt)
            self.th2_vec.append(self.th2_vec[i-1] + self.dth2_vec[i-1] * dt)
            self.dth2_vec.append(self.dth2_vec[i-1] + ddth2 * dt)
            

def run():
    pendulum = double()
    pendulum.simulation(0.001)



if __name__ == "__main__":
    run()
