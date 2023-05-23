import numpy as np
import matplotlib.pyplot as plt
import sympy
import pygame



white = (255,255,255)
black = (0,0,0)
red = (255,0,0)
blue = (0,0,255)
orig = (400,200)


class double:

    def __init__(self,m1 = 0.1,m2 = 0.5,l1 = 150,l2 = 150):
        # Pendulum Parameters:
        self.m_1 = m1  # kg
        self.m_2 = m2  # kg
        self.l_1 = l1  # m
        self.l_2 = l2  # m
        self.g = 9.8  # m/s^2
        self.th1_vec = [1]
        self.dth1_vec = [1]
        self.th2_vec = [0]
        self.dth2_vec = [0]
        self.x1_vec = []
        self.y1_vec = []
        self.x2_vec = []
        self.y2_vec = []

    def plot(self,dt):
        # Create Symbols for Time:
        t = sympy.Symbol('t')  # Creates symbolic variable t

        # Create Generalized Coordinates as a function of time: q = [theta_1, theta_2]
        th1 = sympy.Function('th1')(t)
        th2 = sympy.Function('th2')(t)   

        #origini x axis, origin y axis

        # Position Equation: r = [x, y]
        # Position of first pendulum
        r1 = np.array([orig[0] + self.l_1 * sympy.sin(th1), orig[1] + self.l_1 * sympy.cos(th1)])
        # Position of second pendulum  
        r2 = np.array([self.l_2 * sympy.sin(th2) + r1[0], self.l_2 * sympy.cos(th2) + r1[1]])  

        # Velocity Equation: d/dt(r) = [dx/dt, dy/dt]
        v1 = np.array([r1[0].diff(t), r1[1].diff(t)])  # Velocity of first pendulum
        v2 = np.array([r2[0].diff(t), r2[1].diff(t)])  # Velocity of second pendulum

        # Energy Equations:
        T = 1/2 * self.m_1 * np.dot(v1, v1) + 1/2 * self.m_2 * np.dot(v2, v2)  # Kinetic Energy
        V = - (self.m_1 * self.g * r1[1] + self.m_2 * self.g * r2[1]) # Potential Energy
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

        sim_time = 100
        time = np.arange(0, sim_time, dt)
        sim_length = len(time)


        # Initialize A and B:
        A = np.array([[0, 0], [0, 0]])
        B = np.array([0, 0])

        # Euler Integration:
        for i in range(1, sim_length + 1):
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


            
        
        # plt.figure(1)
        # # 300 =< x1 <= 500
        # plt.plot(time, self.x1_vec, label = "Pendulum 1 - x axis")
        # # -100 =< y1 <= 100
        # plt.plot(time, self.y1_vec, label = "Pendulum 1 - y axis")
        # plt.legend()

        # plt.figure(2)
        # # 200 =< x2 <= 600
        # plt.plot(time, self.x2_vec, label = "Pendulum 2 - x axis")
        # # -200 =< y2 <= 200
        # plt.plot(time, self.y2_vec, label = "Pendulum 2 - y axis")
        # plt.legend()

        # plt.show()

    def simulate(self):
        # Pygame setup
        pygame.init()
        width, height = 800, 600
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()
        # Animation loop
        running = True
        index = 0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Draw the first pendulum
            x1, y1 = self.x1_vec[index], self.y1_vec[index]  # Get the x, y coordinates of the first pendulum at the current index
            x1,y1 = int(x1),int(y1) 
            pygame.draw.circle(screen, red, (x1 , y1), 10)

            #Draw the first rod
            pygame.draw.line(screen, black , orig, (int(x1) , int(y1)), 2)

            # Draw the second pendulum
            x2, y2 = self.x2_vec[index], self.y2_vec[index]  # Get the x, y coordinates of the second pendulum at the current index
            pygame.draw.circle(screen, blue, (int(x2) , int(y2)), 10)

            #Draw the second rod
            pygame.draw.line(screen, black , (int(x1) , int(y1)), (int(x2) , int(y2)),  2)

            
            # Frame rate
            clock.tick(120)  
            pygame.display.flip()
            screen.fill(white)

            # Update the index for the next frame
            index = index + 1

        pygame.quit()



def run():
    pendulum = double()
    pendulum.plot(0.001)
    pendulum.simulate()



if __name__ == "__main__":
    run()
