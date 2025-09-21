import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class RG():
    def __init__(self):
        self.init_integration_constants()
        self.init_combinatorical_constants()

    def init_integration_constants(self,R=np.pi,l_star=0.5*np.pi):
        """
        R is the radius in momentum space. can be 1, pi or other.

        l_star is l until which we integrate. should ne chose manually.
        """
        self.R = R
        self.l_star = l_star
        self.K2 = 1/(2*np.pi)

    def init_combinatorical_constants(self,
                                     C2a = 6, 
                                     C2b = 24, 
                                     C3a = 216, 
                                     C3b = 72, 
                                     C3c = 288, 
                                     C3g = 48):
        """
        set combinatorical factors.

        classic: 2,6,24,?,12,12

        computed: 6,24,216,72,288,48
        """
        self.C2a = C2a
        self.C2b = C2b
        self.C3a = C3a
        self.C3b = C3b
        self.C3c = C3c
        self.C3g = C3g

    def set_integration_constants(self,R=None,l_star=None):
        """
        R is the radius in momentum space. can be 1, pi or other.

        l_star is l until which we integrate. should ne chose manually.
        """
        if R != None: self.R = R
        if l_star != None: self.l_star = l_star

    def set_combinatorical_constants(self,
                                     C2a = None, 
                                     C2b = None, 
                                     C3a = None, 
                                     C3b = None, 
                                     C3c = None, 
                                     C3g = None):
        """
        set combinatorical factors.

        classic: 2,6,24,?,12,12

        computed: 6,24,216,72,288,48
        """
        if C2a != None: self.C2a = C2a
        if C2b != None: self.C2b = C2b
        if C3a != None: self.C3a = C3a
        if C3b != None: self.C3b = C3b
        if C3c != None: self.C3c = C3c
        if C3g != None: self.C3g = C3g

    def S1(self,x,D):
        a,b,c,Gamma = x
        return b*Gamma*self.K2*(self.R**2) / (16*((D*(self.R**2) + (a))**2))

    def S2(self,x,D):
        a,b,c,Gamma = x
        return self.K2*(self.R**2) / (2*(D*(self.R**2) + a))

    def dydl(self,t,x,D):
        a,b,c,Gamma = x
        S1 = self.S1(x,D)
        S2 = self.S2(x,D)
        return np.array([
            # 2*a + (-1.0*self.C2a*S1*a) + (-2.0*self.C2a*b*Gamma*S2),
            2*a + (-1.0*self.C2a*S1*a) + (-2.0*self.C2a*b*Gamma*S2) + ((2/3)*self.C3a*(Gamma*c/b)*S1),
            b + (-2*self.C2a*S1*b) + (-0.5*self.C2b*c*Gamma*S2),
            c*((-3*self.C2a*S1) + ((2/3)*self.C3c*S1)),
            Gamma*((-2*self.C2a*S1) + 1 + ((2/3)*self.C3g*S1))
        ])

    def solve(self,a,b,c,D,sigma,l_max=None, zero_event_fun=None):
        if l_max == None:
            l_max = self.R
        G = 0.5*(sigma**2)
        y0 = np.array([a,b,c,G])
        tspan = (0, l_max)
        t_eval = np.linspace(tspan[0], tspan[1], 50)
        args = np.array([D])
        sol = solve_ivp(self.dydl, t_span=tspan, y0=y0, t_eval=t_eval, args=args, events=zero_event_fun)
        return sol
    
    def zero_alpha_event(self,t,y,args):
        return y[0]

    def check_alphas(self,alphas, b,c,D,sigma,l_max, to_plot=True):
        sols = []
        for a in alphas:
            sols.append(self.solve(a,b,c,D,sigma,l_max))
        if to_plot:
            plt.figure()
            for i,sol in enumerate(sols):
                plt.plot(sol.t, sol.y[0], label=fr'$\alpha =${alphas[i]}')
            plt.legend()
            plt.hlines([0], 0, l_max, linestyles='dashed')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$\alpha_{ren}$')
            plt.title(fr'Renormalization for $\sigma_d ={sigma}$')
            plt.show()
        return sols

    def check_sigmas(self,a,b,c,D,sigmas,l_max, to_plot=True):
        sols = []
        for s in sigmas:
            sols.append(self.solve(a,b,c,D,s,l_max))
        if to_plot:
            plt.figure()
            for i,sol in enumerate(sols):
                plt.plot(sol.t, sol.y[0], label=fr'$\sigma_d =${sigmas[i]}')
            plt.legend()
            plt.hlines([0], 0, l_max, linestyles='dashed')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$\alpha_{ren}$')
            plt.title(fr'$\alpha={a}$ Renormalized')
            plt.show()
        return sols
    
    def check_alphas_sigmas(self,alphas,sigmas,b,c,D,l_max, to_plot=True):
        sols = []
        for a,s in zip(alphas, sigmas):
            sols.append(self.solve(a,b,c,D,s,l_max))
        if to_plot:
            plt.figure()
            for i,sol in enumerate(sols):
                plt.plot(sol.t, sol.y[0], 
                         label=fr'$\sigma_d =${sigmas[i]}, $\alpha =${alphas[i]}')
            plt.legend()
            plt.hlines([0], 0, l_max, linestyles='dashed')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$\alpha_{ren}$')
            plt.title('Renormalization')
            plt.show()
        return sols
    
    def find_a(self,b,c,D,sigma):
        tol = 1e-4
        a_max = (b**2 / (4*c))
        # a_min = -0.5 * a_max
        a_min = 0
        for i in range(1000):
            a = 0.5*(a_min + a_max)
            sol = self.solve(a,b,c,D,sigma,self.l_star)
            a0 = a

            a_end = sol.y[0,-1]

            if np.abs(a_end) < tol:
                break
            if np.abs(a_max - a_min) < 1e-5:
                # if a_max > 0:
                #     a0 = 10*(b**2 / (4*c))
                # else:
                #     a0 = -10*(b**2 / (4*c))
                break
            if a_end < 0:
                a_min = a0
            elif np.all(sol.y[0,:]) >=0:
                a_max = a0
            else:
                a_min = a0

            # print(a0)
            # plt.plot(sol.t, sol.y[0])
            # plt.show()
          
        return a0
    
    def find_a_dep_lstar(self,b,c,D,sigma):
        tol = 1e-4
        a_max = (b**2 / (4*c))
        # a_min = -0.5 * a_max
        a_min = 0
        for i in range(1000):
            a = 0.5*(a_min + a_max)
            self.l_star = l_star_func(a,b,c,D,sigma)
            sol = self.solve(a,b,c,D,sigma,self.l_star)
            a0 = a

            a_end = sol.y[0,-1]

            if np.abs(a_end) < tol:
                break
            if np.abs(a_max - a_min) < 1e-5:
                # if a_max > 0:
                #     a0 = 10*(b**2 / (4*c))
                # else:
                #     a0 = -10*(b**2 / (4*c))
                break
            if a_end < 0:
                a_min = a0
            elif np.all(sol.y[0,:]) >=0:
                a_max = a0
            else:
                a_min = a0

            # print(a0)
            # plt.plot(sol.t, sol.y[0])
            # plt.show()
          
        return a0
    


def l_star_func(a,b,c,D,sigma,params):
    return params[0]+ params[1]*np.log(sigma) + params[2]*np.log((b-np.sqrt(b**2 - (4*a*c)))/(2*c)) + params[3]*np.log(D)

    
class RG_DP(RG):
    def __init__(self):
        self.init_integration_constants()
        self.C1 = 1
        self.C2 = 1

    def S1(self,x,D):
        a,Gamma = x
        return ((Gamma**2) * self.K2 * (self.R**2))/ (16 * (((D*(self.R**2)) -a)**2))

    def S2(self,x,D):
        a,Gamma = x
        return ((Gamma**2) * self.K2 * (self.R**2)) / (4*a * ((D*(self.R**2)) -a))
        # return self.K2*(self.R**2) / (2*(D*(self.R**2) + a))

    def dydl(self,t,x,D):
        a,Gamma = x
        S1 = self.S1(x,D)
        S2 = self.S2(x,D)
        return np.array([
            a*(2+S1-S2),
            Gamma*(1-(6*S1))
        ])
    
    def solve(self,a,b,c,D,sigma,l_max=None, zero_event_fun=None):
        if l_max == None:
            l_max = self.R
        G = np.sqrt(2*(sigma**2)*b)
        y0 = np.array([a,G])
        tspan = (0, l_max)
        t_eval = np.linspace(tspan[0], tspan[1], 50)
        args = np.array([D])
        sol = solve_ivp(self.dydl, t_span=tspan, y0=y0, t_eval=t_eval, args=args, events=zero_event_fun)
        return sol
    
class RG_DP_Not_Hin(RG_DP):
    def dydl(self,t,x,D):
        a,Gamma = x
        S1 = self.S1(x,D)
        S2 = self.S2(x,D)
        return np.array([
            a*(2+self.C1*(S1-S2)),
            # Gamma*(1-(6*S1))
            Gamma*(1 - (8*self.C2*S1) + (2*self.C1*S1))
        ])