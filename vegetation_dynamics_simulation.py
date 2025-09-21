import numpy as np
import time
import multiprocessing
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

class Solver():
    def __init__(self,param):
        self.a = param.a
        self.b = param.b
        self.c = param.c
        self.D = param.D
        self.lambda_m = 2 / (param.dt * param.sigma_d**2)
        self.dt = param.dt
        self.NT = int(param.T / param.dt)
        
    def fft2d(self,t,u):
        v = np.shape(u)
        Nx = v[1]
        Ny = v[0]
        uhat = np.fft.fft2(u)
        kap1x = 2*np.pi*np.fft.fftfreq(Nx,1)
        kap1y = 2*np.pi*np.fft.fftfreq(Ny,1)     
        a1 = np.outer(np.exp(2 * t * (np.cos(kap1y) - 1)), np.exp(2 * t * (np.cos(kap1x) - 1)))
        return np.real(np.fft.ifft2(a1 * uhat))
    
    def step(self, u):
        # Deterministic step (Euler method)
        u = np.maximum(0, u + self.dt * (- self.a * u + self.b * u**2 - self.c * u**3))           
        # Diffusion using spectral
        u = np.maximum(0, self.fft2d(self.D * self.dt, u))
        # Demographic stochasticity step
        Q = np.random.poisson(self.lambda_m * u)
        R = np.random.gamma(Q, 1)
        u = R / self.lambda_m
        return u
    
    def solve(self,u):
        u_arr = np.zeros((self.NT+1,u.shape[0],u.shape[1]))
        u_arr[0,:,:] = u
        for k in range(1,self.NT+1):
            u = self.step(u)
            u_arr[k,:,:] = u
        return u_arr
    

class D0_Solver(Solver):
    def step(self,u):
        # Deterministic step (Euler method)
        u = np.maximum(0, u + self.dt * (- self.a * u + self.b * u**2 - self.c * u**3))           
        # Demographic stochasticity step
        Q = np.random.poisson(self.lambda_m * u)
        R = np.random.gamma(Q, 1)
        u = R / self.lambda_m
        return u
    
    
class Density_Solver(Solver):
    def solve(self, u):
        den = np.zeros(self.NT+1)
        den[0] = np.sum(u) / (u.shape[0]*u.shape[1])
        for k in range(1,self.NT+1):
            u = self.step(u)
            den[k] = np.sum(u) / (u.shape[0]*u.shape[1])
        return den
    

class DP_Solver(Solver):
    def step(self, u):
        # Deterministic step (Euler method)
        u = np.maximum(0, u + self.dt * ((self.a * u) - (self.b * u**2)))           
        # Diffusion using spectral
        u = np.maximum(0, self.fft2d(self.D * self.dt, u))
        # Demographic stochasticity step
        Q = np.random.poisson(self.lambda_m * u)
        R = np.random.gamma(Q, 1)
        u = R / self.lambda_m
        return u
    
class DP_Solver_Density(DP_Solver):
    def solve(self, u):
        den = np.zeros(self.NT+1)
        den[0] = np.sum(u) / (u.shape[0]*u.shape[1])
        for k in range(1,self.NT+1):
            u = self.step(u)
            den[k] = np.sum(u) / (u.shape[0]*u.shape[1])
        return den
    
    
class Veg_System():
    def __init__(self,L,mn,sl,n):
        np.random.seed(n)
        if sl==0:
            self.u = np.ones((L,L))*mn
        else:
            self.u = np.abs(np.random.normal(loc=mn, scale=sl, size=(L, L)))


class Param():
    def __init__(self,a,b,c,D,sigma_d,T,dt):
        self.a = a
        self.b = b
        self.c = c
        self.D = D
        self.sigma_d = sigma_d
        self.T = T
        self.dt = dt


class Simulation():
    def __init__(self):
        pass

    def set_param(self, param):
        self.param = param

    def set_system(self, L, mn, sl):
        self.L = L
        self.mn = mn
        self.sl = sl

    def set_solver(self, solver):
        self.solver = solver

    def run(self, n_res, n_pros=12):
        res = [None] * n_res

        # Parallel processing
        pool = multiprocessing.Pool(processes=n_pros)
        results = [pool.apply_async(self.worker, [n]) for n in range(n_res)]
        pool.close()
        pool.join()

        for n,r in enumerate(results):
            res[n] = r.get()

        self.res = res

    def get_results(self):
        return self.res
    
    def worker(self,n):
        u = Veg_System(self.L,self.mn,self.sl,n).u
        return self.solver.solve(u)

        
# class Simulation_Detailed(Simulation):
#     def __init__(self):
#         super().__init__()
    
#     def worker(self,n):
#         solver = Solver(self.param)
#         u = Veg_System(self.L,self.mn,self.sl,n).u
#         return solver.solve(u)
    
#     def get_results(self):
#         return super().get_results()


# class Simulation_0D(Simulation_Detailed):
#     def worker(self,n):
#         solver = D0_Solver(self.param)
#         u = Veg_System(self.L,self.mn,self.sl,n).u
#         return solver.solve(u)
    

class Simulation_Density(Simulation):
    # def __init__(self):
    #     super().__init__()

    # def worker(self,n):
    #     solver = Density_Solver(self.param)
    #     u = Veg_System(self.L,self.mn,self.sl,n).u
    #     return solver.solve(u)

    def set_param(self, param):
        super().set_param(param)
        self.solver = Density_Solver(param)
    
    def get_results(self):
        den = np.zeros((int(self.param.T/self.param.dt)+1, len(self.res)))
        for n, r in enumerate(self.res):
            den[:, n] = r
        return den


def find_a_ltp(L,sigma_d,a0,n_res,mn=0,var=0.00001,b=1,c=1,D=1.0,dt=0.1,T=5):
    # Parameters
        # mn: mean population density to generate dilute initial condition
        # var: variance of population density to generate dilute initial condition
        # b: constant (feedback)
        # c: constant (carrying capacity)
        # D: Diffusion constant
        # dt: Small time interval 
        # L: Length of the System
        # T: Total run time
        # sigma_d: Demographic noise strength
    
    n_pros = 12
    sl = np.sqrt(var)
    thresh = 2e-5

    all_a = []
    all_q = []

    first = second = True
    max_counter = 7
    counter = 0
    while counter < max_counter:
        counter = counter+1
        if first:
            a = a0[0]

        param = Param(a,b,c,D,sigma_d,T,dt)

        sim = Simulation_Density()
        sim.set_param(param)
        sim.set_system(L,mn,sl)
        sim.run(n_res, n_pros=n_pros)
        den = sim.get_results()

        q = check_neutral(den)
        def fit_func(x,m): return m*x
        m = curve_fit(fit_func,np.arange(0,T+0.5*dt,dt),q)[0][0]

        all_a.append(a)
        all_q.append(q)

        if np.abs(m) < thresh:
            break

        if first:
            m_down = m
            first = False
            a_down = a
            a = a0[1]
            continue
        elif second:
            m_up = m
            second = False
            a_up = a
        else:
            if m>0:
                a_up = a
                m_up = m
            else:
                a_down = a
                m_down = m

        a = a_down - ((a_up - a_down) * m_down/(m_up-m_down))
    
    return all_a, all_q

def find_a_ltp_0D(L,sigma_d,a0,n_res,mn=0,var=0.00001,b=1,c=1,D=1.0,dt=0.1,T=5):
    # Parameters
        # mn: mean population density to generate dilute initial condition
        # var: variance of population density to generate dilute initial condition
        # b: constant (feedback)
        # c: constant (carrying capacity)
        # D: Diffusion constant
        # dt: Small time interval 
        # L: Length of the System
        # T: Total run time
        # sigma_d: Demographic noise strength
    
    n_pros = 12
    sl = np.sqrt(var)
    thresh = 2e-5

    all_a = []
    all_q = []

    first = second = True
    max_counter = 7
    counter = 0
    while counter < max_counter:
        counter = counter+1
        if first:
            a = a0[0]

        param = Param(a,b,c,D,sigma_d,T,dt)

        sim = Simulation()
        sim.set_param(param)
        sim.set_system(L,mn,sl)
        sim.set_solver(D0_Solver(sim.param))
        sim.run(n_res, n_pros=n_pros)
        res = sim.get_results()

        den = np.zeros((int(T/dt)+1, n_res))
        for n, r in enumerate(res):
            den[:, n] = r.flatten()

        q = check_neutral(den)
        def fit_func(x,m): return m*x
        m = curve_fit(fit_func,np.arange(0,T+0.5*dt,dt),q)[0][0]

        all_a.append(a)
        all_q.append(q)

        if np.abs(m) < thresh:
            break

        if first:
            m_down = m
            first = False
            a_down = a
            a = a0[1]
            continue
        elif second:
            m_up = m
            second = False
            a_up = a
        else:
            if m>0:
                a_up = a
                m_up = m
            else:
                a_down = a
                m_down = m

        a = a_down - ((a_up - a_down) * m_down/(m_up-m_down))
    
    return all_a, all_q


def check_neutral(den):
    q = np.zeros(den.shape[0])
    for d in range(1,den.shape[0]):
        d_den = (den[d:,:]-den[:-d,:]) / np.sqrt(den[:-d,:])
        q[d] = np.nanmean(d_den.flatten())
    return q

def find_a_ltp_DP(L,sigma_d,a0,n_res,mn=1e-3,var=0,b=1,c=0,D=1.0,dt=0.1,T=5):
    # Parameters
        # mn: mean population density to generate dilute initial condition
        # var: variance of population density to generate dilute initial condition
        # b: constant (feedback)
        # c: constant (carrying capacity)
        # D: Diffusion constant
        # dt: Small time interval 
        # L: Length of the System
        # T: Total run time
        # sigma_d: Demographic noise strength
    
    n_pros = 12
    sl = np.sqrt(var)
    thresh = 2e-5

    all_a = []
    all_q = []

    first = second = True
    max_counter = 7
    counter = 0
    while counter < max_counter:
        counter = counter+1
        if first:
            a = a0[0]

        param = Param(a,b,c,D,sigma_d,T,dt)

        sim = Simulation_Density()
        sim.set_param(param)
        sim.set_system(L,mn,sl)
        sim.set_solver(DP_Solver_Density(sim.param))
        sim.run(n_res, n_pros=n_pros)
        den = sim.get_results()

        q = check_neutral(den)
        def fit_func(x,m): return m*x
        m = curve_fit(fit_func,np.arange(0,T+0.5*dt,dt),q)[0][0]

        all_a.append(a)
        all_q.append(q)

        if np.abs(m) < thresh:
            break

        if first:
            m_down = m
            first = False
            a_down = a
            a = a0[1]
            continue
        elif second:
            m_up = m
            second = False
            a_up = a
        else:
            if m>0:
                a_up = a
                m_up = m
            else:
                a_down = a
                m_down = m

        a = a_down - ((a_up - a_down) * m_down/(m_up-m_down))
    
    return all_a, all_q


# def find_a_ltp_beta(L,sigma_d,a0,n_res,mn=0,var=0.00001,b=1,c=1,D=1.0,dt=0.1,T=5):
#     # Parameters
#         # mn: mean population density to generate dilute initial condition
#         # var: variance of population density to generate dilute initial condition
#         # b: constant (feedback)
#         # c: constant (carrying capacity)
#         # D: Diffusion constant
#         # dt: Small time interval 
#         # L: Length of the System
#         # T: Total run time
#         # sigma_d: Demographic noise strength
    
#     n_pros = 12
#     sl = np.sqrt(var)
#     thresh = 2e-5

#     all_a = []
#     all_q = []
#     all_dd = []

#     first = second = True
#     max_counter = 7
#     counter = 0
#     while counter < max_counter:
#         counter = counter+1
#         if first:
#             a = a0[0]

#         param = Param(a,b,c,D,sigma_d,T,dt)

#         sim = Simulation_Density()
#         sim.set_param(param)
#         sim.set_system(L,mn,sl)
#         sim.run(n_res, n_pros=n_pros)
#         den = sim.get_results()

#         q = check_neutral(den)
#         def fit_func(x,m): return m*x
#         m = curve_fit(fit_func,np.arange(0,T+0.5*dt,dt),q)[0][0]

#         all_a.append(a)
#         all_q.append(q)
#         all_dd.append(check_neutral_2(den))

#         if np.abs(m) < thresh:
#             break

#         if first:
#             m_down = m
#             first = False
#             a_down = a
#             a = a0[1]
#             continue
#         elif second:
#             m_up = m
#             second = False
#             a_up = a
#         else:
#             if m>0:
#                 a_up = a
#                 m_up = m
#             else:
#                 a_down = a
#                 m_down = m

#         a = a_down - ((a_up - a_down) * m_down/(m_up-m_down))
    
#     return all_a, all_q, all_dd

# def check_neutral_2(den):
#     dd = den[1:,:]-den[:-1,:]
#     return np.nanmean(dd.flatten())

    
if __name__ == '__main__':
    # L = 2**8
    # sigma_d = 0.2
    # alim = [0,0.02]
    # n_res = 10
    # find_a_ltp(L,sigma_d,alim,n_res)

    n_res = 10
    sigma_d = 0.2
    a0 = [0,0.05]
    L=1
    all_a, all_q = find_a_ltp_0D(L,sigma_d,a0,n_res)
    plt.figure
    for j,a in enumerate(all_a):
        plt.plot(np.arange(0.1,5.05,0.1), all_q[j])
    plt.legend(all_a)
    plt.hlines([0.0], 0, 5, linestyles='dotted')
    plt.show()
    a_star = all_a[-1]
    print(a_star)
    
# class Detailed_Solver(Solver):
#     def solve(self, u):
#         u_arr = np.zeros((NT,u.shape[0],u.shape[1]))
#         for k in range(self.NT):
#             u = self.step(u)
#             u_arr[k,:,:] = u
#         return u_arr
    

# # Function for Fast Fourier Transform (on 2D Lattice)
# def fft2d(t, u):
#     v = np.shape(u)
#     Nx = v[1]
#     Ny = v[0]
#     uhat = np.fft.fft2(u)
#     kap1x = 2*np.pi*np.fft.fftfreq(Nx,1)
#     kap1y = 2*np.pi*np.fft.fftfreq(Ny,1)     
#     a1 = np.outer(np.exp(2 * t * (np.cos(kap1y) - 1)), np.exp(2 * t * (np.cos(kap1x) - 1)))
#     return np.real(np.fft.ifft2(a1 * uhat))

# # Worker function for parallel processing
# def worker(n, NT, L, mn, sl, a, b, c, D, dt, lambda_m):
#     np.random.seed(n)
#     u = np.abs(np.random.normal(loc=mn, scale=sl, size=(L, L)))
#     u[u < 0] = 0
#     den = np.zeros(NT)
#     for k in range(NT):

#         # Deterministic step (Euler method)
#         u = np.maximum(0, u + dt * (- a * u + b * u**2 - c * u**3))           
#         # Diffusion using spectral
#         u = np.maximum(0, fft2d(D * dt, u))
#         # Demographic stochasticity step
#         Q = np.random.poisson(lambda_m * u)
#         R = np.random.gamma(Q, 1)
#         u = R / lambda_m

#         den[k] = np.sum(u) / L**2

#     return den

# def detailed_worker(n, NT, L, mn, sl, a, b, c, D, dt, lambda_m):
#     np.random.seed(n)
#     u = np.abs(np.random.normal(loc=mn, scale=sl, size=(L, L)))
#     u[u < 0] = 0
#     u_arr = np.zeros((NT,L,L))
#     for k in range(NT):

#         # Deterministic step (Euler method)
#         u = np.maximum(0, u + dt * (- a * u + b * u**2 - c * u**3))           
#         # Diffusion using spectral
#         u = np.maximum(0, fft2d(D * dt, u))
#         # Demographic stochasticity step
#         Q = np.random.poisson(lambda_m * u)
#         R = np.random.gamma(Q, 1)
#         u = R / lambda_m

#         u_arr[k,:,:] = u

#     return u_arr


# if __name__ == '__main__':
#     # Parameters
#     mn = 0                                        # mean population density to generate dilute initial condition
#     var = 0.00001                                 # variance of population density to generate dilute initial condition
#     sl = np.sqrt(var)                             # standard deviation associated with the above variance
#     b = 1                                         # constant (feedback)
#     c = 1                                         # constant (carrying capacity)
#     D = 1.0                                       # Diffusion constant
#     dt = 0.1                                      # Small time interval 
#     L = 2**8                                      # Length of the System
#     T = 20.2                                      # Total run time
#     NT = int(T / dt)                              # Total number of time-step
#     sigma_d = 0.4                                 # Demographic noise strength
#     lambda_m = 2 / (dt * sigma_d**2)              # constant to generate random number associated with demographic noise

#     # Data for 'a' values
#     # a_min = 0.0608
#     # a_max = 0.0621
#     # a_values = np.arange(a_min, a_max, 0.001)
#     a_values = np.array([0.0608, 0.0609])
    
#     n_pros = 12                                       # Number of processor involve in parallel computation 
#     n_res = n_pros * 100                              # Total number of realizations                                      

#     for a in a_values:
#         # Time initialization
#         s_ti = time.time()

#         # Initialize history
#         den = np.zeros((NT, n_res))

#         # Parallel processing
#         pool = multiprocessing.Pool(processes=n_pros)
#         results = [pool.apply_async(worker, (n, NT, L, mn, sl, a, b, c, D, dt, lambda_m)) for n in range(n_res)]
#         pool.close()
#         pool.join()

#         for n, res in enumerate(results):
#             den[:, n] = res.get()

#         # Save history_avg to file
#         path = f'data_ltp/dent_L{L}_D{D:.2f}_s{sigma_d:.2f}_a{a:.4f}_var{var:.5f}.npy'
#         data = np.column_stack(((np.arange(1, NT + 1) * dt), den[:, 0:n_res]))
#         np.save(path, data)    

#         # Finish time
#         f_ti = time.time()
#         e_ti = (f_ti - s_ti)/60.
#         print(f"Saved data for a={a:.4f}, sd={sigma_d:.2f}. Time taken: {e_ti:.3f} mins")



# old versions of find_a_ltp

# def find_a_ltp(L,sigma_d,a0,n_res,mn=0,var=0.00001,b=1,c=1,D=1.0,dt=0.1,T=5):
#     # Parameters
#         # mn: mean population density to generate dilute initial condition
#         # var: variance of population density to generate dilute initial condition
#         # b: constant (feedback)
#         # c: constant (carrying capacity)
#         # D: Diffusion constant
#         # dt: Small time interval 
#         # L: Length of the System
#         # T: Total run time
#         # sigma_d: Demographic noise strength
    
#     n_pros = 12
#     sl = np.sqrt(var)
#     thresh = 2e-5

#     all_a = []
#     all_q = []

#     first = second = True
#     max_counter = 7
#     counter = 0
#     while counter < max_counter:
#         counter = counter+1
#         if first:
#             a = a0[0]

#         param = Param(a,b,c,D,sigma_d,T,dt)

#         sim = Simulation_Density()
#         sim.set_param(param)
#         sim.set_system(L,mn,sl)
#         sim.run(n_res, n_pros=n_pros)
#         den = sim.get_results()

#         # # # Time initialization
#         # # s_ti = time.time()

#         # # Initialize history
#         # den = np.zeros((int(T/dt), n_res))

#         # # Parallel processing
#         # pool = multiprocessing.Pool(processes=n_pros)
#         # results = [pool.apply_async(dens_worker, (L,mn,sl,n,param)) for n in range(n_res)]
#         # pool.close()
#         # pool.join()

#         # for n, res in enumerate(results):
#         #     den[:, n] = res.get()

#         # # # Save history_avg to file
#         # # # details = np.array([['a',a],['b',b],['c',c],['D',D],['sigma_d',sigma_d],['T',T],['dt',dt]])
#         # # details = {'a': a, 'b': b, 'c': c, 'D': D, 'sigma_d': sigma_d, 'T': T, 'dt': dt, 'mean': mn, 'var': var}
#         # # path = f'data_ltp/L{L}_a{a:.4f}_D{D:.2f}_s{sigma_d:.2f}.npz'
#         # # np.savez(path, t=np.arange(dt, T+0.5*dt, dt), data=den, details=details)

#         # # # Finish time
#         # # f_ti = time.time()
#         # # e_ti = (f_ti - s_ti)/60.
#         # # print(f"Saved data for a={a:.4f}, sd={sigma_d:.2f}. Time taken: {e_ti:.3f} mins")

#         q = check_neutral(den)
#         def fit_func(x,m): return m*x
#         m = curve_fit(fit_func,np.arange(dt,T+0.5*dt,dt),q)[0][0]

#         all_a.append(a)
#         all_q.append(q)

#         if np.abs(m) < thresh:
#             break

#         if first:
#             m_down = m
#             first = False
#             a_down = a
#             a = a0[1]
#             continue
#         elif second:
#             m_up = m
#             second = False
#             a_up = a
#         else:
#             if m>0:
#                 a_up = a
#                 m_up = m
#             else:
#                 a_down = a
#                 m_down = m
#         # qq = np.nanmean(q_down/(q_up-q_down))
#         # da = (qq*(a_up-a_down))
#         # a = a_down-da

#         a = a_down - ((a_up - a_down) * m_down/(m_up-m_down))
        
#         # if np.abs(a_down-a_up) < thresh:
#         #     break
    
#     return all_a, all_q


# def dens_worker(L,mn,sl,n,param):
#     solver = Density_Solver(param)
#     u = Veg_System(L,mn,sl,n).u
#     return solver.solve(u)

# def det_worker(L,mn,sl,n,param):
#     solver = Solver(param)
#     u = Veg_System(L,mn,sl,n).u
#     return solver.solve(u)