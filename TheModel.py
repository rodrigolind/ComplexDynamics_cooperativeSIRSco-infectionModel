
from  Tools import *

####################################################################################################

class model:
    error_printed = 0
    
    def __init__(self, y0, beta, alpha, gamma, nu, tmin, tmax, stepsize, maxstep, direction, s, w, seasonalshift, epsilon_m, mmax, hthres, tau, distinguishable = True, xi=1, phi=1, tolerances=[1e-10,1e-13], reduced_output=False):
        
        self.y0 = y0

        # vectors (different for A and B)
        def set_parameter(param, name):
            if isinstance(param, list):
                setattr(self, name, np.array(param))
            elif isinstance(param, int) or isinstance(param, float):
                setattr(self, name, np.array([param, param]))
            else:
                print(f'ERROR: {name} has to be int, float or list... Not {type(param)}')

        set_parameter(beta, 'beta')
        set_parameter(alpha, 'alpha')
        set_parameter(gamma, 'gamma')
        set_parameter(nu, 'nu')
        set_parameter(mmax, 'mmax')
        
        self.tmin = tmin
        self.tmax = tmax
        self.stepsize = stepsize
        self.maxstep = maxstep
        self.direction = direction
        self.s = s
        self.w = w
        self.seasonalshift = seasonalshift
        self.epsilon_m = epsilon_m
        self.hthres = hthres
        self.tau = tau

        self.distinguishable = distinguishable

        self.xi = xi
        self.phi = phi

        self.tolerances = tolerances
        self.reduced_output = reduced_output

        
        
    # def r_eff(self,t,y):
    #     return self.beta/self.gamma*self.seasonalforcing(t)*self.softplus(y[4])*y[0]-1

    def seasonalforcing(self,t):
        return (1+self.s*np.cos(self.w*(t-self.seasonalshift)))

    def m_normalized(self,H):
        return 1-self.epsilon_m*np.log(np.exp(1/self.epsilon_m*(self.hthres-H))+1)/self.hthres
        
    def fun(self,t,y):
        
        if(self.distinguishable == True): 
            S,A,B,AB,Ab,aB,a,b,ab,H_A,H_A1,H_B,H_B1 = y
            H_list = np.array([H_A, H_B])
        elif(self.distinguishable == False): 
            S,A,B,AB,Ab,aB,a,b,ab,H,H1 = y
            H_list = np.array([H, H]) # both diseases have same Hazard
        else: raise RuntimeError("distinguishable must be True or False")

        Xi_list = np.array([1, self.xi]) # Mitigation diminution factor xi on disease B

        # FoI:
        X_A = A+AB+Ab
        X_B = B+AB+aB
        X_list = np.array([X_A, X_B])
        FOI_alpha = self.alpha*self.seasonalforcing(t)*(1-Xi_list*self.mmax*self.m_normalized(H_list))*X_list
        FOI_beta = self.beta*self.seasonalforcing(t)*(1-Xi_list*self.mmax*self.m_normalized(H_list))*X_list
        
        # Compartment Evolution:
        dS = -(FOI_alpha[0] + FOI_alpha[1])*S + self.nu[0]*a + self.nu[1]*b
        dA = FOI_alpha[0]*S - (FOI_beta[1]+self.gamma[0])*A + self.nu[1]*Ab
        dB = FOI_alpha[1]*S - (FOI_beta[0]+self.gamma[1])*B + self.nu[0]*aB
        dAB = FOI_beta[1]*A + FOI_beta[0]*B - (self.gamma[0]+self.gamma[1])*AB
        dAb = FOI_beta[0]*b + self.gamma[1]*AB -(self.nu[1]+self.gamma[0])*Ab
        daB = FOI_beta[1]*a + self.gamma[0]*AB -(self.nu[0]+self.gamma[1])*aB
        da = self.gamma[0]*A + self.nu[1]*ab - (FOI_beta[1]+self.nu[0])*a
        db = self.gamma[1]*B + self.nu[0]*ab - (FOI_beta[0]+self.nu[1])*b
        dab = self.gamma[1]*aB + self.gamma[0]*Ab - (self.nu[0]+self.nu[1])*ab
        
        # Mitigation:
        if(self.distinguishable == True): 
            dH_A = (H_A1-H_A)/self.tau
            dH_A1 = ((2-self.phi)*X_A-H_A1)/self.tau
            dH_B = (H_B1-H_B)/self.tau
            dH_B1 = (self.phi*X_B-H_B1)/self.tau
            return [dS,dA,dB,dAB,dAb,daB,da,db,dab,dH_A,dH_A1,dH_B,dH_B1]
        elif(self.distinguishable == False): 
            Xtot = 2*(1-self.phi)*(A+Ab)+2*self.phi*(B+aB)+AB #(2-self.phi)*(A+Ab)+self.phi*(B+aB)+np.max(self.phi, 2-self.phi)*AB # both diseases mitigated by Xtot = X_A + X_B - AB, where disease B is considered more dangerous (than A) if phi>1
            dH = (H1-H)/self.tau
            dH1 = (Xtot-H1)/self.tau
            return [dS,dA,dB,dAB,dAb,daB,da,db,dab,dH,dH1]
        else: raise RuntimeError("distinguishable must be True or False")

    def run(self):
        # event1 = lambda t,x: self.i_peaks(t,x)
        # if self.direction:   
        #     event1.direction = -1
            
        # event2 = lambda t,x: self.seasonal_peaks(t,x)
        # event2.direction = -1
        
        # event3 = lambda t,x: self.r_eff(t,x)
        # event3.direction = 1
        
        # event4 = lambda t,x: self.r_eff(t,x)
        # event4.direction = -1

        event_PopBelowZero = lambda t,y: np.min(y)+.001 # allow 0.1% error margin
        event_PopBelowZero.terminal = True
        event_PopBelowZero.direction = -1
    

        toutput = np.arange(self.tmin, self.tmax, self.stepsize)
        res = solve_ivp(self.fun, (self.tmin,self.tmax), self.y0, t_eval=toutput, max_step = self.maxstep, events=[event_PopBelowZero], rtol = self.tolerances[0], atol = self.tolerances[1], method='LSODA')
        # AB(0) intit + mmax=0.25: need rtol=1e-10 amd atol=1e-13 instead of 1e-6 and 1e-9 for no negative 1-S at high rates # Need LSODA instead of default or DOP853 for no numerical errors making trajectories look chaotic despite already have reached lim cycle (this effect is worse the higher the rates / low ratefractions)
        # A and B init + mmmax=0.2: need rtol=1e-15 and atol=1e-18 to avoid negative populations and hazard at high rates (kappa as low as 10^-1.4). For 10^-1.5 and lower no way of getting rid of negative pop.
        self.times = res['t']
        self.data = res['y']

        # Check if the solver stopped due to negative population
        if len(res.t_events[0]) != 0: raise RuntimeError("Solver stopped due to negative population.")
        # Check total pop = const
        if np.sum(self.data[:9],axis=0)[-1] > 1.01: raise RuntimeError("Total population exceeds 1.01.")
        if np.sum(self.data[:9],axis=0)[-1] < 0.99: raise RuntimeError("Total population goes below 0.99.")
        # Check if the solver stopped early
        if self.times[-1] < self.tmax-1: raise RuntimeError("Solver stopped early before reaching the expected end time. solver_tmax=", self.times[-1], "expected_tmax=", self.tmax)

        # self.events = res['y_events']
        # self.t_events = res['t_events']
        if(self.reduced_output == True): return self.times, self.data[0], self.data[9] # return only S and H
        return self.times, self.data#, self.events, self.t_events

###################################################################################
