from  Tools import *


class model:
    def __init__(self, y0, beta, alpha, gamma, nu, tmin, tmax, stepsize, maxstep, direction, s, w, seasonalshift):
        
        self.y0 = y0
        self.beta = beta
        self.alpha = alpha
        self.gamma = gamma
        self.nu = nu
        self.tmin = tmin
        self.tmax = tmax
        self.stepsize = stepsize
        self.maxstep = maxstep
        self.direction = direction
        self.s = s
        self.w = w
        self.seasonalshift = seasonalshift
        
    # def r_eff(self,t,y):
    #     return self.beta/self.gamma*self.seasonalforcing(t)*self.softplus(y[4])*y[0]-1

    def seasonalforcing(self,t):
        return (1+self.s*np.cos(self.w*(t-self.seasonalshift)))
        
    def fun(self,t,y):
        S,X,P,Q,A,AB = y
        
        dS = -2*self.alpha*self.seasonalforcing(t)*X*S + 2*self.nu*(P-A)
        dX = self.alpha*self.seasonalforcing(t)*X*S + self.beta*self.seasonalforcing(t)*X*P-self.gamma*X
        dP = self.alpha*self.seasonalforcing(t)*X*S - self.beta*self.seasonalforcing(t)*X*P + self.nu*(X-P+Q-2*AB)
        dQ = 2*self.beta*self.seasonalforcing(t)*X*A + 2*self.gamma*(-2*AB+X-A) - 2*self.nu*(Q-AB)
        dA = self.alpha*self.seasonalforcing(t)*X*S - self.beta*self.seasonalforcing(t)*X*A - self.gamma*A + self.nu*(X-AB-A)
        dAB = 2*self.beta*self.seasonalforcing(t)*X*A-2*self.gamma*AB
        return [dS,dX,dP,dQ,dA,dAB]

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
    

        toutput = np.arange(self.tmin, self.tmax, self.stepsize)
        res = solve_ivp(self.fun, (self.tmin,self.tmax), self.y0, t_eval=toutput, max_step = self.maxstep, rtol = 1e-6, atol = 1e-9)#,events=[event1,event2,event3,event4])
        self.times = res['t']
        self.data = res['y']

        # self.events = res['y_events']
        # self.t_events = res['t_events']
        return self.times, self.data #, self.events, self.t_events

def VarToCompartments(S,X,P,Q,A,AB):
    ab = Q-AB
    Ab = X-A-AB
    a = P-A
    return S,A,AB,Ab,a,ab

###################################################################################

# SIRS Simulation Funcs

def Simul_Rinf_for_C_alpha(alpha, C, epsi_val, p, kappa_val=1, InInfComp='A and B'):
    R_inf = np.zeros((len(alpha),len(C)))
    SetInCondredSIRS(epsi_val, p, InInfComp)

    for j in range(len(C)):
        for i in range(len(alpha)):
            p['beta'] = C[j]*alpha[i]/kappa_val
            p['alpha'] = alpha[i]/kappa_val
            p['gamma'] = 0.1/kappa_val
            p['nu'] = 0.01/kappa_val
            m = model(**p)
            times, Data = m.run()
            S = Data[0]
            R_inf[i,j] = 1-S[-1]
        if(len(C)>10): 
            if(j % int(len(C)/10)==0): print(int(j*10/len(C)+1),'/10')
    return R_inf

def Simul_Rinf_for_epsi_alpha(alpha, C_val, epsi, p, kappa_val=1, InInfComp='A and B'):
    R_inf = np.zeros((len(alpha),len(epsi)))
    
    for k in range(len(epsi)):
        SetInCondredSIRS(epsi[k], p, InInfComp)
        for i in range(len(alpha)):
            p['beta'] = C_val*alpha[i]/kappa_val
            p['alpha'] = alpha[i]/kappa_val
            p['gamma'] = 0.1/kappa_val
            p['nu'] = 0.01/kappa_val
            m = model(**p)
            times, Data = m.run()
            S = Data[0]
            R_inf[i,k] = 1-S[-1]
        if(len(epsi)>10): 
            if(k % int(len(epsi)/10)==0): print(int(k*10/len(epsi)+1),'/10')
    return R_inf

def Simul_Rinf_for_kappa_alpha(alpha, C_val, epsi_val, p, kappa, InInfComp='A and B'):
    R_inf = np.zeros((len(alpha),len(kappa)))
    SetInCondredSIRS(epsi_val, p, InInfComp)

    for k in range(len(kappa)):
        for i in range(len(alpha)):
            p['beta'] = C_val*alpha[i]/kappa[k]
            p['alpha'] = alpha[i]/kappa[k]
            p['gamma'] = 0.1/kappa[k]
            p['nu'] = 0.01/kappa[k]
            m = model(**p)
            times, Data = m.run()
            S = Data[0]
            R_inf[i,k] = 1-S[-1]
        if(len(kappa)>10): 
            if(k % int(len(kappa)/10)==0): print(int(k*10/len(kappa)+1),'/10')
    return R_inf



def Simul_alphacrit_for_epsi_CorKappa(epsi, C, p, thresh, kappa=1, tmax=None, N=1000, InInfComp='A and B'):
    if(tmax!=None): p['tmax'] = tmax
    alpha = np.linspace(0,.12,N)
    dalpha = (max(alpha)-min(alpha))/N

    if((type(C)==list or type(C)==np.ndarray) and (type(kappa)==list or type(kappa)==np.ndarray)):
        raise ValueError('Only one of C or kappa can be list or array')
    elif(type(C)==list or type(C)==np.ndarray):
        alpha_crit = np.empty((len(epsi),len(C)))
        idx_lastPT2 = 0
        for k in range(len(epsi)):
            idx_lastPT = 0
            for j in range(len(C)):
                for i in range(N-idx_lastPT-idx_lastPT2):
                    SetInCondredSIRS(epsi[k], p, InInfComp)
                    p['beta'] = C[j]*alpha[-1-i-idx_lastPT-idx_lastPT2]/kappa
                    p['alpha'] = alpha[-1-i-idx_lastPT-idx_lastPT2]/kappa
                    p['gamma'] = 0.1/kappa
                    p['nu'] = 0.01/kappa
                    m = model(**p)
                    times, Data = m.run()
                    S = Data[0]
                    R_inf = 1-S[-1]
                    if(i==N-idx_lastPT-idx_lastPT2-1): alpha_crit[k,j] = 0 # No Phasetransition
                    if(i>0):
                        if(np.abs(R_inf-R_inf_prev)/dalpha > thresh):
                            alpha_crit[k,j] = alpha[-1-i-idx_lastPT-idx_lastPT2]
                            if(j==0): idx_lastPT2 += i
                            else: idx_lastPT += i
                            break
                    R_inf_prev = R_inf
            print(k+1,'/',len(epsi))
    elif(type(kappa)==list or type(kappa)==np.ndarray):
        alpha_crit = np.empty((len(epsi),len(kappa)))
        idx_lastPT2 = 0
        for k in range(len(epsi)):
            idx_lastPT = 0
            for j in range(len(kappa)):
                for i in range(N-idx_lastPT-idx_lastPT2):
                    SetInCondredSIRS(epsi[k], p, InInfComp)
                    p['beta'] = C*alpha[-1-i-idx_lastPT-idx_lastPT2]/kappa[j]
                    p['alpha'] = alpha[-1-i-idx_lastPT-idx_lastPT2]/kappa[j]
                    p['gamma'] = 0.1/kappa[j]
                    p['nu'] = 0.01/kappa[j]
                    m = model(**p)
                    times, Data = m.run()
                    S = Data[0]
                    R_inf = 1-S[-1]
                    if(i==N-idx_lastPT-idx_lastPT2-1): alpha_crit[k,j] = 0 # No Phasetransition
                    if(i>0):
                        if(np.abs(R_inf-R_inf_prev)/dalpha > thresh):
                            alpha_crit[k,j] = alpha[-1-i-idx_lastPT-idx_lastPT2]
                            if(j==0): idx_lastPT2 += i
                            else: idx_lastPT += i
                            break
                    R_inf_prev = R_inf
            print(k+1,'/',len(epsi))
    else: raise ValueError('Either C or kappa has to be list or array')
    
    return alpha_crit

def Simul_alphacritWithDerivValue_for_epsi_CorKappa(epsi, C, p, kappa=1, tmax=None, N=1000, InInfComp='A and B'):
    if(tmax!=None): p['tmax'] = tmax
    alpha = np.linspace(0,.11,N)
    dalpha = (max(alpha)-min(alpha))/N

    if((type(C)==list or type(C)==np.ndarray) and (type(kappa)==list or type(kappa)==np.ndarray)):
        raise ValueError('Only one of C or kappa can be list or array')
    elif(type(C)==list or type(C)==np.ndarray):
        alpha_crit_andDerivValues = np.empty((len(epsi),len(C),2)) # last 2 for [possible alpha_crit, corresponding derivative jump]
        for k in range(len(epsi)):
            for j in range(len(C)):
                Rinf = np.zeros(len(alpha))
                for i in range(len(alpha)):
                    SetInCondredSIRS(epsi[k], p, InInfComp)
                    p['beta'] = C[j]*alpha[i]/kappa
                    p['alpha'] = alpha[i]/kappa
                    p['gamma'] = 0.1/kappa
                    p['nu'] = 0.01/kappa
                    m = model(**p)
                    times, Data = m.run()
                    S = Data[0]
                    Rinf[i] = 1-S[-1]
                delRinf = np.gradient(Rinf,dalpha)
                delRinf_max_idx = np.argmax(delRinf[1:-2]) + 1 # avoid edges
                alpha_crit_andDerivValues[k,j,:] = [alpha[delRinf_max_idx], delRinf[delRinf_max_idx]]
            print(k+1,'/',len(epsi))
    elif(type(kappa)==list or type(kappa)==np.ndarray):
        alpha_crit_andDerivValues = np.empty((len(epsi),len(kappa),2)) # last 2 for [possible alpha_crit, corresponding derivative jump]
        for k in range(len(epsi)):
            for j in range(len(kappa)):
                Rinf = np.zeros(len(alpha))
                for i in range(len(alpha)):
                    SetInCondredSIRS(epsi[k], p, InInfComp)
                    p['beta'] = C*alpha[i]/kappa[j]
                    p['alpha'] = alpha[i]/kappa[j]
                    p['gamma'] = 0.1/kappa[j]
                    p['nu'] = 0.01/kappa[j]
                    m = model(**p)
                    times, Data = m.run()
                    S = Data[0]
                    Rinf[i] = 1-S[-1]
                delRinf = np.gradient(Rinf,dalpha)
                delRinf_max_idx = np.argmax(delRinf[1:-2]) + 1 # avoid edges
                alpha_crit_andDerivValues[k,j,:] = [alpha[delRinf_max_idx], delRinf[delRinf_max_idx]]
            print(k+1,'/',len(epsi))
    else: raise ValueError('Either C or kappa has to be list or array')
    
    return alpha_crit_andDerivValues


def Set_alphacrit_forthreshhold(alpha_crit_andDerivValues, thresh, C_Val_Set_Reduction=False, C_Val_Set_original=False): # original set only needed if we want reduction
    
                 ################# NOT USED AFTER ALL ################
    if np.all(C_Val_Set_Reduction) != False:
        if np.all(C_Val_Set_original) == False:
            print('ERROR: Need original set as input to reduce to new C value set!')
            return 1
        else:
            temp = np.zeros((alpha_crit_andDerivValues.shape[0],len(C_Val_Set_Reduction),2))
            for j_red in range(len(C_Val_Set_Reduction)):
                j_orig = np.argmin(np.abs(C_Val_Set_Reduction[j_red]-C_Val_Set_original))
                temp[:,j_red,:] = alpha_crit_andDerivValues[:,j_orig,:]
            alpha_crit_andDerivValues = temp
                 #####################################################

    alpha_crit = np.zeros((alpha_crit_andDerivValues.shape[0],alpha_crit_andDerivValues.shape[1]))
    for i in range(alpha_crit_andDerivValues.shape[0]):
        for j in range(alpha_crit_andDerivValues.shape[1]):
            if(alpha_crit_andDerivValues[i,j,1]>thresh):
                alpha_crit[i,j] = alpha_crit_andDerivValues[i,j,0]
            else:
                alpha_crit[i,j] = 0
    return alpha_crit

#######################################################################
