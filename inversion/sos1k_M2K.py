# M2K = model to kinematics plane (L_Z - V_R)
from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
import scipy.special as sp
from scipy.integrate import odeint
from scipy.optimize import brentq
import time
import kcf_L4_mod                        #  module with dphi, phi code
import config                            # config function to pass global variables

""" sos1.py is second pass at plotting a set of invariant curves etc for a
    fixed value of E_J.  Plot the ZVC, bar, L3,L4 in function ZVC. Then
    do the invariant curves one by one in function invcur  """

# this version uses a softened disk potential 0.5 V^2 log(r^2 + AA^2): take AA = 0.5

# The periods of the short-period L4-symmetric orbits are almost independent of amplitude
# for y0 = 0.3 to 1.5
# set parameters for potential:   V is softened flat rotation curve, Mbar is bar mass,
# eps is a.sqrt(1-e^2)   Omega_bar is bar pattern speed

# ##########################################################################
# set parameters for Ferrers ellipsoid  e = c/a,  a^2 - c^2 = 1 from config
# ##########################################################################

e = config.e
a = 1/np.sqrt(1 - e**2)
c = e*a
AA = config.AA      # disk potential is 0.5 V^2 log(r^2 + AA^2)
Mbar = config.Mbar  # units M_sun
V = config.V        # units km/s
eps = config.eps    # units kpc
P = config.P        # P = 1,2 => CR at r = eps, 2eps
Q = P*0.98*(Mbar/2e+10)*(5./eps)*((240/V)**2)


# with these parameters P = 2.
#  y0_L4, E_JL4 = 1.47371, -0.597203
#  y0_L3, E_JL3 = 1.50719, -0.629444

# with P = 1.44
# y0_L4, E_JL4 = 1.24692, -0.68622
# y0_L3, E_JL3 = 1.29546, -0.72506


# --------------------------------------------------------------------------

def ZVC(ax,E_J,axmax):   # E_J is the E_J for the inv curves          takes about 2.5 sec
                      # axmax is the max |x| etc for the plot
                    #   ax is subplot to be plotted in

# plot ZVCs for E_J = -Phi0
# plot the bar, L3,L4 points, ZVC where E_J = - Phi
# to plot ZVC through L3, take Phi0 = .629444

# draw the Ferrers bar
    theta = np.linspace(0, 2*np.pi, 100)
    xf = a*np.sin(theta)
    yf = c*np.cos(theta)
    
    ax.plot(xf,yf, color='0.85')
#   plt.fill(xf,yf,'gray')
    ax.fill(xf,yf,color='0.85')
    ax.axis('equal')
    # ax.show()

# plot L3, L4 for P = 1.44

    ax.plot(0.,1.24692,'bo', markersize = 3)
    ax.plot(1.29546,0.,'bo', markersize = 3)
    ax.plot(0.,-1.24692,'bo', markersize = 3)
    ax.plot(-1.29546,0.,'bo', markersize = 3)
    ax.axis([-axmax,axmax,-axmax,axmax])
    
# ZVC - set level for contour

    x = np.linspace(-axmax,axmax,201) + 0.00001  
    y = np.linspace(-axmax,axmax,201)

    ll = len(x)
    Phi = np.zeros((ll,ll))
    for i in np.arange(ll):
        for j in np.arange(ll):
            Phi[i,j] = kcf_L4_mod.phi(y[i],x[j])     # Phi = 0.597241 to 4.381738

    ax.contour(x,y,Phi.T,[-E_J], colors=['blue'],linestyles='dashed')
    # plt.show()
     
    return

# -------------------------------------------

# run surfaces of section at E_J = -Phi:  takes about 0.2 sec for nsamples = 100,000

def invcur(ax,x,y,vx,vy,nsamples,E_J,y0,vx0):    
#ax is the subplot to be ploted on
    
# find (x,y) positions where vy = 0 and vx > 0 - do linear interpolation 

    q = np.where(vy[0:nsamples-2]*vy[1:nsamples-1] < 0.)[0]

    xs = np.zeros(len(q))
    ys = np.zeros(len(q))
    vxs = np.zeros(len(q))
    for k in np.arange(len(q)-1):
        dq = np.abs(vy[q[k]]) / (np.abs(vy[q[k]]) + np.abs(vy[q[k]+1]))
        xs[k] = x[q[k]] + dq*(x[q[k]+1] - x[q[k]])
        ys[k] = y[q[k]] + dq*(y[q[k]+1] - y[q[k]])
        vxs[k] = vx[q[k]] + dq*(vx[q[k]+1] - vx[q[k]])

    # plot xs,ys if vxs >= 0
    c = (vxs >= 0.)           ### change to <= 0 as needed ####
    ax.plot(xs[c],ys[c],'mo', markersize = 1.0)     # was 0.5
    ax.set_xlabel('xs')
    ax.set_ylabel('ys')
    ax.set_title('EJ = {:.6f},  y0 = {:.3f}, vx0 = {:.3f}'.format(E_J,y0,vx0))
    # plt.show()
    ic = (xs[c],ys[c])
  
    return ic

# --------------------------------------

def plot_period(axs,y0_periodic):
    ax=axs[1]
    # ax.plot(0.,y0_periodic,'ro', markersize = 2)
    ax.annotate(r'$y_\text{perodic}$ ='+' {:.1f}'.format(y0_periodic), xy=(0.2, 0.1), 
                xycoords='axes fraction', ha='center', va='center',
                bbox=dict(boxstyle='square', fc='w'),fontsize=12)
    
def getc(theta):
    r=1.6 #8km to the center
    # theta=30 #in degrees
    theta=theta/180*3.14
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    centre=[x,y]
    return centre

##################
#  main program  #
##################
def sos1k(axs,y0,E_J,tmax=500, centre=[1.4,0.8],theta=np.nan,colour='g',labelname='orbit'): #centre defaulted to solar
    # E_J =  -0.63                      # -0.427760
    # E_J=float(input('input y0 for sos: '))
    # y0 = float(input('input y0 for sos: '))
    if ~np.isnan(theta):    #calculate centre based on theta
        make_label=True
        centre=getc(theta)
    else:  
        make_label=False

    vx0 = np.sqrt( 2.* (E_J + kcf_L4_mod.phi(0,y0)))     # vx0 > 0, < 0 give different orbits
    axmax = 2.2                                     # changing sign of c in line 107 gives other half of IVC plane                                                 # max |x|, |y| in plot
    # tmax = 500                  # two Hubble times
    # let user define tmax
    nsamples = 200*tmax   # time is ~ independent of nsamples   
    tmin = 0
    
    #plt.close('all')                  # this shows only the latest sos
    # plt.figure()
    # fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    # plot the bar, L3, L4
    ax=axs[0]
    ot = ZVC(ax,E_J,axmax)

    # do orbit integration for several periods

    start = time.time()
    ts = np.linspace(tmin,tmax,nsamples)
    initial_cond = np.array([0., y0, vx0, 0.])
    orbit = odeint(kcf_L4_mod.dphi, initial_cond, ts)
    x = orbit[:,0]
    y = orbit[:,1]
    vx = orbit[:,2]
    vy = orbit[:,3]  
    end = time.time()
    # print('tmax, end-start = ', tmax, end-start)

    F = np.zeros(2)
    Jac = np.zeros(2)
    F[0] = kcf_L4_mod.phi(x[1],y[1])
    F[1] = kcf_L4_mod.phi(x[nsamples-1],y[nsamples-1])
    Jac[0] = 0.5*(vx[1]**2) + 0.5*(vy[1]**2) - F[0]                      # Jacobi integral
    Jac[1] = 0.5*(vx[nsamples-1]**2) + 0.5*(vy[nsamples-1]**2) - F[1]
    # print('Jac[start], Jac[end] = ',Jac[0],Jac[1])


    # plot the invariant curves
    # ax=axs[1]
    # ot = ZVC(ax,E_J,axmax)

    # ic = invcur(ax,x,y,vx,vy,nsamples,E_J,y0,vx0)              # returns (xs,ys)
    
    # plot the orbit

    # plt.subplot(1,2,2)
    ax=axs[0]
    ax.plot(x,y,'g')                # was green
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.plot(0.,1.24692,'bo', markersize = 3)   # P = 1.44
    ax.plot(1.29546,0.,'bo', markersize = 3)
    ax.plot(0.,-1.24692,'bo', markersize = 3)
    ax.plot(-1.29546,0.,'bo', markersize = 3)
    # plt.plot(0.,1.47371,'bo', markersize = 3)   # P = 2.
    # plt.plot(1.50719,0.,'bo', markersize = 3)
    # plt.plot(0.,-1.47371,'bo', markersize = 3)
    # plt.plot(-1.50719,0.,'bo', markersize = 3)
    ax.axis([-axmax,axmax,-axmax,axmax])
    ax.set_title(r'$y_0$ ='+' {:.4f}'.format(y0))
    # plt.show()


    # plotting Lz-Vr of the model
    # want distribution of Lz at r = 1.6 and vr > 0


    Lz = x**2 + y**2 - x*vy + y*vx #(vphi*R))
    r = np.sqrt(x**2+y**2)
    rdot = (x*vx + y*vy)/r
    vr = (x*vx + y*vy)/r

    # as the star goes outwards through r = 1.6 (solar neighborhood), what changes are
    # expected in Lz

    # k = 0
    # vrz = np.zeros(500)
    # tz = np.zeros(500)
    # rz = np.zeros(500)
    # Lza = np.zeros(500)
    # for i in np.arange(len(x)-1):
    #     if ((r[i]-1.6)*(r[i+1]-1.6) < 0) & (r[i] < r[i+1]):
    #         f = (1.6 - r[i])/(r[i+1] - r[i])
    #         tz[k] = ts[i] + f*(ts[i+1]-ts[i])
    #         rz[k] = r[i] + f*(r[i+1]-r[i])
    #         vrz[k] = vr[i] + f*(vr[i+1] - vr[i])
    #         Lza[k] = Lz[i] + f*(Lz[i+1]-Lz[i])
    #         k = k+1
    #         # print(i,k)
    #         if (k == 500) :break
    # print(vrz,Lza)


    # r=1.6 y=0.8 x=1.4 (Solar position)
    x_c=centre[0]
    y_c=centre[1]
    # convert Lz Vr to physical units
    vr_ph=vr*5*40
    Lz_ph=Lz*5**2*40

    cdt_SNd = ((x-x_c)**2+(y-y_c)**2 < 1/25)
    Lz_SNd = Lz_ph[cdt_SNd]
    vr_SNd = vr_ph[cdt_SNd]
    ax=axs[0]
    ax.plot(x_c,y_c,'r*',zorder=12,markersize=10) #plot the centre
    # ax=axs[1]
    ax.plot(x[cdt_SNd],y[cdt_SNd],'lime',marker ='.',zorder=10)
    ax=axs[1]
    ax.annotate('$E_J$ = {:.4f}'.format(E_J), xy=(0.2, 0.9), 
                xycoords='axes fraction', ha='center', va='center',
                bbox=dict(boxstyle='square', fc='w'),fontsize=12)
    ax.set_xlim([-150,150])
    ax.set_ylim([1000,2500])
    ax.plot(vr_SNd,Lz_SNd,'.',markersize=.5,zorder=10,label=str(labelname),color=colour)
    # else:
    #     ax.plot(vr_SNd,Lz_SNd,'.',markersize=.5,zorder=10,color=colour)
    return Lz_SNd,vr_SNd

