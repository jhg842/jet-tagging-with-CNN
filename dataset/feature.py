

from lib2to3.pgen2.pgen import DFAState
from random import weibullvariate
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot
import matplotlib.colors as mcl
import awkward as ak
from matplotlib.colors import LogNorm
import feature as fe
from multiprocessing import Pool
from multiprocessing import Process
from matplotlib.colors import LogNorm
########################################################################################################################################################
def jet04(jeteta,jetphi,jetpt,pareta,parphi,parpt,vx,vy,vz):
    jet_eta = []; jet_phi = []; jet_pt = []; verx = []; very = [];verz = []
     
    for i in range(1):
            if -0.5<=jeteta[i]<=0.5:
                for j in range(len(pareta)):
                    del_eta = (jeteta[i] - pareta[j])
                    del_phi = (jetphi[i] - parphi[j])
                    if del_phi > 3.14:
                        del_phi  = del_phi -3.14

                    elif del_phi <-3.14:
                        del_phi = del_phi + 3.14
                    else:
                        del_phi = del_phi
                    r_range = np.sqrt((del_eta)**2 + (del_phi)**2)
                    if r_range <= 0.4:
                        jet_eta.append(pareta[j])
                        jet_phi.append(parphi[j])       
                        jet_pt.append(parpt[j])
                        verx.append(vx[j])
                        very.append(vy[j])
                        verz.append(vz[j])
    
    return jet_eta, jet_phi, jet_pt, verx, very, verz
########################################################################################################################################################


def jet07(jeteta,jetphi,jetpt,pareta,parphi,parpt,vx,vy,vz):
    jet_eta = []; jet_phi = []; jet_pt = []; verx = []; very = [];verz = []
     
    for i in range(1):
        if 30<jetpt[i]<40:
            if -0.5<=jeteta[i]<=0.5:
                for j in range(len(pareta)):
                
                    r_range = np.sqrt((jeteta[i] - pareta[j])**2 + (jetphi[i] - parphi[j])**2)
                    if r_range <= 0.7:
                        jet_eta.append(pareta[j])
                        jet_phi.append(parphi[j])       
                        jet_pt.append(parpt[j])
                        verx.append(vx[j])
                        very.append(vy[j])
                        verz.append(vz[j])
    
    return jet_eta, jet_phi, jet_pt, verx, very, verz

    
########################################################################################################################################################
def shift(jeteta,jetphi,jetpt,pareta,parphi,parpt):
    new_eta = []; new_phi = []; new_pt = []; verx = []; very = [];verz = []
    
    for jet in range(1):
        if 30<jetpt[jet]<40:
            if -0.5<=jeteta[jet]<=0.5:
                for par in range(len(pareta)):
                
                    neta = jeteta[jet] - pareta[par]
                    nphi = jetphi[jet] - parphi[par]
                    new_eta.append(neta)
                    new_phi.append(nphi)
                    new_pt.append(parpt[par])
            


    return new_eta, new_phi, new_pt
            
########################################################################################################################################################

def vertex(vx,vy,vz):
    R= [23,31,39,194,247,353,405]
    IB1 = []; IB2 = []; IB3 = []; MB1 = []; MB2 = []; OB1 = []; OB2 = []
    for i in range(len(vx)):
        if np.sqrt(vx[i]**2 + vy[i]**2) <23:
            IB1.append(R[0])
            IB2.append(R[1])
            IB3.append(R[2])
            MB1.append(R[3])
            MB2.append(R[4])
            OB1.append(R[5])
            OB2.append(R[6])
        elif np.sqrt(vx[i]**2 + vy[i]**2) <31:
            IB2.append(R[1])
            IB3.append(R[2])
            MB1.append(R[3])
            MB2.append(R[4])
            OB1.append(R[5])
            OB2.append(R[6])
        elif np.sqrt(vx[i]**2 + vy[i]**2) <39:
            IB3.append(R[2])
            MB1.append(R[3])
            MB2.append(R[4])
            OB1.append(R[5])
            OB2.append(R[6])
        elif np.sqrt(vx[i]**2 + vy[i]**2) <194:
            MB1.append(R[3])
            MB2.append(R[4])
            OB1.append(R[5])
            OB2.append(R[6])
        elif np.sqrt(vx[i]**2 + vy[i]**2)<247:
            MB2.append(R[4])
            OB1.append(R[5])
            OB2.append(R[6])
        elif np.sqrt(vx[i]**2 + vy[i]**2)<353:
            OB1.append(R[5])
            OB2.append(R[6])
        elif np.sqrt(vx[i]**2 + vy[i]**2) < 405:
            OB2.append(R[6])

    return IB1,IB2,IB3,MB1,MB2,OB1,OB2 
            
########################################################################################################################################################

# def eta(a,vz):
#     R= [23,31,39,194,247,353,405]
#     IB1 = []; IB2 = []; IB3 = []; MB1 = []; MB2 = []; OB1 = []; OB2 = []

#     for r in R:
#         for par in range(len(a)):
#             x = 2*np.arctan(np.exp(-a[par]))
#             if r == 23:
#                 fir =     r/np.tan(x)
#                 IB1.append(round(fir,3)+vz[par])

#             elif r == 31:
#                 sec =     r/np.tan(x)
#                 IB2.append(round(sec,3)+vz[par])

#             elif r == 39:
#                 thr =     r/np.tan(x)
#                 IB3.append(round(thr,3)+vz[par])
                
#             elif r == 194:
#                 fort =    r/np.tan(x)
#                 MB1.append(round(fort,3)+vz[par])

#             elif r == 247:
#                 fift =    r/np.tan(x)
#                 MB2.append(round(fift,3)+vz[par]) 

#             elif r == 353:
#                 six =     r/np.tan(x)
#                 OB1.append(round(six,3)+vz[par])

#             elif r == 405:
#                 sev =     r/np.tan(x)
#                 OB2.append(round(sev,3)+vz[par])

#     return IB1,IB2,IB3,MB1,MB2,OB1,OB2

########################################################################################################################################################
def rpos(vtx,vty,pphi):
    X = []; Y = []; X1 = []; Y1 = []
    xl = [ 0, 0, 0, 0, 0, 0,0]
    yl = [ 0, 0, 0, 0, 0, 0,0]
    # yr = [ 0, 0, 0, 0, 0, 0,0]
    R= [23,31,39,194,247,353,405]
    for par in range(len(vtx)):
        if vtx[par] == 0 and vty[par] ==0:
            for i in range(7):
                xl[i] = R[i]*np.cos(pphi[par])
                yl[i] = R[i]*np.sin(pphi[par])

                X.append(xl[i])
                Y.append(yl[i])
                
        elif vtx[par] != 0 or vty[par] != 0:
            for i in range(7):
                m = np.tan(pphi[par])
                a = vtx[par]
                b = vty[par]
                # layer_r = np.sqrt(a**2 + b**2)
                x = 1+m**2
                y = -2*a*(m**2) + 2*m*b
                z = (m**2)*(a**2) + -2*a*b*m+b**2 + -R[i]**2
                x_value = np.roots([x,y,z])
                y_value = m*(x_value-a) +b
                # for linear
                # if abs(x_value[0]) > abs(a) or abs(x_value[1]) > abs(a):
                #     if x_value[0] > 0 and x_value[1]<0:
                #         X1.append(x_value[0])
                #         Y1.append(y_value[0])
                #     elif x_value[1] > 0 and x_value[0]<0:
                #         X1.append(x_value[1])
                #         Y1.append(y_value[1])
                
                # for vtx
                if abs(x_value[0]) > abs(a) or abs(x_value[1]) > abs(a):
                    fir_x = x_value[0] - a
                    sec_x = x_value[1] - a
                    if abs(fir_x) < abs(sec_x):
                        X1.append(x_value[0])
                        Y1.append(y_value[0])

                    elif abs(fir_x) > abs(sec_x):
                        X1.append(x_value[1])
                        Y1.append(y_value[1])

    return X,Y,X1,Y1
                # X = [list(arr) for arr in X1]
                # Y = [list(arr) for arr in Y1]

            # x = np.linspace(-410,410,10)
            # y = np.tan(pphi[par])*(x-vtx[par]) + vty[par]
            # yr = np.sqrt(405**2 - x**2)
            # ryr = -1*yr
            # for i in range(len(x)):
            #     if round(y[i],1) == round(yr[i],1) or round(y[i],1) == round(ryr[i],1):
                 
            #     X.append(vtx[par])
            #     Y.append(vty[par])
            #     X.append(x[i])
            #     Y.append(y[i])
               



    # return X,Y,X1,Y1

        

########################################################################################################################################################
def eta(a):
    R= [23,31,39,194,247,353,405]
    IB1 = []; IB2 = []; IB3 = []; MB1 = []; MB2 = []; OB1 = []; OB2 = []

    for r in R:
        for par in range(len(a)):
            x = 2*np.arctan(np.exp(-a[par]))
            if r == 23:
                fir =   r/np.tan(x)
                IB1.append(round(fir,3))

            elif r == 31:
                sec =   r/np.tan(x)
                IB2.append(round(sec,3))

            elif r == 39:
                thr =   r/np.tan(x)
                IB3.append(round(thr,3))
                
            elif r == 194:
                fort =  r/np.tan(x)
                MB1.append(round(fort,3))

            elif r == 247:
                fift =  r/np.tan(x)
                MB2.append(round(fift,3)) 

            elif r == 353:
                six =   r/np.tan(x)
                OB1.append(round(six,3))

            elif r == 405:
                sev =   r/np.tan(x)
                OB2.append(round(sev,3))

    return IB1,IB2,IB3,MB1,MB2,OB1,OB2
######################################################################################################################################################## z축 계산 파트

def event(jdf,parpt,pareta,parphi,events):
    for eve in range(events):
        
        df = jdf.iloc[eve:eve+1]

        chg_eta = ak.flatten(df['p_eta']).to_numpy()
        chg_phi = ak.flatten(df['p_phi']).to_numpy()
        chg_pt = ak.flatten(df['p_pt']).to_numpy()
        par_chg = ak.flatten(df['p_chg']).to_numpy()
        

        for i in range(len(par_chg)):
            if par_chg[i] == True:
                parpt.append(chg_pt[i])
                pareta.append(chg_eta[i])
                parphi.append(chg_phi[i])
######################################################################################################################################################## root 파일 읽는 파트
# def image(a,b,c,i,event,cmap):
#     for num in range(event):
#         plt.hist2d(a,b,weights=c,cmap = cmap)
#         plt.xlabel('z[cm]')
#         plt.ylabel('phi')
#         plt.colorbar()
#         plt.savefig(('./{}/event{}.png'.format(i,num)),dpi=200)
        
#         plt.close('all')  
        
def IB1image(a,b,c,i,event,cmap):
    
    plt.hist2d(a,b,weights=c,norm=LogNorm(vmin=0.1, vmax=10**2))#9,12
    plt.title('IB1')
    plt.xlabel('z[mm]')
    plt.ylabel('phi')
    plt.xlim(-135,135)#135
 
    plt.ylim(-0.5,0.5)
    plt.colorbar()
    plt.savefig(('./layer/layerc/{}/event{}.png'.format(i,event)),dpi=200)
    
    plt.close('all')    

def IB2image(a,b,c,i,event,cmap):
    
    plt.hist2d(a,b,weights=c,norm=LogNorm(vmin=0.1, vmax=10**2))#9,16
    plt.title('IB2')
    plt.xlabel('z[mm]')
    plt.ylabel('phi')
    plt.xlim(-135,135)#135
    plt.ylim(-0.5,0.5)
    plt.colorbar()
    plt.savefig(('./layer/layerc/{}/event{}.png'.format(i,event)),dpi=200)
    
    plt.close('all')

def IB3image(a,b,c,i,event,cmap):
    
    plt.hist2d(a,b,weights=c,norm=LogNorm(vmin=0.1, vmax=10**2))#15,30
    plt.title('IB3')
    plt.xlabel('z[mm]')
    plt.ylabel('phi')
    plt.xlim(-135,135)#135
    plt.ylim(-0.5,0.5)
    plt.colorbar()
    plt.savefig(('./layer/layerc/{}/event{}.png'.format(i,event)),dpi=200)
    
    plt.close('all')

def MB1image(a,b,c,i,event,cmap):
    
    
    plt.hist2d(a,b,weights=c,norm=LogNorm(vmin=0.1, vmax=10**2))#28,96
    plt.title('MB1')
    plt.xlabel('z[mm]')
    plt.ylabel('phi')
    plt.xlim(-420,420)#420
    plt.ylim(-0.5,0.5)
    plt.colorbar()
    plt.savefig(('./layer/layerc/{}/event{}.png'.format(i,event)),dpi=200)
    
    plt.close('all')

def MB2image(a,b,c,i,event,cmap):
    
    plt.hist2d(a,b,weights=c,norm=LogNorm(vmin=0.1, vmax=10**2)) # 28,120
    plt.title('MB2')
    plt.xlabel('z[mm]')
    plt.ylabel('phi')
    plt.xlim(-420,420)#420
    plt.ylim(-0.5,0.5)
    plt.colorbar()
    plt.savefig(('./layer/layerc/{}/event{}.png'.format(i,event)),dpi=200)
    
    plt.close('all')

def OB1image(a,b,c,i,event,cmap):
    
    plt.hist2d(a,b,weights=c,norm=LogNorm(vmin=0.1, vmax=10**2))# 49,168
    plt.title('OB1')
    plt.xlabel('z[mm]')
    plt.ylabel('phi')
    plt.xlim(-735,735)#735
    plt.ylim(-0.5,0.5)
    plt.colorbar()
    plt.savefig(('./layer/layerc/{}/event{}.png'.format(i,event)),dpi=200)
    
    plt.close('all')

def OB2image(a,b,c,i,event,cmap):
    
    plt.hist2d(a,b,weights=c,norm=LogNorm(vmin=0.1, vmax=10**2))#49,192
    plt.title('OB2')
    plt.xlabel('z[mm]')
    plt.ylabel('phi')
    plt.xlim(-735,735)#735
    plt.ylim(-0.5,0.5)
    plt.colorbar()
    plt.savefig(('./layer/layerc/{}/event{}.png'.format(i,event)),dpi=200)
    
    plt.close('all')  
######################################################################################################################################################## 그림 그리는 파트 위:1개씩 아래:합쳐서
def fullimage(B,b,c,event,cmap):
    fig, axs = plt.subplots(2,4, figsize=(20,15))
    zleng = [13.5,13.5,13.5,42,42,73.5,73.5,0,0]
    bin = []
    for i in range(2):
        for j in range(4):
            if i ==0:
                axs[i,j].hist2d(B[i+j],b,weights=c,bins= [], norm=LogNorm(vmin=0.1,vmax=10))
                axs[i,j].set_title('IB_{}'.format(i+j+1))
                axs[i, j].set_xlim(-zleng[i+j], zleng[i+j])
            elif i == 1:
               axs[i,j].hist2d(B[i+j+3],b,weights=c,bins=[], norm=LogNorm(vmin=0.1,vmax=10)) 
               axs[i,j].set_title('IB_{}'.format(i+j+4))
               axs[i, j].set_xlim(-zleng[i+j+3], zleng[i+j+3])
                
            axs[i,j].set_xlabel('z[cm]')
            axs[i,j].set_ylabel('phi')
            axs[i, j].set_ylim(-3, 3)
            # if i ==0:
            #     axs[i,j].set_title('IB_{}'.format(i+j+1))
            #     axs[i, j].set_xlim(-zleng[i+j], zleng[i+j])
            # elif i == 1:
            #     axs[i,j].set_title('IB_{}'.format(i+j+3))
            #     axs[i, j].set_xlim(-zleng[i+j+3], zleng[i+j+3])
            

    plt.savefig(('./event{}.png'.format(event)),dpi=400)




# a = np.random.randint(1,4, size = len(jet_df.index))