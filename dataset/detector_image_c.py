
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
from sympy import symbols, Eq, solve
import random
# import proplot as pplot
import matplotlib
matplotlib.use('Agg')

h = 24
s = 0.99
v = 1
 
colors = [
    mcl.hsv_to_rgb((h/360,0,v)),
    mcl.hsv_to_rgb((h/360,0.9,v)),
    mcl.hsv_to_rgb((h/360,1,v))
]
cmap = mcl.LinearSegmentedColormap.from_list('my_cmap',colors,gamma=2)
vmin = 0.0001
vmax = 100


tree_b = uproot.open('/home/jhg842/jet_tagging/data/ROOT/mpi_cpythiajet_2030.root')['T']

jet_df = tree_b.arrays(['p_pt','p_eta','p_phi','njetR04','jetR04_pt','jetR04_eta','jetR04_phi','p_vx','p_vy','p_vz'], library='pd')

count = 0
total_events = len(jet_df.index)
d0va = []
for eve in range(36196,total_events): #34550

    px = []; py = []
    pphi = []
    resol_px = []; resol_py = []
    df = jet_df.iloc[eve:eve+1]

    chg_eta = ak.flatten(df['p_eta']).to_numpy()
    chg_phi = ak.flatten(df['p_phi']).to_numpy()
    chg_pt = ak.flatten(df['p_pt']).to_numpy()
    jet04_pt = ak.flatten(df['jetR04_pt']).to_numpy()
    jet04_eta = ak.flatten(df['jetR04_eta']).to_numpy()
    jet04_phi = ak.flatten(df['jetR04_phi']).to_numpy()
    verx = ak.flatten(df['p_vx']).to_numpy()
    very = ak.flatten(df['p_vy']).to_numpy()
    verz = ak.flatten(df['p_vz']).to_numpy()
    # for i in range(len(d0)):
    #     if d0[i] >2:
    #         print(d0[i],eve)


# 각 값에 대해 조건 검사 후 수정

    if len(jet04_pt) == 0:
        continue

    detec = (0.15 <= chg_pt) & (chg_pt <= 0.4)
    trk_eta, trk_phi, trk_pt, vx ,vy ,vz = chg_eta[detec].tolist(), chg_phi[detec].tolist(), chg_pt[detec].tolist(), verx[detec].tolist(), very[detec].tolist(), verz[detec].tolist()
    detec2 = chg_pt>0.4
    par_eta, par_phi, par_pt, pvx ,pvy ,pvz = chg_eta[detec2].tolist(), chg_phi[detec2].tolist(), chg_pt[detec2].tolist(), verx[detec2].tolist(), very[detec2].tolist(), verz[detec2].tolist() 
       
    min_effi = int(len(trk_eta)*0.932) #run2 0.886 #run3 0.932
    max_effi = int(len(trk_eta)*0.984)
    over_effi = int(len(par_eta)*0.984)
    if len(trk_pt) >1:
        combined_list = list(zip(trk_eta, trk_phi, trk_pt, vx ,vy ,vz))
        num_to_select = random.randint(min_effi, max_effi)
        selected_items = random.sample(combined_list, num_to_select)
    if len(par_pt)>1:
        combined_list2 = list(zip(par_eta, par_phi, par_pt, pvx ,pvy ,pvz))
        over_to_select = random.randint(over_effi, over_effi)
        selected_items2 = random.sample(combined_list2, over_to_select)

    # 뽑힌 값들을 다시 풀어내기
    se_eta, se_phi, se_pt, se_vx, se_vy, se_vz = zip(*selected_items)
    se_eta2, se_phi2, se_pt2, se_vx2, se_vy2, se_vz2 = zip(*selected_items2)

    se_pt3 = list(se_pt) + list(se_pt2)
    se_eta3 = list(se_eta) + list(se_eta2)
    se_phi3 = list(se_phi) + list(se_phi2)
    se_vx3 = list(se_vx) + list(se_vx2)
    se_vy3 = list(se_vy) + list(se_vy2)
    se_vz3 = list(se_vz) + list(se_vz2)


    jet_eta, jet_phi, jet_pt,vtx,vty,vtz,  = fe.jet04(jet04_eta, jet04_phi, jet04_pt, se_eta3, se_phi3, se_pt3, se_vx3, se_vy3, se_vz3) # run2, run3
    # jet_eta, jet_phi, jet_pt,vtx,vty,vtz  = fe.jet04(jet04_eta, jet04_phi, jet04_pt, chg_eta, chg_phi, chg_pt, verx, very, verz) #mpi
    
    if len(jet_eta) == 0:
        continue



    pt_range = [(0.15,0.16,0.072),(0.16,0.18,0.062),(0.18,0.21,0.055),(0.21, 0.23,0.048),(0.23,0.27,0.041),(0.27,0.34,0.031),(0.34,0.5,0.023),(0.5,0.8,0.016),(0.8,1.4,0.011),(1.4,5,0.006),(5,200,0.003)]
    # reol = [0.072, 0.062, 0.055, 0.048, 0.041, 0.031, 0.023, 0.016, 0.011, 0.006, 0.003]
    for lower, upper,resol in pt_range:
        for i in range(len(jet_pt)):
            if lower < jet_pt[i] < upper:
                npx = np.random.normal(vtx[i], resol,1)
                npy = np.random.normal(vty[i], resol,1)
                resol_px.append(npx[0])
                resol_py.append(npy[0])
                pphi.append(jet_phi[i])
 

########################################################################################################################
    # num_samples = len(jet_eta)
    # random_position = np.random.normal(0, 0.1, num_samples)

    # for i in range(len(jet_eta)):
    #     resol_px.append(vtx[i] + random_position[i])
    #     resol_py.append(vty[i] + random_position[i])
    # print(resol_px, resol_py)
########################################################################################################################
    lin_x = np.linspace(-10,10,10)
    # direct = [np.tan(dir) for dir in jet_phi]
    direct = [np.tan(dir) for dir in pphi]
  
    x, y = symbols('x y')
    try:
        for i in range(len(direct)):
            f1 = Eq(direct[i] * (x - resol_px[i]) + resol_py[i], y)  # vtx, vty

            for j in range(i + 1, len(direct)):
                f2 = Eq(direct[j] * (x - resol_px[j]) + resol_py[j], y)  # vtx, vty

                results = solve([f1, f2], [x, y])
                if not results:
                    print(f"No solution found for indices {i} and {j}, skipping...")
                    raise ValueError("No solution found")
                px.append(results[x])
                py.append(results[y])
    except Exception as e:
        print(f"Error solving equations for event {eve}: {e}")
        continue 


    # for i in range(len(direct)):
    #     f1 = Eq(direct[i]*(x - resol_px[i]) + resol_py[i], y) #vtx, vty
        
    #     for j in range(i+1, len(direct)):
    #         f2 = Eq(direct[j]*(x - resol_px[j]) + resol_py[j], y)# vtx, vty

    #         results = solve([f1,f2], [x,y])
    #         px.append(results[x])
    #         py.append(results[y])

    # if len(px) == 0:
    #     continue

    # for num in range(len(direct)):
        
    #     lin_y = direct[num]*(lin_x - resol_px[num]) + resol_py[num] #vtx, vty
    #     plt.plot(lin_x, lin_y, color='skyblue')

    plt.scatter(px,py,s=100, color='skyblue')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    # plt.xlabel('x [mm]',fontsize=18)
    # plt.ylabel('y [mm]',fontsize=18)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    plt.gca().axes.xaxis.set_visible(False)
    plt.gca().axes.yaxis.set_visible(False)
    # plt.text(0.03, 0.98, 'PYTHIA8 pp $\sqrt{s} = 5 \, \mathrm{TeV}$',transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    # plt.text(0.03, 0.91, '$30 < p_{T} < 40 \, \mathrm{GeV/c}, \, |\eta| < 0.5$',transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    # plt.text(0.03, 0.84, f'Number of dots: {len(px)}',transform=plt.gca().transAxes, fontsize=14, verticalalignment='top')
    # plt.axis('off')
    plt.tight_layout()
    # plt.savefig(f'/home/jhg842/jet_tagging/images/ideal_line_20k/80100/test/b/{eve}.png',dpi=200)
    plt.savefig(f'/home/jhg842/jet_tagging/data/test/dot/2030/c/{eve}.png',dpi=200)
    plt.close('all')

    count += 1
    if count == 20000:
        break




