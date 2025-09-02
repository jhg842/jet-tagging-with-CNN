from re import L
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import matplotlib
import matplotlib.gridspec as gridspec
from scipy.integrate import simps
import cv2
from collections import namedtuple
from torchvision.transforms import ToTensor
# import explainable as expl
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')




##########################################################################################################################################
def Db(pb,pc,pl,fc):
    # f_c = 0.016 for dot, f_c = 0.05 for line
    d_b = np.log(pb/((1-fc)*pl + fc*pc))
    return d_b
##########################################################################################################################################
def Dc(pb,pc,pl,fb):
    # f_c = 0.016 for dot, f_c = 0.05 for line
    d_c = np.log(pc/((1-fb)*pl + fb*pb))
    return d_c

##########################################################################################################################################




def db(opt1, opt2):
    green = '#008000'
    yellow = '#FFB300'
    blue = '#00BFFF'
    dot_fc = 0.017
    line_fc = 0.06
    dot_softmax = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/softmax_{opt1}.npy')
    dot_label = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/label_{opt1}.npy')
    line_softmax = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/softmax_{opt2}.npy')
    line_label = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/label_{opt2}.npy')
    D_bb = []; D_bc = []; D_bl = []
    L_bb = []; L_bc = []; L_bl = []
    for i in range(len(dot_label)):
        if line_label[i] == 0:
            dpb_b = dot_softmax[i][0]; dpc_b = dot_softmax[i][1]; dpl_b = dot_softmax[i][2]
            lpb_b = line_softmax[i][0]; lpc_b = line_softmax[i][1]; lpl_b = line_softmax[i][2]
            d_bb = Db(dpb_b, dpc_b,dpl_b,dot_fc)
            l_bb = Db(lpb_b, lpc_b, lpl_b,line_fc)
            D_bb.append(d_bb)
            L_bb.append(l_bb)
        elif line_label[i] == 1:
            dpb_c = dot_softmax[i][0]; dpc_c = dot_softmax[i][1]; dpl_c = dot_softmax[i][2]
            lpb_c = line_softmax[i][0]; lpc_c = line_softmax[i][1]; lpl_c = line_softmax[i][2]
            d_bc = Db(dpb_c, dpc_c, dpl_c,dot_fc)
            l_bc = Db(lpb_c, lpc_c, lpl_c,line_fc)
            D_bc.append(d_bc)
            L_bc.append(l_bc)    
        else:
            dpb_l = dot_softmax[i][0]; dpc_l = dot_softmax[i][1]; dpl_l = dot_softmax[i][2]
            lpb_l = line_softmax[i][0]; lpc_l = line_softmax[i][1]; lpl_l = line_softmax[i][2]
            d_bl = Db(dpb_l, dpc_l, dpl_l,dot_fc)
            l_bl = Db(lpb_l, lpc_l, lpl_l,line_fc)
            D_bl.append(d_bl)
            L_bl.append(l_bl)

    # weight_1 = np.ones_like(D_bb) * 0.04
    # weight_2 = np.ones_like(D_bc) * 0.08
    fig = plt.figure(figsize=(14, 8))
    plt.hist(D_bb, bins = 20, histtype='step',label='dot b-jets', color=green,linestyle='dashed',linewidth=2,density=True)
    plt.hist(D_bc, bins = 20, histtype='step',label = 'dot c-jets', color=blue,linestyle='dashed',linewidth=2,density=True)
    plt.hist(D_bl, bins = 20, histtype='step',label='dot l-jets', color=yellow,linestyle='dashed',linewidth=2,density=True)
    plt.hist(L_bb, bins = 20, histtype='step',label='line b-jets', color=green,density=True)
    plt.hist(L_bc, bins = 20, histtype='step',label = 'line c-jets', color=blue,density=True)
    plt.hist(L_bl, bins = 20, histtype='step',label='line l-jets', color=yellow,density=True)

    plt.tick_params(axis='x', labelsize=22)  # x축 눈금의 폰트 사이즈를 12로 설정
    plt.tick_params(axis='y', labelsize=22)
    plt.yscale('log')
    plt.ylim((10**-4)/2, (10**1))
    plt.xlim(-11,20)
    plt.xlabel('$D_{c}$',fontsize=28)
    plt.ylabel('a.u.',fontsize=28)
    plt.text(0.03, 0.98, 'PYTHIA8 pp $\sqrt{s} = 5 \, \mathrm{TeV}$',transform=plt.gca().transAxes, fontsize=24, verticalalignment='top')
    # plt.text(0.03, 0.91, '$\sqrt{s} = 5 \, \mathrm{TeV}, \, t\bar{t} \, \mathrm{events}$',transform=plt.gca().transAxes, fontsize=20, verticalalignment='top')
    # plt.text(0.03, 0.88, 'Anti-$k_{T} \, R = 0.4 \, \mathrm{PFlow \, jets}$\n',transform=plt.gca().transAxes, fontsize=18, verticalalignment='top')
    plt.text(0.03, 0.91, '$30 < p_{T} < 40 \, \mathrm{GeV/c}, \, |\eta| < 0.5$',transform=plt.gca().transAxes, fontsize=24, verticalalignment='top')
    # plt.text(0.03, 0.84, 'w/ realistic tracking efficiency',transform=plt.gca().transAxes, fontsize=24, verticalalignment='top')
    # plt.text(0.03, 1.05, 'w/ realistic tracking efficiency and momentum resolution',transform=plt.gca().transAxes, fontsize=24, verticalalignment='top')
    
    plt.tight_layout(pad=2.5)
    plt.legend(loc='upper right', frameon=False, ncol=2, bbox_to_anchor=(1, 1), fontsize=20)
    plt.savefig(f'/home/jhg842/jet_tagging/eval_performance/results/D_b_test_full.pdf', dpi=300)
    plt.close('all')

##########################################################################################################################################


#################################################################################################################################################################
def rejection():
    dot_softmax = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/softmax_dot_3040.npy')
    dot_label = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/label_dot_3040.npy')
    line_softmax = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/softmax_line_3040.npy')
    line_label = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/label_line_3040.npy')

    orange = '#FF4500'
    sky_blue = '#87CEFA'
    dot_fc = 0.017
    line_fc = 0.06


    c_ratio = []; l_ratio = []; cc_ratio = []; ll_ratio = []    

 



    ld_cratio = [b_i / a_i for a_i, b_i in zip(dot_cjet_rejec, line_cjet_rejec)]
    # dd_cratio = [b_i / a_i for a_i, b_i in zip(dot_bjet_rejec, dot_bjet_rejec)]
    # dd_lratio = [b_i / a_i for a_i, b_i in zip(dot_ljet_rejec_cjet, line_ljet_rejec_cjet)]
    ld_lratio = [b_i / a_i for a_i, b_i in zip(dot_ljet_rejec, line_ljet_rejec)]

    print(ld_cratio, ld_lratio)

    fig = plt.figure(figsize=(12, 6))
    # gs = fig.add_gridspec(3, 2, height_ratios=[3, 2, 2])
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])
    # 첫 번째 행 서브플롯
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.plot(dot_bjet_effi, dot_cjet_rejec,color=sky_blue, label='dot images')
    ax1.plot(line_bjet_effi, line_cjet_rejec, color=orange, label='line images')
    ax1.set_yscale('log')
    ax1.set_ylabel('c-jet rejection',fontsize=18)
    ax1.set_xlim(0.3,1)
    ax1.set_ylim(0,10**3)
    ax1.legend(loc='upper right', frameon=False, ncol=1, fontsize=13)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)  # X축 틱 유지
    ax1.text(0.03, 0.98, 'PYTHIA8 pp $\sqrt{s} = 5 \, \mathrm{TeV}$',transform=ax1.transAxes, fontsize=11, verticalalignment='top')
    # ax1.text(0.03, 0.89, '$\sqrt{s} = 5 \, \mathrm{TeV}, \, t\bar{t} \, \mathrm{events}$',transform=ax1.transAxes, fontsize=14, verticalalignment='top')
    # ax1.text(0.03, 0.86, 'Anti-$k_{T} \, R = 0.4 \, \mathrm{PFlow \, jets}$\n',transform=ax1.transAxes, fontsize=10, verticalalignment='top')
    ax1.text(0.03, 0.90, '$30 < p_{T} < 40 \, \mathrm{GeV/c}, \, |\eta| < 0.5$',transform=ax1.transAxes, fontsize=11, verticalalignment='top')
    ax1.text(0.03, 0.82, 'w/ realistic tracking efficiency and momentum resolution',transform=ax1.transAxes, fontsize=11, verticalalignment='top')


    ax2.plot(dot_bjet_effi, dot_ljet_rejec, color=sky_blue, label='dot images')
    ax2.plot(line_bjet_effi, line_ljet_rejec, color=orange, label='line images')
    ax2.set_yscale('log')
    ax2.set_ylabel('Light-jet rejection',fontsize=18)
    ax2.set_xlim(0.3,1)
    ax2.set_ylim(0,10**5)
    ax2.legend(loc='upper right', frameon=False, ncol=1, fontsize=13)
    ax2.tick_params(axis='y', labelsize=14)
    ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)  # X축 틱 유지
    ax2.text(0.03, 0.98, 'PYTHIA8 pp $\sqrt{s} = 5 \, \mathrm{TeV}$',transform=ax2.transAxes, fontsize=11, verticalalignment='top')
    # ax2.text(0.03, 0.91, '$\sqrt{s} = 5 \, \mathrm{TeV}, \, t\bar{t} \, \mathrm{events}$',transform=ax2.transAxes, fontsize=14, verticalalignment='top')
    # ax2.text(0.03, 0.86, 'Anti-$k_{T} \, R = 0.4 \, \mathrm{PFlow \, jets}$\n',transform=ax2.transAxes, fontsize=10, verticalalignment='top')
    # ax2.text(0.03, 0.90, '$30 < p_{T} < 40 \, \mathrm{GeV/c}, \, |\eta| < 0.5$, $f_{b} = 0.03$',transform=ax2.transAxes, fontsize=11, verticalalignment='top')
    ax2.text(0.03, 0.90, '$30 < p_{T} < 40 \, \mathrm{GeV/c}, \, |\eta| < 0.5$',transform=ax2.transAxes, fontsize=11, verticalalignment='top')
    ax2.text(0.03, 0.82, 'w/ realistic tracking efficiency and momentum resolution',transform=ax2.transAxes, fontsize=11, verticalalignment='top')
    # 두 번째 행 서브플롯
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax3.plot(dot_bjet_effi, ld_cratio,color=orange)
    # ax3.plot(dot_bjet_effi, dd_cratio,color= sky_blue)
    ax3.yaxis.set_label_coords(-0.1, 0.5)
    ax3.tick_params(axis='x', bottom=True, labelbottom=True,labelsize=14)
    ax3.tick_params(axis='y', left=True, labelleft=True,labelsize=14)

    # 위쪽과 오른쪽 틱을 설정하고, 라벨을 제거
    ax3.tick_params(axis='x', top=True, labeltop=False)
    ax3.set_xlim(0.3,1)
    ax3.set_ylabel('Ratio (line/dot)',fontsize=18)
    ax3.set_xlabel('b-jet tagging efficiency',fontsize=18)
    ax3.set_ylim(0,5)

    ax4.plot(dot_bjet_effi, ld_lratio,color= orange)
    # ax4.plot(dot_bjet_effi, dd_lratio, color = sky_blue)
    ax4.tick_params(axis='x', bottom=True, labelbottom=True,labelsize=14)
    ax4.tick_params(axis='y', left=True, labelleft=True,labelsize=14)

    # 위쪽과 오른쪽 틱을 설정하고, 라벨을 제거
    ax4.tick_params(axis='x', top=True, labeltop=False)
    ax4.set_xlim(0.3,1)
    ax4.yaxis.set_label_coords(-0.1, 0.5)
    ax4.set_ylabel('Ratio (line/dot)',fontsize=18)
    ax4.set_xlabel('b-jet tagging efficiency',fontsize=18)



    plt.tight_layout()  # 위아래 간격 조정
    # plt.savefig('/Users/janghangil/Desktop/jet_tagging/eval_performance/results/figures/effi_rejection_efficiency_full.pdf', dpi=300)
    plt.savefig('/home/jhg842/jet_tagging/eval_performance/results/effi_rejection_efficiency_full.pdf', dpi=300)





#################################################################################################################################################################
def classification():
    dot_softmax = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/softmax_dot_3040.npy')
    dot_label = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/label_dot_3040.npy')
    line_softmax = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/softmax_line_3040.npy')
    line_label = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/label_line_3040.npy')

    orange = '#FF4500'
    sky_blue = '#87CEFA'
    dot_fc = 0.017
    line_fc = 0.06
    thresholds1 = np.arange(-10, 9, 0.5)
    thresholds2 = np.arange(-10, 13, 0.5)

    line_effi = []; line_pur = []; line_spec = []; dot_effi = []; dot_pur = []; dot_spec = []
    weight_b = 0.04
    weight_c = 0.08
    weight_l = 1
    for thr in thresholds1:
        tp, tn, fp, fn = 0, 0, 0, 0

        for pr in range(len(dot_label)):
            pb = dot_softmax[pr][0]; pc = dot_softmax[pr][1]; pl = dot_softmax[pr][2]
            bjet_db = Db(pb, pc, pl,dot_fc)
            weight = weight_b if dot_label[pr] == 0 else (weight_c if dot_label[pr] == 1 else weight_l)
            if bjet_db > thr:
                if dot_label[pr] == 0:
                    tp += weight
                else:
                    fp += weight
            else:
                if dot_label[pr] == 0:
                    fn += weight
                else:
                    tn += weight

        efficiency = tp / (tp + fn) if (tp + fn) != 0 else 0
        purity = tp / (tp + fp) if (tp + fp) != 0 else 0
        spec = fp/(tn + fp) if (tn + fp) != 0 else 0
        dot_effi.append(efficiency)
        dot_pur.append(purity)
        dot_spec.append(spec)

    #######################################################################################################################

    for thr in thresholds2:
        tp, tn, fp, fn = 0, 0, 0, 0

        for pr in range(len(line_label)):
            pb = line_softmax[pr][0]; pc = line_softmax[pr][1]; pl = line_softmax[pr][2]
            bjet_db = Db(pb, pc, pl, line_fc)
            weight = weight_b if line_label[pr] == 0 else (weight_c if line_label[pr] == 1 else weight_l)
            if bjet_db > thr:
                if line_label[pr] == 0:
                    tp += weight
                else:
                    fp += weight
            else:
                if line_label[pr] == 0:
                    fn += weight
                else:
                    tn += weight

        efficiency = tp / (tp + fn) if (tp + fn) != 0 else 0
        purity = tp / (tp + fp) if (tp + fp) != 0 else 0
        spec = fp/(tn + fp) if (tn + fp) != 0 else 0
        line_effi.append(efficiency)
        line_pur.append(purity)
        line_spec.append(spec)
    #######################################################################################################################
    for i in range(len(dot_effi)):
        if 0.69<dot_effi[i]<0.71:
            print('dot efficiency',dot_effi, 'dot purity', dot_pur[i])
        elif 0.69<line_effi[i] < 0.71:
            print('line efficiency',line_effi, 'line purity', line_pur[i])
    # train_auc = simps(line_effi, line_spec)
    # val_auc = simps(dot_effi, dot_spec)
    line_auc = np.trapz(line_effi[::-1], line_spec[::-1])
    dot_auc = np.trapz(dot_effi[::-1], dot_spec[::-1])
    sv_eff = [0.2517, 0.3406, 0.3872, 0.2367, 0.3170, 0.3586, 0.3785, 0.2208, 0.2979, 0.3390, 0.2134, 0.2802, 0.3167, 0.3315]
    # sv_fpr = [0.0307, 0.0513, 0.0700, 0.0243, 0.0408, 0.0556, 0.0699, 0.0200, 0.0338, 0.0467, 0.0177, 0.0288, 0.0398, 0.0508]
    sv_pur = [0.4222, 0.3718, 0.3303, 0.4650, 0.4094, 0.3650, 0.3256, 0.4957, 0.4402, 0.3927, 0.5179, 0.4641, 0.4150, 0.3675]
    
    fig = plt.figure(figsize=(14,7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dot_pur,dot_effi,color=sky_blue, label = 'dot images')
    ax1.plot(line_pur,line_effi,color=orange, label = 'line images')
    ax1.text(0.03, 0.98, 'PYTHIA8 pp $\sqrt{s} = 5 \, \mathrm{TeV}$',transform=ax1.transAxes, fontsize=18, verticalalignment='top')
    # ax1.text(0.03, 0.91, '$\sqrt{s} = 5 \, \mathrm{TeV}, \, t\bar{t} \, \mathrm{events}$',transform=ax1.transAxes, fontsize=18, verticalalignment='top')
    # ax1.text(0.03, 0.88, 'Anti-$k_{T} \, R = 0.4 \, \mathrm{PFlow \, jets}$\n',transform=ax1.gca().transAxes, fontsize=10, verticalalignment='top')
    ax1.text(0.03, 0.91, '$30 < p_{T} < 40 \, \mathrm{GeV/c}, \, |\eta| < 0.5$',transform=ax1.transAxes, fontsize=18, verticalalignment='top')
    # ax1.text(0.03, 0.83, 'w/ realistic tracking efficiency and momentum resolution',transform=ax1.transAxes, fontsize=18, verticalalignment='top')
    # ax1.scatter(sv_eff,sv_pur, label='SV',color='red')
    ax1.set_ylabel('b-jet efficiency',fontsize=24)
    ax1.set_xlabel('b-jet purity',fontsize=24)
    ax1.tick_params(axis='x', bottom=True, labelbottom=True,labelsize=18)
    ax1.tick_params(axis='y', left=True, labelleft=True,labelsize=18)
    ax1.set_xlim(0,1)
    ax1.set_ylim(0,1.4)
    ax1.legend(loc='upper right', frameon=False, ncol=1,bbox_to_anchor=(0.6, 0.2), fontsize=18)


    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(dot_spec, dot_effi,color=sky_blue, label = f'dot images, AUC = {dot_auc:.2f}')
    ax2.plot(line_spec, line_effi,color=orange, label = f'line images, AUC = {line_auc:.2f}')
    # ax1.plot([0,1],[0,1], linestyle='dashed')
    ax2.text(0.03, 0.98, 'PYTHIA8 pp $\sqrt{s} = 5 \, \mathrm{TeV}$',transform=ax2.transAxes, fontsize=18, verticalalignment='top')
    # ax2.text(0.03, 0.91, '$\sqrt{s} = 5 \, \mathrm{TeV}, \, t\bar{t} \, \mathrm{events}$',transform=ax2.transAxes, fontsize=18, verticalalignment='top')
    # ax2.text(0.03, 0.88, 'Anti-$k_{T} \, R = 0.4 \, \mathrm{PFlow \, jets}$\n',transform=ax1.gca().transAxes, fontsize=10, verticalalignment='top')
    ax2.text(0.03, 0.91, '$30 < p_{T} < 40 \, \mathrm{GeV/c}, \, |\eta| < 0.5$',transform=ax2.transAxes, fontsize=18, verticalalignment='top')
    # ax2.text(0.03, 0.83, 'w/ realistic tracking efficiency and momentum resolution',transform=ax2.transAxes, fontsize=18, verticalalignment='top')
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1.4)
    ax2.set_xlabel('False Positive Rate',fontsize=24)
    ax2.set_ylabel('True Positive Rate',fontsize=24)
    ax2.tick_params(axis='x', bottom=True, labelbottom=True,labelsize=18)
    ax2.tick_params(axis='y', left=True, labelleft=True,labelsize=18)
    ax2.legend(loc='upper right', frameon=False, ncol=1,bbox_to_anchor=(0.8, 0.2), fontsize=18)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.25, wspace=0.2, hspace=0.2)
    plt.savefig('/home/jhg842/jet_tagging/eval_performance/results/classification.pdf',dpi=300)
    plt.close('all')

    # num_data(testloader)
    # db()
    # rejection()
    # classification() 
# #######################################################################################################################
def probability():
    classes = ['$p_b$','$p_c$','$p_l$']
    # 클래스 별로 히스토그램을 저장할 리스트 생성
    hist_pb = [[] for _ in range(3)]
    hist_pc = [[] for _ in range(3)]
    hist_pl = [[] for _ in range(3)]
    line_hist_pb = [[] for _ in range(3)]
    line_hist_pc = [[] for _ in range(3)]
    line_hist_pl = [[] for _ in range(3)]

    green = '#008000'
    yellow = '#FFB300'
    blue = '#00BFFF'
    dot_softmax = np.load(f'/Users/janghangil/Desktop/jet_tagging/eval_performance/results/figures/probability/dot/softmax_test_dot.npy')
    dot_label = np.load(f'/Users/janghangil/Desktop/jet_tagging/eval_performance/results/figures/probability/dot/label_test_dot.npy')
    line_softmax = np.load(f'/Users/janghangil/Desktop/jet_tagging/eval_performance/results/figures/probability/line/softmax_test_line.npy')
    line_label = np.load(f'/Users/janghangil/Desktop/jet_tagging/eval_performance/results/figures/probability/line/label_test_line.npy')

    for i in range(len(dot_label)):
        if dot_label[i] == 0:
            hist_pb[0].append(dot_softmax[i][0])
            hist_pc[0].append(dot_softmax[i][1])
            hist_pl[0].append(dot_softmax[i][2])

            line_hist_pb[0].append(line_softmax[i][0])
            line_hist_pc[0].append(line_softmax[i][1])
            line_hist_pl[0].append(line_softmax[i][2])
        elif dot_label[i] == 1:
            hist_pb[1].append(dot_softmax[i][0])
            hist_pc[1].append(dot_softmax[i][1])
            hist_pl[1].append(dot_softmax[i][2])

            line_hist_pb[1].append(line_softmax[i][0])
            line_hist_pc[1].append(line_softmax[i][1])
            line_hist_pl[1].append(line_softmax[i][2])
        else:
            hist_pb[2].append(dot_softmax[i][0])
            hist_pc[2].append(dot_softmax[i][1])
            hist_pl[2].append(dot_softmax[i][2])

            line_hist_pb[2].append(line_softmax[i][0])
            line_hist_pc[2].append(line_softmax[i][1])
            line_hist_pl[2].append(line_softmax[i][2])
    # 클래스 별로 히스토그램 그리기
    for i in range(3):
        # weight_b = np.ones_like(hist_pb[i]) * 0.04
        # weight_c = np.ones_like(hist_pc[i]) * 0.08
        plt.figure()
        plt.hist(hist_pb[i],bins = 100, histtype='step', label='dot b-jet',color=green)
        plt.hist(hist_pc[i],bins = 100, histtype='step', label='dot c-jet',color=blue)
        plt.hist(hist_pl[i],bins = 100, histtype='step', label='dot l-jet',color=yellow)

        plt.hist(line_hist_pb[i],bins = 100, histtype='step', label='line b-jet',color=green, linestyle='dashed')
        plt.hist(line_hist_pc[i],bins = 100, histtype='step', label='line c-jet',color=blue, linestyle='dashed')
        plt.hist(line_hist_pl[i],bins = 100, histtype='step', label='line l-jet',color=yellow, linestyle='dashed')

        plt.text(0.03, 0.98, 'PYTHIA8 Simulation',transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
        plt.text(0.03, 0.93, '$\sqrt{s} = 13 \, \mathrm{TeV}, \, t\bar{t} \, \mathrm{events}$',transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
        plt.text(0.03, 0.88, 'Anti-$k_{T} \, R = 0.4 \, \mathrm{PFlow \, jets}$\n',transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
        plt.text(0.03, 0.83, '$30 < p_{T} < 40 \, \mathrm{GeV/c}, \, |\eta| < 0.5$',transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    
        plt.legend(loc='upper right', frameon=False, ncol=2, bbox_to_anchor=(1, 1), fontsize='small')
        plt.xlabel(classes[i])
        plt.tick_params(axis='x', labelsize=12)  # x축 눈금의 폰트 사이즈를 12로 설정
        plt.tick_params(axis='y', labelsize=12)
        plt.yscale('log')
        plt.ylim(0, (10**6)/2)
        plt.xlabel(classes[i],fontsize=14)
        # plt.title('Histogram for Class ' + str(i))
        plt.savefig(f'/Users/janghangil/Desktop/jet_tagging/eval_performance/results/figures/histogram_class_{i}.pdf',dpi=300)
        plt.close('all')

#######################################################################################################################

def roc():
    dot_softmax = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/softmax_dot_3040.npy')
    dot_label = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/label_dot_3040.npy')
    line_softmax = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/softmax_line_3040.npy')
    line_label = np.load(f'/home/jhg842/jet_tagging/eval_performance/saved_model/label_line_3040.npy')

    orange = '#FF4500'
    sky_blue = '#87CEFA'

    dot_fc = 0.017
    line_fc = 0.06
    # dot_softmax =oad('/Users/janghangil/Desktop/jet_tagging/eval_performance/results/figures/probability/dot/label.npy')
    thresholds1 = np.arange(-11, 13, 0.1)
    thresholds2 = np.arange(-11, 13, 0.1)

    line_spec = []; line_sen = []; dot_spec = []; dot_sen = [];

    for thr in thresholds1:
        tp, tn, fp, fn = 0, 0, 0, 0

        for pr in range(len(dot_label)):
            pb = dot_softmax[pr][0]; pc = dot_softmax[pr][1]; pl = dot_softmax[pr][2]
            bjet_db = Db(pb, pc, pl, dot_fc)
            if bjet_db > thr:
                if dot_label[pr] == 0:
                    tp += 1
                else:
                    fp += 1
            else:
                if dot_label[pr] == 0:
                    fn += 1
                else:
                    tn += 1

            # print(tn + fp)
        spec = fp/(tn + fp)
        sen = tp/(tp + fn)
        dot_spec.append(spec)
        dot_sen.append(sen)

   
    for thr in thresholds2:
        tp, tn, fp, fn = 0, 0, 0, 0

        for pr in range(len(line_label)):
            pb = line_softmax[pr][0]; pc = line_softmax[pr][1]; pl = line_softmax[pr][2]
            bjet_db = Db(pb, pc, pl, line_fc)
            if bjet_db > thr:
                if line_label[pr] == 0:
                    tp += 1
                else:
                    fp += 1
            else:
                if line_label[pr] == 0:
                    fn += 1
                else:
                    tn += 1


            # print(tn + fp)
        spec = fp/(tn + fp)
        sen = tp/(tp + fn)
        line_spec.append(spec)
        line_sen.append(sen)





    train_auc = simps(line_sen, line_spec)
    val_auc = simps(dot_sen, dot_spec)
    # plt.plot(run1_spec, run1_sen, label = f'dot data, AUC:{dot_auc:.2f}')
    # plt.plot(dot_spec, dot_sen, label = f'line data, AUC:{line_auc:.2f}')
    plt.plot(dot_spec, dot_sen,color=sky_blue, label = f'dot images, AUC:{val_auc:.2f}')
    plt.plot(line_spec, line_sen,color=orange, label = f'line images, AUC:{train_auc:.2f}')
    # plt.plot([0,1],[0,1], linestyle='dashed')
    plt.text(0.03, 0.98, 'PYTHIA8 Simulation',transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
    # plt.text(0.03, 0.93, '$\sqrt{s} = 13 \, \mathrm{TeV}, \, t\bar{t} \, \mathrm{events}$',transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    # plt.text(0.03, 0.88, 'Anti-$k_{T} \, R = 0.4 \, \mathrm{PFlow \, jets}$\n',transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.text(0.03, 0.83, '$30 < p_{T} < 40 \, \mathrm{GeV/c}, \, |\eta| < 0.5$',transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')
    plt.ylim(0,1.4)
    plt.xlabel('False Positive Rate',fontsize=18)
    plt.ylabel('True Positive Rate',fontsize=18)
    plt.legend(loc='upper right', frameon=False, ncol=1, bbox_to_anchor=(1, 1), fontsize=10)
    plt.savefig('/home/jhg842/jet_tagging/eval_performance/results/roc_curve.pdf',dpi=300)
##############################################################################################################################

if __name__ =='__main__':
    # num_data(testloader)
    # db('dot_3040','line_3040')
    # probability('line_3040')
    # rejection()
    # pur_effi()
    # roc()
    classification()
