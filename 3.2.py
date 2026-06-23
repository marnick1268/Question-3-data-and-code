import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from scipy.fft import fft, ifft, fftfreq,fftn,ifftn,fftshift
from scipy.io import loadmat

from main import load_data,plot,plot_fourier,Fourier_transform,inverse_Fourier_transform


u_lst,v_lst,w_lst=load_data()
M=u_lst.shape[1]
N=u_lst.shape[0]

Re_tau=934
dx_plus=7.6413
dz_plus=3.8206
u_s=0.0454

x=np.arange(0,(N)*dx_plus,dx_plus)
z=np.arange(0,(M)*dz_plus,dz_plus)
X, Z = np.meshgrid(z, x)
u_mean=np.mean(u_lst)
u_mean_plus=np.mean(u_lst)/u_s

u_fluct=u_lst-u_mean
u_fluct_sq_bar=np.mean((u_fluct/u_s)**2)


v_mean=np.mean(v_lst)
v_mean_plus=np.mean(v_lst)/u_s

v_fluct=v_lst-v_mean
v_fluct_sq_bar=np.mean((v_fluct/u_s)**2)



w_mean=np.mean(w_lst)
w_mean_plus=np.mean(w_lst)/u_s

w_fluct=w_lst-w_mean
w_fluct_sq_bar=np.mean((w_fluct/u_s)**2)


Reynolds_stress=np.mean(u_fluct*v_fluct)/u_s**2

print(f'u"_+_bar={u_mean_plus},v_+_bar={v_mean_plus},w_+_bar={w_mean_plus}')
print(f'u^2_bar={u_fluct_sq_bar},v^2_bar={v_fluct_sq_bar},w^2_bar={w_fluct_sq_bar}')
print(f'u"v"_plus_bar={Reynolds_stress}')