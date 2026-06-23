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

u_fluct_plus=(u_lst-u_mean)/u_s
u_fluct=u_fluct_plus*u_s
w_mean=np.mean(w_lst)

w_fluct_plus=(w_lst-w_mean)/u_s
w_fluct=w_fluct_plus*u_s
v_mag_fluct_star=np.sqrt(u_fluct_plus**2+w_fluct_plus**2)



row_mask = x < 2500
col_mask = z < 1000

# u_subset = u_lst
skip=8
fig,ax=plt.subplots(2,2)
plot(ax[0][0],u_lst[row_mask][:, col_mask])
plot(ax[0][1],u_fluct_plus[row_mask][:, col_mask])

plot(ax[1][0],w_lst[row_mask][:, col_mask])
plot(ax[1][1],w_fluct_plus[row_mask][:, col_mask])


fig,ax=plt.subplots(1,1)
plot(ax,v_mag_fluct_star[row_mask][:, col_mask])

ax.quiver(
    Z[row_mask][:, col_mask][::skip,::skip],          # x-locations of arrows
    X[row_mask][:, col_mask][::skip,::skip],          # y-locations of arrows
    u_fluct_plus[row_mask][:, col_mask][::skip,::skip],      # horizontal component
    w_fluct_plus[row_mask][:, col_mask][::skip,::skip],      # vertical component
    color='white'
)


plt.show()


# Fourier-space wavenumbers
kx = 2*np.pi*fftfreq(N, d=dx_plus)
kz = 2*np.pi*fftfreq(M, d=dz_plus)

KZ, KX = np.meshgrid(kz, kx)

# Fourier transforms
u_hat = fftn(u_fluct)
w_hat = fftn(w_fluct)

# Spectral derivatives
dudz = np.real(ifftn(1j * KZ * u_hat))
dwdx = np.real(ifftn(1j * KX * w_hat))

# Out-of-plane vorticity
omega_y = dudz - dwdx

# Wall-unit normalisation
omega_y_plus = omega_y / (u_s**2)

print(np.mean(omega_y_plus))
fig,ax=plt.subplots(1,1)
plot(ax,omega_y_plus[row_mask][:, col_mask])
print(np.mean(omega_y_plus))
print(np.log(188)/0.41+5)
plt.show()