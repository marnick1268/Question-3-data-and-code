import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors as colors
from scipy.fft import fft, ifft, fftfreq,fftn,ifftn,fftshift
from scipy.io import loadmat


"data loading"
def load_data():
    data = loadmat("Re950_190_plane80.mat")
    u_lst=data['u']  #x,y,z
    v_lst=data['v']  #x,y,z
    w_lst=data['w']  #x,y,z
    return u_lst,v_lst,w_lst


"Fourier transforms"
def Fourier_transform(x,N=192):
    return fftn((x-np.mean(x)),norm='ortho')/(np.sqrt((N)**3))

def inverse_Fourier_transform(x,N=192):
    return np.real(ifftn((x),norm='ortho')*(np.sqrt((N)**3)))



"Plots"
def plot(ax,u_lst,title=""):
    M=u_lst.shape[1]
    N=u_lst.shape[0]
    dx_plus=7.6413
    dz_plus=3.8206
    x=np.arange(0,(N)*dx_plus,dx_plus)
    y=np.arange(0,(M)*dz_plus,dz_plus)

    


    Y, X = np.meshgrid(y, x)
    
    pcm=ax.pcolormesh(X,Y,u_lst, vmin=np.min(u_lst), vmax=np.max(u_lst))
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.colorbar(pcm, ax=ax)




def plot_fourier(ax, u_lst, title=""):
    # Calculate magnitude
    mag = np.abs(u_lst)
    
    pcm = ax.pcolormesh(mag, 
                           norm=colors.LogNorm(vmin=max(1e-5, mag.min()), vmax=mag.max()),
                           cmap='viridis')
    ax.set_title(title)
    ax.set_xlabel(r'$ \xi_x $')
    ax.set_ylabel(r'$ \xi_y $')
    plt.colorbar(pcm, ax=ax)







if __name__=='__main__':
    u_lst,v_lst,w_lst=load_data()
    M=u_lst.shape[1]
    N=u_lst.shape[0]

    dx_plus=7.6413
    dz_plus=3.8206

    fig,ax=plt.subplots(1,1)
    plot(ax,u_lst)

    plt.show()