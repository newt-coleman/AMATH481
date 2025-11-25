import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import gmres, bicgstab
from scipy.integrate import solve_ivp
from scipy.fft import fft2, ifft2, ifftshift
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmath

### OPERATORS
n = 128
nn = n**2
L = 20
h = L/n
nu = 0.001

### Laplace operator: same as in class
## AD: diagonal
AD = sp.spdiags(np.ones(nn) * -4, np.array([0]), m=nn, n=nn)
## AX
data = np.ones(nn)
AX = sp.spdiags(np.array([data, data, data, data]), np.array([-n, n, n-nn, nn-n]), m=nn, n=nn)

## AY
up_bound = np.zeros(nn)
up_bound[n-1::n] = 1
low_diag = np.ones(nn)
low_diag[n-1::n] = 0
up_diag = np.ones(nn)
up_diag[0::n] = 0
low_bound = np.zeros(nn)
low_bound[0::n] = 1

AY = sp.spdiags(np.array([up_bound, up_diag, low_diag, low_bound]), np.array([n-1, 1, -1, 1-n]), m=nn, n=nn)

A = (AD + AY  + AX) / h**2
A = A.todense()
A[0,0] = 2 / h**2
A = sp.csc_matrix(A)
# plt.imshow(AD.todense(), cmap='grey')
# plt.show()
# plt.imshow(AY.todense(), cmap='grey')
# plt.show()
# plt.imshow(AX.todense(), cmap='grey')
# plt.show()
# plt.imshow(A.todense(), cmap='grey')
# plt.show()

## partial x operator
diags = np.array([-n, n, -(nn-n), nn-n])
# data = np.array([[-1*np.ones(nn-2*n)], [np.ones(nn-2*n)], [np.ones(n)], [np.ones(n)]])
upperoff = np.ones(nn)
upperoff[:n] = np.zeros(n)
data = np.array([-1*np.ones(nn), upperoff, np.ones(nn), -np.ones(nn)])
B = sp.spdiags(data, diags, m=nn, n=nn) / (2*h)
# plt.imshow(B.todense(), cmap='grey')
# plt.show()

## partial y operator
bound = np.zeros(nn)
bound[0::n] = 1
low_diag = np.ones(nn)*-1
low_diag[n-1::n] = 0
up_diag = np.ones(nn)*1
up_diag[0::n] = 0
up_bound = np.zeros(nn)
up_bound[n-1::n] = -1

diags = np.array([1, -1, n-1, -n+1])
data = np.array([up_diag, low_diag, up_bound, bound])

C = sp.spdiags(data, diags, m=nn, n=nn) / (2*h)
# plt.imshow(C.todense(), cmap='grey')
# plt.show()

x2 = np.linspace(-L/2, L/2, n+1)

X, Y = np.meshgrid(x2[:-1], x2[:-1])


def solve_stream(init, vis=True, name=None):

    tspan = np.arange(0, 8.1, 0.5)
    def domega_dt(t, omega, A, B, C, nu):
        k = np.concatenate((np.arange(0, n//2), np.arange(-n//2, 0))) * 2*np.pi / 20 
        k[0] = 1e-6
        KX, KY = np.meshgrid(k, k)
        K = KX**2 + KY**2

        phi = ifft2(-1*fft2(omega.reshape(n,n, order="F")) / K)
        phi = np.real(phi.reshape(n*n, order="F"))
        return  (nu*(A@omega)) - ((B@phi) * (C@omega)) + ((C@phi) * (B@omega)) 

    sol = solve_ivp(domega_dt, (tspan[0], tspan[-1]), w0, t_eval=tspan, args = (A, B, C, nu))

    if vis:
        ims = []
        fig, ax = plt.subplots()
        for k in range(len(tspan)):
            frame = sol.y[:, k].reshape(n,n, order="F")
            im = ax.imshow(frame, cmap='RdPu', animated=True)
            if k == 0:
                ax.imshow(frame, cmap='RdPu')  # show an initial one first
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,
                                        repeat=True)
        if name != None:
            ani.save(filename=r"C:\Users\newtc\OneDrive\Desktop\AUT 25\AMATH 481\ACAIG\0" + name + ".gif", writer="pillow")
        plt.show()
    return sol.y.T

w0 = np.exp(-(X**2) - (Y**2)/20)
w0 = w0.reshape(-1, order="F")


w0 = np.exp(-((X-2)**2) - (Y**2)) + np.exp(-((X+2)**2) - (Y**2))
w0 = w0.reshape(-1, order="F")


######## Same charge vortex
w0 = np.exp(-((X-3)**2) - (Y**2)) + np.exp(-((X+3)**2) - (Y**2))
w0 = w0.reshape(-1, order="F")

# solve_stream(w0)

######## Opposite charge vortex
w0 = np.exp(-((X-3)**2) - (Y**2)/5) - np.exp(-((X+3)**2) - (Y**2)/5)
w0 = w0.reshape(-1, order="F")

# solve_stream(w0)


######## Random
num = np.random.randint(10, 15)
print(num)
w0 = np.zeros((n,n))

for k in range(num):
    posx = np.random.uniform(-L/2 + 2, L/2 - 2)
    posy = np.random.uniform(-L/2 + 2, L/2 - 2)

    stretchx = np.random.uniform(1, 3)
    stretchy=np.random.uniform(1,3)

    charge = np.random.choice([-1, 1])

    w0 += charge*np.exp(-((X-posx)**2)/stretchx - ((Y-posy)**2)/stretchy)
w0 = w0.reshape(-1, order="F")

# solve_stream(w0)

####### Collision
w0 = np.exp(-((X+2)**2) - ((Y-3)**2)/5) - np.exp(-((X-2)**2) - ((Y-3)**2)/5)

w0 += np.exp(-((X+1)**2) - ((Y+3)**2)/5) - np.exp(-((X-1)**2) - ((Y+3)**2)/5)
w0 = w0.reshape(-1, order="F")

# solve_stream(w0, name='collision')

####### Part B

n = 128
ys = np.linspace(-1, 1, n)
xs = np.linspace(-2, 0.5, n)

X, Y = np.meshgrid(xs[:-1], ys[:-1])

w0 = np.ones((n, n))


for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        z = complex(x, y)
        c = complex(x, y)

        color = 0
        for m in range(40):
            if cmath.isinf(z) or not cmath.isnan(z):
                try:
                    z = z**2 + c
                    color += 1
                except OverflowError:
                    break#k = 30
        color = color / 40 * 0.8
        if cmath.isinf(z) or cmath.isnan(z):
            w0[i, j] = color
plt.imshow(w0, cmap = 'RdPu')
plt.show()

w0 = w0.reshape(-1, order="F")

solve_stream(w0, name='mandelbrot')



