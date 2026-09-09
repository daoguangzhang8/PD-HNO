import numpy as np
import math as mm
import numpy.linalg
import numpy.matlib

def optimal_Parameters(Gmin, Gmax):
    
    if Gmin == Gmax:     # It happens when we have a homogeneous media
        l = 500;
        r = 1;
    else:
        l = 10
        r = 100
    
    m = np.arange(1, l+1, 1); 
    n = np.arange(1, r+1, 1); 

    theta = (m-1)*mm.pi/(4*(l-1))
    theta = np.matlib.repmat(theta, r, 1)

    if Gmin == Gmax:
        iG = np.ones((1,r))*1/Gmax
    else:
        iG = 1/Gmax + (n-1)/(r-1)*(1/Gmin-1/Gmax)

    G = 1/iG
    G = np.matlib.repmat(G, l, 1).transpose()


    P = np.cos(2*mm.pi/G*np.cos(theta));
    Q = np.cos(2*mm.pi/G*np.sin(theta));
    


    S1= 2*G**2*(1 - P - Q + P*Q);

    S2= mm.pi**2*(2 - P - Q);

    S3= 2*mm.pi**2*(1 - P*Q);

    S4= 2*mm.pi**2 + G**2*(2*P*Q - P - Q);

    S1 = np.reshape(S1.transpose(), (len(S1)*len(S1[0]), 1))
    S2 = np.reshape(S2.transpose(), (len(S2)*len(S2[0]), 1))
    S3 = np.reshape(S3.transpose(), (len(S3)*len(S3[0]), 1))
    y = np.reshape(S4.transpose(), (len(S4)*len(S4[0]), 1))

    A = np.zeros( (len(S1)*len(S1[0]),3) );
    A[0:,0] = np.ravel(S1);
    A[0:,1] = np.ravel(S2);
    A[0:,2] = np.ravel(S3);

    b, d, e = np.linalg.lstsq(A, y, rcond=None)[0]
    c = 1 - d - e
    
    return b, c, d, e
