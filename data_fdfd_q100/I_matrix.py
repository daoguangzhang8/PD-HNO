import numpy as np
import numpy.linalg
from scipy.sparse import spdiags


def I_matrix(nx, nz, c, d, e):
    val_l = np.ones((nz,nx)); val_l[:,nx-1]=0;
    val_l = np.reshape(val_l.transpose(), (nx*nz, 1))
    val_r = np.ones((nz,nx)); val_r[:,0]=0;
    val_r = np.reshape(val_r.transpose(), (nx*nz, 1))
    val_u = np.ones((nz,nx)); val_u[nz-1,:]=0;
    val_u = np.reshape(val_u.transpose(), (nx*nz, 1))
    val_d = np.ones((nz,nx)); val_d[0,:]=0;
    val_d = np.reshape(val_d.transpose(), (nx*nz, 1))

    A = np.zeros( (nx*nz,4) );   
    A[0:,2] = np.ravel(val_l);
    A[0:,3] = np.ravel(val_r);
    A[0:,0] = np.ravel(val_u);
    A[0:,1] = np.ravel(val_d);

    I0 = spdiags( (1/4)*d*A.transpose(),[-1, 1, -nz, nz], nz*nx , nz*nx);


    val_l = np.ones((nz,nx)); val_l[nz-1,:]=0;
    val_l = np.reshape(val_l.transpose(), (nx*nz, 1))
    val_r = np.ones((nz,nx)); val_r[0,:]=0;
    val_r = np.reshape(val_r.transpose(), (nx*nz, 1))
    val_u = np.ones((nz,nx)); val_u[:,0]=0; val_u[nz-1,:]=0;
    val_u = np.reshape(val_u.transpose(), (nx*nz, 1))
    val_d = np.ones((nz,nx)); val_d[0,:]=0;
    val_d = np.reshape(val_d.transpose(), (nx*nz, 1))

    A = np.zeros( (nx*nz,4) );   
    A[0:,0] = np.ravel(val_l);
    A[0:,1] = np.ravel(val_r);
    A[0:,2] = np.ravel(val_u);
    A[0:,3] = np.ravel(val_d);

    I45 = spdiags( (1/4)*e*A.transpose(),[-nz-1, -nz+1, nz-1, nz+1], nz*nx , nz*nx );

    I = spdiags( np.ones((1,nx*nz))*c,[0], nz*nx , nz*nx)  + I0 + I45;  


    return I