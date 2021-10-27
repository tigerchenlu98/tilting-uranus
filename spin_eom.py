import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import sys

def ward(s, t, alpha_const, pfunc, qfunc, hfunc, kfunc):
    '''Integrates the E.O.M from Ward 1979'''
    # unpack parameters
    sx, sy, sz = s
    
    # define constants
    alpha = alpha_const / 206265
    p = pfunc(t)
    q = qfunc(t)
    h = hfunc(t)
    k = kfunc(t)
    
    xi = np.sqrt(1 - (1 / 4) * (p**2 + q**2))
    eta = 1 - (1 / 2) * (p**2 + q**2)

    prefactor = alpha * (1 - h**2 - k**2)**(-3 / 2) * (sx * p * xi - sy * q * xi + sz * eta)
    dsx = prefactor * (sy * eta + sz * q * xi)
    dsy = prefactor * (sz * p * xi - sx * eta)
    dsz = prefactor * (-sx * q * xi - sy * p * xi)

    return np.array([dsx, dsy, dsz])

def obliquity(vector, norm):
    u_vector = vector / np.linalg.norm(vector)
    return np.arccos(np.dot(u_vector, norm))

def rot_matrix(omega, inc):
    '''Defines the rotation matrix as in Ward 1974 eq. 11'''
    r_matrix = np.zeros((3,3))
    r_matrix[0][0] = np.cos(omega)
    r_matrix[0][1] = np.sin(omega)
    r_matrix[0][2] = 0
    r_matrix[1][0] = -np.cos(inc) * np.sin(omega)
    r_matrix[1][1] = np.cos(inc) * np.cos(omega)
    r_matrix[1][2] = np.sin(inc)
    r_matrix[2][0] = np.sin(inc) * np.sin(omega)
    r_matrix[2][1] = -np.sin(inc) * np.cos(omega)
    r_matrix[2][2] = np.cos(inc)

    return r_matrix

def main():
    alphas = np.arange(0.05, 6, 0.05) 
    run_name = '210706h'    
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith('rn='):
            run_name = arg[3:]

    times = np.load(run_name + '_times.npz')['arr_0'] / (2 * np.pi)
    ur = np.load(run_name + '_ur.npz')['arr_0']
    incs = ur[:,0]
    omegas = ur[:,1]
    eccs = ur[:,2]
    varpis = ur[:,3]

    ifunc = interp1d(times, incs, bounds_error=False, fill_value="extrapolate")
    ofunc = interp1d(times, omegas, bounds_error=False, fill_value="extrapolate")
    hfunc = interp1d(times, eccs * np.sin(varpis), bounds_error=False, fill_value="extrapolate")
    kfunc = interp1d(times, eccs * np.cos(varpis), bounds_error=False, fill_value="extrapolate")
    pfunc = interp1d(times, incs * np.sin(omegas), bounds_error=False, fill_value="extrapolate")
    qfunc = interp1d(times, incs * np.cos(omegas), bounds_error=False, fill_value="extrapolate")
    
    print('Interpolation Complete')

    angle = np.radians(2.5)
    initial_condition = [0, np.sin(angle), np.cos(angle)]  
    
    print('Begin Ward EOM')
    grid = np.zeros((len(alphas), 2))
    for i in range(len(alphas)):
        s = odeint(ward, initial_condition, times, args=(alphas[i], pfunc, qfunc, hfunc, kfunc))
        
        rotated = np.zeros(s.shape)
        for j, t in enumerate(times):
            vector = s[j]
            rotation_matrix = rot_matrix(ofunc(t), ifunc(t))
            rotated[j] = np.matmul(rotation_matrix, vector) 

        angles = np.zeros(len(rotated))
        for k, spin_vector in enumerate(rotated):
            angles[k] = obliquity(spin_vector, [0, 0, 1])

        grid[i][0] = alphas[i]
        grid[i][1] = np.max(np.degrees(angles))  
    
    np.savetxt('ward_' + run_name + '.txt', grid)

if __name__ == "__main__":
        main()
