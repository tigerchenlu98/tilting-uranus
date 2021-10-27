import numpy as np
import rebound
import reboundx
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from tqdm import tqdm
import time
import sys
import matplotlib.pyplot as plt

save_data = 1 # SAVE DATA

def invariant_plane(sim):
    '''
    Calculates the invariant plane of a system
    We pass in a sim in the ecliptic - returns one in the invariable plane
    '''
    ps = sim.particles
    l_tot = np.zeros(3)
    
    # This calculates the invariant plane vector
    for i, particle in enumerate(ps):
        momentum = particle.m * np.array([particle.x, particle.y, particle.z])
        velocity = np.array([particle.vx, particle.vy, particle.vz])
        
        # L = r x p
        l_particle = np.cross(momentum, velocity)
        l_tot += l_particle
    
    uv_tot = l_tot / np.linalg.norm(l_tot)
    
    # Now, let us re-set up the sim in the invariant frame
    sim_new = rebound.Simulation()
    sim_new.add("Sun")
    for p in ps[1:]:
        # Calculate the unit vector of momentum
        p_p = p.m * np.array([p.x, p.y, p.z])
        p_v = np.array([p.vx, p.vy, p.vz])
        l_p = np.cross(p_p, p_v)
        uv_p = l_p / np.linalg.norm(l_p)
        
        # Calculate the inclination between each planet and the invariable plane
        p_inc = np.arccos(np.dot(uv_p, uv_tot))
        
        # Add each planet to the new sim, with new inclination
        sim_new.add(m = p.m, a = p.a, e = p.e, inc = p_inc, Omega = p.Omega, pomega = p.pomega)
    
    return sim_new

def main():

    # Planet 9 parameters
    m9 = 5 # units of Earth Mass
    a9 = 40 # AU
    e9 = 0.3
    i9 = np.radians(1) # in radians
    run_name = '210717a'
    tmax = 1e8    
    tau_a = (tmax * 2 * np.pi) / np.log(460 / 40) # migration parameter, set so two e-folds

    # Overwrite with arguments from the command line if needed
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith('m='):
            m9 = float(arg[2:])
        elif arg.startswith('a='):
            a9 = float(arg[2:])
        elif arg.startswith('e='):
            e9 = float(arg[2:])
        elif arg.startswith('i='):
            i9 = np.radians(float(arg[2:]))
        elif arg.startswith('rn='):
            run_name = arg[3:]
        #elif arg.startswith('tau='):
        #    tau_a = float(arg[4:])
        elif arg.startswith('tmax='):
            tmax = float(arg[5:])
    
    tau_a = (tmax * 2 * np.pi) / np.log(460 / 40) # Reset tau_a after new tmax
    
    # Run the rebound simulation
    reb_times = np.arange(0, tmax, 100) * (2 * np.pi)# Rebound times units are 1 yr = 2 pi
    
    # Set up ecliptic simulation
    sim_setup = rebound.Simulation()
    sim_setup.add("Sun")
    sim_setup.add(m=0.0009548, a=5.2035, e=0.0485, inc=0.02275, Omega = 1.7543, pomega = 0.2427) # Jupiter
    sim_setup.add(m=0.00028588, a=9.5407, e=0.05497, inc=0.04342, Omega=1.9827, pomega=1.612) # Saturn
    sim_setup.add(m=4.366e-5, a=19.19, e=0.04723, inc=0.0135, Omega=1.292, pomega=2.9834) # Uranus
    sim_setup.add(m=5.151e-5, a=30.075, e=0.0087, inc=0.0309, Omega=2.3, pomega=0.7728) # Neptune
    sim_setup.add(m=m9 * 3e-6, a=a9, e=e9, inc=i9, Omega=0.017, pomega=4.311) # p9
    print(m9, a9, e9, np.degrees(i9), tau_a / (2 * np.pi), run_name)
    # move to the invariant plane
    sim = invariant_plane(sim_setup)
    sim.move_to_com()

    # Set up migration force
    ps = sim.particles
    rebx = reboundx.Extras(sim)
    mof = rebx.load_force("modify_orbits_forces")
    rebx.add_force(mof)
    sim.particles[5].params["tau_a"] = tau_a

    # Stochastic timesteps
    Nout = len(reb_times)
    ur = np.zeros((Nout, 4))
    nep = np.zeros((Nout, 4))
    p9 = np.zeros((Nout, 6))
    pas = np.zeros((Nout, 4))
    print('rebound start')                                                                                                                                                                                                                                                         
    for i, time_step in tqdm(enumerate(reb_times)):
        sim.integrate(time_step)
        
        planets = sim.particles

        ur[i] = sim.particles[3].inc, sim.particles[3].Omega, sim.particles[3].e, sim.particles[3].pomega
        nep[i] = sim.particles[4].inc, sim.particles[4].Omega, sim.particles[4].e, sim.particles[4].pomega
        p9[i] = sim.particles[5].inc, sim.particles[5].a, sim.particles[5].m, sim.particles[5].e, sim.particles[5].Omega, sim.particles[5].pomega
        pas[i] = sim.particles[1].a, sim.particles[2].a, sim.particles[3].a, sim.particles[4].a

    print('inc, ecc: ', np.degrees(sim.particles[5].inc), sim.particles[5].e)
    if save_data == 1:
        np.savez(run_name + '_times.npz', reb_times) # in units of 2pi 
        np.savez(run_name + '_ur.npz', ur)
        np.savez(run_name + '_nep.npz', nep)
        np.savez(run_name + '_p9.npz', p9)
        np.savez(run_name + '_planets.npz', pas)

        incs = ur[:,0]
        omegas = ur[:,1]
        eccs = ur[:,2]
        varpis = ur[:,3]

if __name__ == "__main__":
    main()
