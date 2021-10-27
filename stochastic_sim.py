import rebound
import reboundx
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import sys

# BOOLEANS
load_rebound = 0 # perform rebound simulation if 0, load in previous data if 1
save_data = 1 # SAVE DATA

# From Batygin et al 2021
def chi(a, a_p):
    '''
    Arguments are semimajor axis of the body and planet, in AU

    '''
    return np.rint(2 * (a / a_p)**(3/2))

def delta_a(planet, p9):
    '''
    Arguments are the planet in question and Planet 9, both as REBOUND particles
    '''
    a_p, m_p = planet.a, planet.m # in AU and solar masses
    a, q = p9.a, p9.a * (1 - p9.e)
    chi_val = chi(a, a_p)

    return 4 * a_p * np.sqrt(2 * chi_val * m_p / 5) * np.exp(-(q / (2 * a_p))**2) # in AU

def lyapunov(p9):
    '''
    Takes a REBOUND particle (for p9) as an argument
    '''
    a = p9.a

    return np.sqrt(4 * np.pi**2 * a**3) # In REBOUND time units

def diffusion(planet, p9):
    a_p, m_p = planet.a, planet.m # mass in solar masses
    q = p9.a * (1 - p9.e)
    
    return (8 / (5 * np.pi)) * m_p * np.sqrt(a_p) * np.exp(-0.5 * (q / a_p)**2)

def kick(planet, p9):
    kick_mag = diffusion(planet, p9)
    stochastic_kick = np.random.normal(kick_mag, 0.5 * kick_mag)
    return stochastic_kick

def part_at_perihelion(p9):
    '''
    Checks Planet 9 true anomaly to verify if it is at perihelion - needs to be kicked
    '''
    
    f = p9.f
    
    # within 5 percent of perihelion
    if f < 0.05 * 2 * np.pi or f > 0.95 * 2 * np.pi:
        return True
    else:
        return False

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
    sim_new.add(m=1.0, x=-0.008197241482825336, y=0.004151193016612189, z=0.000157692530259119, vx=-0.00026437163520439327, vy=-0.00046513151406683264, vz=9.990395060541673e-06)
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
    m9 = 6.9 # units of Earth Mass
    a9 = 40 # AU
    p9 = 25 # AU
    e9 = 1 - (p9 / a9)
    i9 = np.radians(15) # in radians
    date = '2110012_'
    run_name = date
    tmax = 1e7 * 2 * np.pi
    kick_time = 0.01 * 2 * np.pi
    # tau_a = (tmax * 2 * np.pi) / np.log(460 / 100) # migration parameter, set so two e-folds.  Must be in rebound time units
    

    # Overwrite with arguments from the command line if needed
    args = sys.argv[1:]
    for arg in args:
        if arg.startswith('m='):
            m9 = float(arg[2:])
        elif arg.startswith('a='):
            a9 = float(arg[2:])
            e9 = 1 - (p9 / a9)
        elif arg.startswith('e='):
            e9 = float(arg[2:])
        elif arg.startswith('p='):
            p9 = float(arg[2:])
        elif arg.startswith('i='):
            i9 = np.radians(float(arg[2:]))
        elif arg.startswith('rn='):
            run_name = arg[3:]
        elif arg.startswith('tmax='):
            tmax = float(arg[5:])
        elif arg.startswith('kt='):
            kick_time = float(arg[3:]) * 2 * np.pi
    
    kick_time *= ((3e7 * 2 * np.pi) / tmax)

    # Run the rebound simulation
    KICKED_THIS_ORBIT = 0 
    tstep = 10 * 2 * np.pi
    times = np.arange(0, tmax, tstep)
 
    sim_setup = rebound.Simulation()
    sim_setup.add(m=1.0, x=-0.008197241482825336, y=0.004151193016612189, z=0.000157692530259119, vx=-0.00026437163520439327, vy=-0.00046513151406683264, vz=9.990395060541673e-06) # Sun
    sim_setup.add(m=0.0009548, a=5.2035, e=0.0485, inc=0.02275, Omega = 1.7543, pomega = 0.2427) # Jupiter
    sim_setup.add(m=0.00028588, a=9.5407, e=0.05497, inc=0.04342, Omega=1.9827, pomega=1.612) # Saturn
    sim_setup.add(m=4.366e-5, a=19.19, e=0.04723, inc=0.0135, Omega=1.292, pomega=2.9834) # Neptune
    sim_setup.add(m=5.151e-5, a=30.075, e=0.0087, inc=0.0309, Omega=2.3, pomega=0.7728) # Uranus
    sim_setup.add(m=m9 * 3e-6, a = a9, e=e9, inc=np.radians(15))
    
    sim = invariant_plane(sim_setup)
    sim.move_to_com()
    ps = sim.particles

    Nout = len(times)
    ur = np.zeros((Nout, 4))
    nep = np.zeros((Nout, 4))
    p9 = np.zeros((Nout, 6))
    pas = np.zeros((Nout, 4))
    print('rebound start')
    previous_f = ps[5].f
    for i, time_step in enumerate(times):
        sim.integrate(time_step)

        ps = sim.particles
        if ps[5].f < previous_f:
            KICKED_THIS_ORBIT = 0
        
        if KICKED_THIS_ORBIT == 0 and part_at_perihelion(ps[5]):
            # get unit velocity vector of Planet 9
            vel_vector = [ps[5].vx, ps[5].vy, ps[5].vz]
            vel_vec_norm = vel_vector / np.linalg.norm(vel_vector)
            
            # Add a kick
            sto_kick = kick(ps[2], ps[5]) * vel_vec_norm
            
            ps[5].vx += (sto_kick[0] * kick_time)
            ps[5].vy += (sto_kick[1] * kick_time)
            ps[5].vz += (sto_kick[2] * kick_time)
            
            KICKED_THIS_ORBIT = 1

        ur[i] = sim.particles[3].inc, sim.particles[3].Omega, sim.particles[3].e, sim.particles[3].pomega
        nep[i] = sim.particles[4].inc, sim.particles[4].Omega, sim.particles[4].e, sim.particles[4].pomega
        p9[i] = sim.particles[5].inc, sim.particles[5].a, sim.particles[5].m, sim.particles[5].e, sim.particles[5].Omega, sim.particles[5].pomega
        pas[i] = sim.particles[1].a, sim.particles[2].a, sim.particles[3].a, sim.particles[4].a
        
        if ps[5].f < previous_f:
            KICKED_THIS_ORBIT = 0
        
        previous_f = ps[5].f

        print(ps[1].a, ps[2].a, ps[3].a, ps[4].a, ps[5].a)
        print('Final inc, ecc: ', np.degrees(sim.particles[5].inc), sim.particles[5].e)
        if save_data == 1:
            np.savez(run_name + '_times.npz', np.array(times)) # in units of 2pi
            np.savez(run_name + '_ur.npz', np.array(ur))
            np.savez(run_name + '_nep.npz', np.array(nep))
            np.savez(run_name + '_p9.npz', np.array(p9))
            np.savez(run_name + '_planets.npz', np.array(pas))

if __name__ == "__main__":
        main()     
