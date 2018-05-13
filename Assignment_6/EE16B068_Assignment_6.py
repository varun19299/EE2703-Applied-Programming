import argparse as ap
import numpy as np
import matplotlib.pyplot as plt

class Simulate_tubelight(object):
    '''Simulate a tubelight'''
    def __init__(self,n=100,M=5,nk=1000,u0=7,p=0.5,Msig=0.2):
        self.create_universe(n,M,nk,u0,p,Msig)
        self.create_vectors()

    def create_universe(self,n=100,M=5,nk=1000,u0=7,p=0.5,Msig=0.2):
        self.n=n  # spatial grid size.
        self.M=M    # number of electrons injected per turn.
        self.nk=nk # number of turns to simulate.
        self.u0=u0  # threshold velocity.
        self.p=p # probability that ionization will occur
        self.Msig=Msig

    def create_vectors(self):
        dim=self.n*self.M
        self.xx=np.zeros(dim) # Electron position
        self.u=np.zeros(dim) # Electron velocity
        self.dx=np.zeros(dim) # Displacement in current turn
        self.I=[] # Intensity of emitted light,
        self.X=[] # Electron position
        self.V=[] # Electron velocity

    def loop(self):
        # Find the electrons present in the chamber.
        self.ii=np.where(self.xx>0)[0]

        # Compute the displacement during this turn
        self.dx[self.ii]=self.u[self.ii]+0.5

        # Advance the electron position and velocity for the turn.
        self.xx[self.ii]+=self.dx[self.ii]
        self.u[self.ii]+=1

        # Determine which particles have hit the anode
        self.hit_anode=np.where(self.xx>self.n)[0]
        self.xx[self.hit_anode]=0
        self.u[self.hit_anode]=0
        self.dx[self.hit_anode]=0

        # Find those electrons whose velocity is greater than or equal to the threshold.
        self.kk=np.where(self.u>=self.u0)[0]
        # Of these, which electrons are ionized
        self.ll=np.where(np.random.rand(len(self.kk))<=self.p)[0]
        self.kl=self.kk[self.ll]

        # Reset the velocities of these electrons to zero (they suffered an inelastic collision)
        self.u[self.kl]=0
        # The collision could have occurred at any point between the previous xi and the current xi
        self.xx[self.kl]-=self.dx[self.kl]*np.random.rand()

        # Excited atoms at this location resulted in emission from that point.
        self.I.extend(self.xx[self.kl].tolist())

        # Inject M new electrons
        m= int(np.random.randn()*self.Msig+self.M)  # actual number of electrons injected
        # Add them to unused slots. Adding randomly
        self.slots_to_add_to=np.where(self.xx==0)[0]

        if len(self.slots_to_add_to)>=m:
            random_start=np.random.randint(len(self.slots_to_add_to))
            self.xx[self.slots_to_add_to[random_start:m+random_start]]=1
            self.u[self.slots_to_add_to[random_start-m:random_start]]=0
        else: # If no free slots
            self.xx[self.slots_to_add_to]=1
            self.u[self.slots_to_add_to]=0

        self.existing_electrons=np.where(self.xx>0)[0]
        self.X.extend(self.xx[self.existing_electrons].tolist())
        self.V.extend(self.u[self.existing_electrons].tolist())

    def run_loop(self):
        [self.loop() for i in range(self.nk)]

    def plot_intensity(self):
        plt.hist(self.I,bins=np.arange(1,100),ec='black',alpha=0.5)
        plt.title("Light Intensity Histogram ")
        plt.show()

    def plot_electron_density(self):
        plt.hist(self.X,bins=np.arange(1,100),ec='black',alpha=0.5)
        plt.title("Electron Density Histogram")
        plt.show()

    def plot_intensity_table(self):
        import pandas
        a,bins,c=plt.hist(self.I,bins=np.arange(1,100),ec='black',alpha=0.5)
        xpos=0.5*(bins[0:-1]+bins[1:])
        d={'Position':xpos,'Count':a}
        p=pandas.DataFrame(data=d)
        print(p)

    def plot_electron_phase_space(self):
        plt.plot(self.xx,self.u,'x')
        plt.title("Electron Phase Space")
        plt.xlabel("Position")
        plt.figsize=(15,10)
        plt.ylabel("Velocity")
        plt.show()



if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('-n', "--grid_size", help="Grid Size",default=100)
    parser.add_argument('-M','--no_of_electrons', help="Downscale ratio", default=5)
    parser.add_argument('-nk', '--turns', help="Number of iterations",default=500)
    parser.add_argument('-u0', '--threshold_velocity', help="Threshold Velocity for electron",default=7)
    parser.add_argument('-p', '--probability', help="Probability of collision",default=0.5)
    parser.add_argument('-Msig', '--variance', help="Variance of probability distribution ",default=0.2)
    args = vars(parser.parse_args())
    n,M,nk,u0,p,Msig=(args['grid_size'],args['no_of_electrons'],args['turns'],args['threshold_velocity'],\
                     args['probability'],args['variance'])
    a=Simulate_tubelight(n,M,nk,u0,p,Msig)
    a.run_loop()
    a.plot_electron_density()
    a.plot_electron_phase_space()
    a.plot_intensity()
    a.plot_intensity_table()
