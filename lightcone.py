'''
Class to host lightcone loaded from data
or generated using an Nbody class
'''

class LightCone:
    '''
    Lightcone class to store RA, DEC, z and compute multipole moments 
    '''

    def __init__(self, Redshift, CosmoParams=None):
        return

    def LoadGalaxies(self, SimDir, SimType):
        if SimType=='Magneticum':
            halos = np.loadtxt(SimDir)
        else:
            raise Exception("Simulation type not implemented")

        return

    def LoadCMBMap(self, SimDir, SimType):
        return

    def CalculateWindowFunction(self):
        return

    def DeconvolveWindowFunction(self):
        return
    
    def PaintMeshes(self):
        return
    
    def CalculateMultipoles(self):
        return
