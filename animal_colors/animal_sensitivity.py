import numpy as np
import matplotlib.pyplot as plt

__all__ = ['Sensitivity']


class Sensitivity(object):

    def __init__(self, animal):
        """
        Sets the sensitivity scaling for different animals.
        Sensitivity scalings are approximated as Gaussians.

        Parameters
        ----------
        animal : str
           The name of the animal you want to imitate. Current
           options are: human, blue tit, turkey, honeybee, pigeon,
           and house fly.
        """
        self.animal = animal
        self.wave_x = np.linspace(300,700,1000)
        self.red_lim = 650
        self.blue_lim = 500

        if animal.lower() == 'human':
            self.human()

        elif animal.lower() == 'pigeon':
            self.pigeon()
            
        elif animal.lower() == 'honeybee':
            self.honeybee()

        elif animal.lower() == 'blue tit':
            self.bluetit()

        elif animal.lower() == 'turkey':
            self.turkey()

        elif animal.lower() == 'house fly':
            self.housefly()

        else:
            raise ValueError('Animal not implemented yet.')

        self.set_contributions()

    
    def pdf(self, x, mu, std):
        """
        Creates Gaussian distribution for given colors.
        
        Parameters
        ----------
        x : float or np.ndarray
        mu : float
           Mean value.
        std : float
           Std value.
        """
        fact = np.sqrt(2 * np.pi * std**2)
        exp  = np.exp(-0.5 * ( (x-mu) / std)**2)
        return 1.0/fact * exp


    def set_contributions(self):
        """
        Makes sure the appropriate wavelengths are contributing
        to the color map (e.g. removes red when the sensitivity
        function doesn't extend into red wavelengths).
        """
        reset = np.zeros(self.mapped.shape)
        r = np.where(self.wave_x>=self.red_lim)[0]
        b = np.where(self.wave_x<=self.blue_lim)[0]
        g = np.where( (self.wave_x<self.red_lim) &
                      (self.wave_x>self.blue_lim) )[0]

        tot = np.nansum(self.mapped, axis=1)
        tot /= np.nanmax(tot)

        reset[:,0][r] = self.mapped[:,0][r]
        reset[:,1][g] = self.mapped[:,1][g]
        reset[:,2][b] = self.mapped[:,2][b]

        self.total_map = reset


    def plot(self):
        """
        Plots sensitivity functions.
        """
        for i in range(self.mapped.shape[1]):
            plt.plot(self.wave_x, self.mapped[:,i], lw=4, label='Cone {}'.format(i))

        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), 
                   loc='lower left',
                   ncol=self.mapped.shape[1], mode="expand", 
                   borderaxespad=0.)

        plt.xlabel('wavelength [nm]', fontsize=16)
        plt.ylabel('sensitivity', fontsize=16)
        plt.show()

    
    def human(self):
        """
        Creates sensitivity distribution for humans.
        """
        human_blue = self.pdf(self.wave_x, 420.0, 40.0)
        human_blue /= np.nanmax(human_blue)

        human_red = self.pdf(self.wave_x, 590, 50)
        human_red /= np.nanmax(human_red)
        
        human_green = self.pdf(self.wave_x, 550, 50)
        human_green /= np.nanmax(human_green)

        self.mapped = np.array([human_red, human_green, human_blue]).T
        

    def pigeon(self):
        """
        Creates sensitivity distribution for pigeons.
        """
        bird_blue = self.pdf(self.wave_x, 490.0, 20.0)
        bird_blue /= np.nanmax(bird_blue)

        bird_ultra_blue = self.pdf(self.wave_x, 400, 40)
        bird_ultra_blue /= np.nanmax(bird_ultra_blue)
        
        bird_blue = (bird_blue+bird_ultra_blue)/np.nanmax(bird_blue+bird_ultra_blue)

        bird_green = self.pdf(self.wave_x, 550, 20)
        bird_green /= np.nanmax(bird_green)

        bird_red = self.pdf(self.wave_x, 630, 20)
        bird_red /= np.nanmax(bird_red)

        self.mapped = np.array([bird_red, bird_green, bird_blue]).T


    def honeybee(self):
        """
        Creates sensitivity distribution for honeybees.
        """
        hb_blue = self.pdf(self.wave_x, 350.0, 30.0)
        hb_blue /= np.nanmax(hb_blue)

        hb_red = self.pdf(self.wave_x, 550, 40)
        hb_red /= np.nanmax(hb_red)
        
        hb_red_lower = self.pdf(self.wave_x, 400, 60.) * 30
        
        red = (hb_red+hb_red_lower)/np.nanmax(hb_red+hb_red_lower)
        
        hb_green = self.pdf(self.wave_x, 450, 30)
        hb_green /= np.nanmax(hb_green)
        
        hb_green_lower = self.pdf(self.wave_x, 370, 30) * 30
        green = (hb_green+hb_green_lower)/np.nanmax(hb_green+hb_green_lower)

        self.mapped = np.array([red, green, hb_blue]).T

    
    def bluetit(self):
        """
        Creates sensitivity distribution for the blue tit.
        """
        red = self.pdf(self.wave_x, 580, 40)
        red /= np.nanmax(red)

        green = self.pdf(self.wave_x, 500, 40)
        green /= np.nanmax(green)
        
        blue = self.pdf(self.wave_x, 420, 30)
        blue /= np.nanmax(blue)
        
        ultra = self.pdf(self.wave_x, 340, 30)
        ultra /= np.nanmax(ultra)
        
        blue = (blue+ultra)/np.nanmax(blue+ultra)

        self.mapped = np.array([red, green, blue]).T

    
    def turkey(self):
        """
        Creates sensitivity distribution for the turkey.
        """
        red = self.pdf(self.wave_x, 590, 40)
        red /= np.nanmax(red)
        
        green = self.pdf(self.wave_x, 530, 40)
        green /= np.nanmax(green)
        
        blue = self.pdf(self.wave_x, 470, 30)
        blue /= np.nanmax(blue)
        
        ultra = self.pdf(self.wave_x, 410, 30)
        ultra /= np.nanmax(ultra)
        
        blue = (blue+ultra)/np.nanmax(blue+ultra)

        self.mapped = np.array([red, green, blue]).T


    def housefly(self):
        """
        Creates sensitivity distribution for the house fly.
        """
        red = self.pdf(self.wave_x, 590, 20)
        red /= np.nanmax(red)

        green = self.pdf(self.wave_x, 500, 40)
        green /= np.nanmax(green)

        subgreen = self.pdf(self.wave_x, 410, 60)
        subgreen /= (np.nanmax(subgreen)*2)

        green = (green+subgreen)/np.nanmax(green+subgreen)

        blue = self.pdf(self.wave_x, 360, 30)
        blue /= np.nanmax(blue)

        self.mapped = np.array([red, green, blue]).T
