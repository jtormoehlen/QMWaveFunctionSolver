from matplotlib import animation as anim
from matplotlib import pyplot as plt


class Animator:
    def __init__(self, wave_packet):
        self.time = 0.
        self.wave_packet = wave_packet
        self.fig, self.ax = plt.subplots()
        plt.plot(self.wave_packet.x, self.wave_packet.potential, '--k')

        self.time_text = self.ax.text(0.05, 0.95, '', horizontalalignment='left',
                                      verticalalignment='top', transform=self.ax.transAxes)
        self.prob_text = self.ax.text(0.05, 0.90, '', horizontalalignment='left',
                                      verticalalignment='top', transform=self.ax.transAxes)
        self.wave_packet_line, = self.ax.plot(self.wave_packet.x, self.wave_packet.evolve())
        self.ax.set_xlim(self.wave_packet.x_begin, self.wave_packet.x_end)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_xlabel(r'$x$')
        self.ax.set_ylabel(r'$|\psi(x,t)|^2$')

    def update(self, data):
        self.wave_packet_line.set_ydata(data)
        return self.wave_packet_line

    def time_step(self):
        while True:
            self.time += self.wave_packet.dt
            self.time_text.set_text('Zeit: ' + str(self.time))
            if self.time % 10 == 0:
                self.prob_text.set_text('Wahrscheinlichkeit: ' + str(self.wave_packet.norm))
            yield self.wave_packet.evolve()

    def animate(self):
        self.ani = anim.FuncAnimation(self.fig, self.update, self.time_step, interval=5, blit=False)
