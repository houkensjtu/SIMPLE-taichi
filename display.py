import taichi as ti
import numpy as np

@ti.data_oriented
class Display:
    def __init__(self, SIMPLESolver, *args):
        self.solver = SIMPLESolver
        self.nx = self.solver.nx
        self.ny = self.solver.ny
        self.real = self.solver.real
        self.udisp = ti.field(dtype=self.solver.real, shape=((self.nx+2), (self.ny+2)))
        self.vdisp = ti.field(dtype=self.solver.real, shape=((self.nx+2), (self.ny+2)))
        self.pdisp = ti.field(dtype=self.solver.real, shape=((self.nx+2), (self.ny+2)))
        self.pcordisp = ti.field(dtype=self.solver.real, shape=((self.nx+2), (self.ny+2)))
        self.mdivdisp = ti.field(dtype=self.solver.real, shape=((self.nx+2), (self.ny+2)))
        self.gui = ti.GUI("SIMPLESolver", ((self.nx+2),5*(self.ny+2)))        

    @ti.func
    def scale_field(self, f):
        f_max = -1.0e9
        f_min = 1.0e9
        for i,j in f:
            if f[i,j] > f_max:
                f_max = f[i,j]
            if f[i,j] < f_min:
                f_min = f[i,j]
        for i,j in f:
            f[i,j] = (f[i,j] - f_min) / (f_max - f_min + 1.0e-9)
        
    @ti.kernel
    def post_process_field(self):
        for i,j in ti.ndrange(self.nx+2, self.ny+2):
            self.udisp[i,j] = 0.5 * (self.solver.u[i,j] + self.solver.u[i+1,j])
            self.vdisp[i,j] = 0.5 * (self.solver.v[i,j] + self.solver.v[i,j+1])
            self.pdisp[i,j] = self.solver.p[i,j]
            self.pcordisp[i,j] = self.solver.pcor[i,j]
            self.mdivdisp[i,j] = self.solver.mdiv[i,j]            
        self.scale_field(self.udisp)
        self.scale_field(self.vdisp)
        self.scale_field(self.pdisp)
        self.scale_field(self.pcordisp)
        self.scale_field(self.mdivdisp)        
            
    def ti_gui_display(self, filename, show_gui=False):
        import numpy as np        
        self.post_process_field()
        img = np.concatenate((self.udisp.to_numpy(), self.vdisp.to_numpy(), self.pdisp.to_numpy(), \
                              self.pcordisp.to_numpy(), self.mdivdisp.to_numpy()), axis=1)
        self.gui.set_image(img)
        if show_gui:
            self.gui.show()
        else:
            self.gui.show(filename)

    def matplt_display_init(self):
        pass

    def dump_field(self, step, msg): # Save u,v,p at step to csv files
        for name,val in {'u':self.solver.u, 'v':self.solver.v, 'p':self.solver.p, \
                         'mdiv':self.solver.mdiv, 'pcor':self.solver.pcor}.items():
            np.savetxt(f'log/{step:06}-{name}-{msg}.csv', val.to_numpy(), delimiter=',')

    def dump_coef(self, step, msg):
        np.savetxt(f'log/{step:06}-apu-{msg}.csv', self.coef_u.to_numpy()[:,:,0], delimiter=',')
        np.savetxt(f'log/{step:06}-awu-{msg}.csv', self.coef_u.to_numpy()[:,:,1], delimiter=',')
        np.savetxt(f'log/{step:06}-aeu-{msg}.csv', self.coef_u.to_numpy()[:,:,2], delimiter=',')
        np.savetxt(f'log/{step:06}-anu-{msg}.csv', self.coef_u.to_numpy()[:,:,3], delimiter=',')
        np.savetxt(f'log/{step:06}-asu-{msg}.csv', self.coef_u.to_numpy()[:,:,4], delimiter=',')
        np.savetxt(f'log/{step:06}-bu -{msg}.csv', self.b_u.to_numpy(),           delimiter=',')
        
        np.savetxt(f'log/{step:06}-apv-{msg}.csv', self.coef_v.to_numpy()[:,:,0], delimiter=',')
        np.savetxt(f'log/{step:06}-awv-{msg}.csv', self.coef_v.to_numpy()[:,:,1], delimiter=',')
        np.savetxt(f'log/{step:06}-aev-{msg}.csv', self.coef_v.to_numpy()[:,:,2], delimiter=',')
        np.savetxt(f'log/{step:06}-anv-{msg}.csv', self.coef_v.to_numpy()[:,:,3], delimiter=',')
        np.savetxt(f'log/{step:06}-asv-{msg}.csv', self.coef_v.to_numpy()[:,:,4], delimiter=',')        
        np.savetxt(f'log/{step:06}-bv -{msg}.csv', self.b_v.to_numpy(),           delimiter=',')

        np.savetxt(f'log/{step:06}-app-{msg}.csv', self.coef_p.to_numpy()[:,:,0], delimiter=',')
        np.savetxt(f'log/{step:06}-awp-{msg}.csv', self.coef_p.to_numpy()[:,:,1], delimiter=',')
        np.savetxt(f'log/{step:06}-aep-{msg}.csv', self.coef_p.to_numpy()[:,:,2], delimiter=',')
        np.savetxt(f'log/{step:06}-anp-{msg}.csv', self.coef_p.to_numpy()[:,:,3], delimiter=',')
        np.savetxt(f'log/{step:06}-asp-{msg}.csv', self.coef_p.to_numpy()[:,:,4], delimiter=',')        
        np.savetxt(f'log/{step:06}-bp -{msg}.csv', self.b_p.to_numpy(),           delimiter=',')        
    
