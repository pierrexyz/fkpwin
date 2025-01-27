from fkpwin.module import *
from fftlog.fftlog import FFTLog
from fftlog.sbt import SBT, MPC

class WindowMatrix(): 
    def __init__(self, s=None, k=None, dk=None, kedges=None, ells=[0, 2, 4], NFFT=1024 * 8, smin=1., smax=1e5):
        
        self.ells = ells

        self.fftsettings = dict(Nmax=NFFT, xmin=smin, xmax=smax, bias=-1.6, window=.2) 
        self.fft = FFTLog(**self.fftsettings)

        if s is None: s = self.fft.x # s = np.geomspace(1e-4, 1e5, 1024*16); s = s[s > smin]
        if k is None: 
            dk = 1e-3
            k = np.arange(1e-4, .5, dk)
            dk = array(len(k) * [dk])
        self.s, self.k, self.dk, self.kedges = s, k, dk, kedges
        
        self.fft.mode = 'exact' if array_equal(self.s, self.fft.x) else 'interp' 
        self.pPow = exp(einsum('n,s->ns', -self.fft.Pow-3., log(self.k)))
        self.M = 4 * pi * array([[MPC(ell, p) for p in -0.5*self.fft.Pow] for ell in self.ells]) 
        self.ClmL = array([[[(2*l+1) * float(wigner_3j(l, m, L, 0, 0, 0)**2) for L in self.ells] for m in self.ells] for l in self.ells])
        ks = self.k[..., newaxis] * self.s[newaxis, ...]
        self.jlks = array([spherical_jn(ell, ks) for ell in self.ells])
        self.signs = array([[(-1j)**l * 1j**m for m in self.ells] for l in self.ells])

        if self.kedges is not None: 
            self.points = array([linspace(kbinmin, kbinmax, 30) for (kbinmin, kbinmax) in zip(self.kedges[:-1], self.kedges[1:])])
            self.binvol = array([quad(lambda k: k**2, kbinmin, kbinmax)[0] for (kbinmin, kbinmax) in zip(self.kedges[:-1], self.kedges[1:])])

        self.set_f2c()

    def set_f2c(self, NFFT=1024, kmin=1e-5, kmax=1e3): 
        self.sbt = SBT(ells=self.ells)
        self.sbt.set_f2c(self.s, kmin=kmin, kmax=kmax, bias=-2.1, NFFT=NFFT, extrap='padding') 
        return 

    def qs_from_qk(self, qk, k=None):
        kk = self.k if k is None else k
        qs = self.sbt.get_transform(kk, qk, sum_ell=True)
        qs /= qs[0,0]
        return qs

    def i(self, f, x):
        return interp1d(x, f, axis=-1, kind='cubic', bounds_error=False, fill_value=0.)

    def set_qs(self, qs, s=None):
        self.qs = qs if s is None else self.i(qs, s)(self.s)
        return 

    def set_qk(self, qk, k=None):
        self.qk = qk if k is None else self.i(qk, k)(self.k) 
        if self.qk.ndim == 1: self.qk = self.qk.reshape(1,-1) # if we provide only ell = 0, put self.qk in the correct (l,k)-shape
        return

    def set(self, qk, k=None, qs=None, s=None):
        self.set_qk(qk, k=k)
        self.set_qs(self.qs_from_qk(qk, k=k)) if qs is None else self.set_qs(qs, s=s)
        return 

    def compute(self, ic=True, binning=True, measure=True):
        qlm = einsum('lmL,Ls->lms', self.ClmL, self.qs)
        qj = einsum('lms,lks->lmks', qlm, self.jlks)
        coef = self.fft.Coef(self.s, qj, mode=self.fft.mode, extrap ='padding') 
        self.wlm = einsum('lmkn,np,mn->lmkp', coef, self.pPow, self.M)
        self.wlm = einsum('lm,lmkp->lmkp', self.signs,self. wlm)
        if ic: self.wlm = self.ic(self.wlm) # integral constraints
        if binning: self.wlm = self.bin(self.wlm) # binning
        if measure: self.wlm = einsum("p,lmkp->lmkp", self.k**2, self.wlm)
        return 

    def ic(self, wlm, only_ell0=True):
        wlm_ic = 1. * wlm 
        if only_ell0: wlm_ic[0] -= einsum('k,mp->mkp', self.qk[0], wlm[0,:,0,:]) # by default, we do only ell=0 for numerical stability 
        else: wlm_ic -= einsum('lk,mp->lmkp', self.qk, wlm[0,:,0,:]) # the IC is anyway negligible for ell > 0
        return wlm_ic

    def bin(self, wlm): 
        iwlm = self.i(wlm, self.k) # wlm(k,p) = wlm(p,k); here interpolating along axis = -1 corresponds to interpolating along k
        wlm_bin = array([trapz(einsum('lmpk,k->lmpk', iwlm(pts), pts**2), x=pts, axis=-1) for pts in self.points]) 
        return np.einsum('k,klmp->lmkp', 1/self.binvol, wlm_bin)

    def get(self, compute=True):
        self.compute()
        return self.wlm

    def save_to(self, filename): 
        to_save = {'ells': self.ells, 'kedges': self.kedges, 'p': self.k, 'wlmkp': real(self.wlm), 'wlmkp_dp': einsum('lmkp,p->lmkp', real(self.wlm), self.dk)} 
        save(filename, to_save) 
        return
    

    