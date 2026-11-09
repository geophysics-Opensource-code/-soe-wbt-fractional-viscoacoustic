function Dm=FDMx_fft2d(p,M,N,LX,LZ,f,axis_class)
mm=1:1:M;nn=1:1:N;
[mmm,nnn]=meshgrid(mm,nn);
u=1:1:M;v=1:1:N;
[uu,vv]=meshgrid(u,v);
c1=(-1).^(mmm-1).*(-1).^(nnn-1).*f;
C1=fft2(c1);
if axis_class =='X'
    C2=ifft2(C1.*(2*pi/LX*1i.*(uu-1-M/2)).^p);
    Dm=real(C2.*(-1).^(mmm-1).*(-1).^(nnn-1));
elseif axis_class =='Z'
    C2=ifft2(C1.*(2*pi/LZ*1i.*(vv-1-N/2)).^p);
    Dm=real(C2.*(-1).^(mmm-1).*(-1).^(nnn-1));
end
end