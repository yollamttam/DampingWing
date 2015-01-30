function [res0 FWHMfinal nratio sigmaF SNRprovided] = SetSpectraResolution(filename,z,snrfilename)
FWHMhires = 6.7;
FWHMmike = 13.6;
FWHMfinal = 140;
c = 3e5;
%calculated ratio of red-side variance to blue side
alphabar = 0.087133;

%cite 1306.2314 for this:
%HIRES FWHM = 6.7 km/s
%MIKE FWHM = 13.6 km/s
%HIRES vbin = 2.1 km/s
%MIKE vbin = 5.0 km/s

data = load(filename);
disp(size(data));
snrEstimate = load(snrfilename);
zs = data(1,:);
fluxA = data(2,:);
snr = data(3,:);
if (abs(mean(snr)-snrEstimate)<0.1)
  SNRprovided = 0;
  disp([mean(snr) snrEstimate])
else
  SNRprovided = 1;
end
lambdaA0 = 1216*(1+z);
lambdas = 1216*(1+zs);
vs = c*(lambdas-lambdaA0)/lambdaA0;
Npixels = length(vs);
dv = vs(2)-vs(1);
%vs = dV*(is - 1);
center = floor(Npixels/2) + 1;
is = 1:1:Npixels;
%vbins for output spectra
minv = min(vs); maxv = max(vs);
vbinHIRES = 2.1;
vbinMIKE = 5.0;
vbinfinal = 140;
vsHIRES = minv:vbinHIRES:maxv;
vsMIKE = minv:vbinMIKE:maxv;
vsfinal = minv:vbinfinal:maxv;

%construct filter array

sigmaHIRES = FWHMhires/2.355;
sigmaMIKE = FWHMmike/2.355;
sigmafinal = FWHMfinal/2.355;
dk = 2*pi/(maxv-minv);
ks = (is-center)*dk; 
HIRESfilter = exp(-ks.^2*sigmaHIRES^2/2);
MIKEfilter = exp(-ks.^2*sigmaMIKE^2/2);
finalfilter = exp(-ks.^2*sigmafinal^2/2);


%perform convolution to find new noise variance.
meansnr = mean(snr); sigmaN0 = 1/meansnr; varN0 = sigmaN0^2;
L = abs(maxv-minv); N = Npixels;
Pk = varN0*L/N;
varN = Pk*dk*sum(finalfilter.^2)/(2*pi);
nratio = varN/varN0;
res0 = L/Npixels;


%FOURIER TRANSFORM
FLUXA = fftshift(fft(fluxA));

%apply filter for HIRES
FLUXAfinal = finalfilter.*FLUXA;
  
%inverse Fourier transform
fluxAfinal = ifft(ifftshift(FLUXAfinal));

%now we need to sparsely sample. 
fluxAfinal = interp1(vs,fluxAfinal,vsfinal);
snrinterp = interp1(vs,snr,vsfinal);
%snrfinal = snrinterp/sqrt(nratio);
varEstimate = 1/snrEstimate^2;
smoothedBlueVarEstimate = alphabar*varEstimate*nratio;
finalsnrestimate = 1/sqrt(smoothedBlueVarEstimate);
finalsnrestimate = real(finalsnrestimate);
snrfinal = finalsnrestimate*ones(size(snrinterp));

%ok, for our output file, we actually want corresponding
% redshifts: z = zQ - v/c(1+zQ), zQ = redshift of quasar
zsfinal = z - (-1*vsfinal/c)*(1+z);

vlength = 1000;
plots = 0;
if (plots == 1)
% mini = 1; maxi = floor(vlength/dv);
% minif = 1; maxif = floor(vlength/vbinfinal);
mini = 1; maxi = length(fluxA);
minif = 1; maxif = length(fluxAfinal);
figure; plot(vs(mini:maxi),fluxA(mini:maxi));
hold on; plot(vsfinal(minif:maxif),fluxAfinal(minif:maxif),'o','Color','red');
xlabel('v (km/s)'); ylabel('F');
legend('Original Spectrum','Smoothed Spectrum');
end

%save files
saveBool = 0;
%saveBool = input('Would you like to save smoothed spectrum?');
if (saveBool == 1)
  [pathstr, name, ext, versn] = fileparts(filename);
  fluxAfile = sprintf('%s_SMOOTHED_sigma%d-kms_vbin%d-kms.tex',name,sigmafinal,vbinfinal);
  
  fid = fopen(fluxAfile,'w');
  dlmwrite(fluxAfile,fluxAfinal,'precision',6);
  fclose(fid);
end

%calculate estimate of contribution to variance due to resonant
%absorption
varT = var(fluxAfinal);
varF = varT - varN;
if (varF <= 0)
  SNRprovided = 0;
  sigmaF = -100000;
  varF
else
  sigmaF = sqrt(varF)
end

[pathstr, name, ext, versn] = fileparts(filename);
sigmaNblue = sprintf('%s_SmoothedSigmaNBlueside.tex',name);
fid = fopen(sigmaNblue,'w');
dlmwrite(sigmaNblue,sqrt(varN));
fclose(fid);

[pathstr, name, ext, versn] = fileparts(filename);
nRatioFile = sprintf('%s_NratioFromSmoothing.tex',name);
fid = fopen(nRatioFile,'w');
dlmwrite(nRatioFile,nratio);
fclose(fid);

outputmatrix = [zsfinal; fluxAfinal; snrfinal];
if (sum(imag(zsfinal).^2) > .1)
  disp('zsfinal has an imaginary part...');
end
if (sum(imag(fluxAfinal).^2) > .1)
  disp('fluxAfinal has an imaginary part...');
end
if (sum(imag(snrfinal).^2) > .1)
  disp('snrfinal has an imaginary part...');
end
outputmatrix = real(outputmatrix);
disp(size(outputmatrix));
[pathstr, name, ext, versn] = fileparts(filename);
smoothfile = sprintf('%s_smoothed.tex',name);
fid = fopen(smoothfile,'w');
dlmwrite(smoothfile,outputmatrix);
fclose(fid);
disps = sprintf('saved output to %s...',smoothfile);
disp(disps);

nratio = 1;
[pathstr, name, ext, versn] = fileparts(smoothfile);
nRatioFile = sprintf('%s_NratioFromSmoothing.tex',name);
fid = fopen(nRatioFile,'w');
dlmwrite(nRatioFile,nratio);
fclose(fid);

end