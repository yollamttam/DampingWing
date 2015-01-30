function meanSigmaF = AllResolutionChanges()

% filenames of spectra, we will iterate through
a = [
'Q0836-spec_LyaMatrix.tex';
'Q1306-spec_LyaMatrix.tex';
'qso_z_628_lo_res_LyaMatrix.txt';
'qso_z_637_lo_res_LyaMatrix.txt';
'z517.spec_LyaMatrix.tex';
'z521.spec_LyaMatrix.tex';
'z531.spec_LyaMatrix.tex';
'z541.spec_LyaMatrix.tex';
'z574_LyaMatrix.tex';
'z580_LyaMatrix.tex';
'z582_hres.spec_LyaMatrix.tex';
'z582_LyaMatrix.tex';
'z595_LyaMatrix.tex';
'z599_hres.spec_LyaMatrix.tex';
'z599_LyaMatrix.tex';
'z605_LyaMatrix.tex';
'z607_LyaMatrix.tex';
'z614_LyaMatrix.tex';
'z621_LyaMatrix.tex';
];

filenames = cellstr(a);
nfiles = length(filenames);

b = [
'Q0836-spec_snrEstimate.tex';
'Q1306-spec_snrEstimate.tex';
'qso_z_628_lo_res_snrEstimate.txt';
'qso_z_637_lo_res_snrEstimate.txt';
'z517.spec_snrEstimate.tex';
'z521.spec_snrEstimate.tex';
'z531.spec_snrEstimate.tex';
'z541.spec_snrEstimate.tex';
'z574_snrEstimate.tex';
'z580_snrEstimate.tex';
'z582_hres.spec_snrEstimate.tex';
'z582_snrEstimate.tex';
'z595_snrEstimate.tex';
'z599_hres.spec_snrEstimate.tex';
'z599_snrEstimate.tex';
'z605_snrEstimate.tex';
'z607_snrEstimate.tex';
'z614_snrEstimate.tex';
'z621_snrEstimate.tex';
];    
snrfiles = cellstr(b);


% corresponding spectra redshifts
zs = [
5.82;
5.99;
6.28;
6.37;
5.17;
5.21;
5.31;
5.41;
5.74;
5.80;
5.82;
5.82;
5.95;
5.99;
5.99;
6.05;
6.07;
6.14;
6.21;
];

sigmaF = [];

% loop through, alter spectral resolution
figure; 
resratios = linspace(1,30,1000);
sigratios = 1./resratios;
plot(resratios,sigratios);
for i = 1:nfiles
  z = zs(i);
  filename = filenames{i}
  snrfilename = snrfiles{i};
  [res0 FWHMfinal nratio OneSigmaF useBool] = SetSpectraResolution(filename,z,snrfilename);  
  hold on; plot(FWHMfinal/res0,nratio,'x');
  
  
  if (useBool)
    sigmaF = [sigmaF OneSigmaF];
  else
    disp('SKIPPING SPECTRA IN AVERAGE!!!');
  end
  
end  
hold on;
plot(resratios,sigratios);
xlabel('FWHM_{final}/v_{res,0}');
ylabel('\sigma_{N}^2/\sigma_{N,0}^2');
legend('1/(FWHM/v_{res})')
saveas(gcf,'NoisePlot.png');
  
%%% After an initial run, it looks like we get an average of 
%%% sigmaF = 0.161197
meanSigmaF = mean(sigmaF);
disp(sigmaF)
disps = sprintf('The average sigmaF that you found was %f...',meanSigmaF);
disp(disps);
close all;
end
