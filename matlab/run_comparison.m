function [out, metrics] = run_comparison(outputFolder)
% Run the nominal, P1, and P2 comparisons and save Fig. 5 and tracking errors.
folder = fileparts(mfilename('fullpath'));
if nargin == 0, outputFolder = fullfile(folder, 'figures'); end
if ~isfolder(outputFolder), mkdir(outputFolder); end
model = 'Hinf_BS_Comparison';
modelFile = fullfile(folder, [model '.slx']);
if bdIsLoaded(model) && ~strcmp(get_param(model, 'FileName'), modelFile)
    error('Close the other Hinf_BS_Comparison model before running this copy.');
end
load_system(modelFile);
% SimulationOutput is available after sim returns, so plot here.
input = Simulink.SimulationInput(model);
input = input.setModelParameter('StopFcn', '');
out = sim(input);
plotCartPendComparison(out, fullfile(outputFolder, 'tracking_comparison'));
t = out.outputs.time(:);
y = squeeze(out.outputs.signals.values);
errors = y(:,7) - y(:,1:6);
SampleRMSE = sqrt(mean(errors.^2, 1)).';
TimeRMSE = sqrt(trapz(t, errors.^2, 1)/(t(end)-t(1))).';
PeakError = max(abs(errors), [], 1).';
Controller = [repmat("Hinf",3,1); repmat("Backstepping",3,1)];
Plant = repmat(["Nominal"; "P1"; "P2"],2,1);
metrics = table(Controller, Plant, SampleRMSE, TimeRMSE, PeakError);
disp(metrics);
writetable(metrics, fullfile(outputFolder, 'tracking_metrics.csv'));
end
