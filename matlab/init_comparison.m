function init_comparison(model)
% Load the common controller, reference, and nominal/perturbed plant parameters.
if nargin == 0, model = bdroot; end
folder = fileparts(get_param(model, 'FileName'));
data = load(fullfile(folder, 'comparison_data.mat'));
exportFile = fullfile(folder, '..', 'python', 'Galerkin_Control.mat');
if isfile(exportFile)
    generated = load(exportFile, 'AK', 'BK', 'CK', 'DK');
    for name = {'A', 'B', 'C', 'D'}
        field = [name{1} 'K'];
        if ~isfield(generated, field), error('Missing controller matrix %s.', field); end
        data.(name{1}) = generated.(field);
    end
end
if ~isequal(size(data.A), [14 14]) || ~isequal(size(data.B), [14 1]) ...
        || ~isequal(size(data.C), [1 14]) || ~isequal(size(data.D), [1 1])
    error('Expected the 14-state continuous-time controller.');
end
workspace = get_param(model, 'ModelWorkspace');
for name = {'A', 'B', 'C', 'D', 'r_test'}
    assignin(workspace, name{1}, data.(name{1}));
end
for name = {'a1', 'a2', 'a3', 'a4'}, assignin(workspace, name{1}, 3); end
assignin(workspace, 'b', -100);
assignin(workspace, 'w', 500); % Nominal tracking bandwidth: 225.57 rad/s.
% Point masses: nominal and doubled pendulum mass. P2: uniform rod.
parameters = [1.1 0.9 1 1; 1.1 1.8 1 1; 5.5 0.45 2 4];
suffix = {'_0', '', '_3'};
for j = 1:3
    M = parameters(j,1); m = parameters(j,2);
    L = parameters(j,3); damping = parameters(j,4);
    ell = L; inertia = m * L^2;
    if j == 3, ell = L/2; inertia = m * L^2/3; end
    values = [m*9.81*ell/inertia, damping/inertia, ...
        inertia*(M+m)/(m*ell), m*ell, m^2*ell^2/((M+m)*inertia)];
    for k = 1:5
        assignin(workspace, sprintf('k%d%s', k, suffix{j}), values(k));
    end
end
end
