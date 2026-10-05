function [fig, ax] = plotCartPendComparison(out, varargin)
% Columns: Hinf nominal/P1/P2, backstepping nominal/P1/P2, reference.
% plotCartPendComparison(out, 'tracking_comparison') saves a PDF.
% The reference is read from the seventh logged column.

exportBase = '';
if nargin >= 2
    if ischar(varargin{end}) || isstring(varargin{end})
        exportBase = char(varargin{end});
    end
end

logged = out.outputs;
if isfield(logged, 'time') && ~isempty(logged.time)
    t = logged.time(:);
else
    t = out.tout(:);
end
yRaw = squeeze(logged.signals.values);
if size(yRaw, 1) ~= numel(t) && size(yRaw, 2) == numel(t)
    yRaw = yRaw.';
end
nModels = 3;
nTrajectories = 2 * nModels;
refColumn = nTrajectories + 1;
nCols = size(yRaw, 2);
assert(nCols == refColumn, ...
    'Expected seven columns: three Hinf, three backstepping, and reference.');
assert(size(yRaw, 1) == numel(t), ...
    'The first output dimension must match the logged time vector length.');

y = yRaw(:, 1:nTrajectories);
r = yRaw(:, refColumn);
tr = t;

assert(numel(t) >= 2 && all(isfinite(t)) && ...
    all(diff(t) >= 0) && t(end) > t(1), ...
    'The logged time vector must span a nonzero interval.');

colors = [230 154 55; 99 69 146] / 255;  % H_inf: gold, Backstepping: purple
refColor = [20 20 20] / 255;
styleColor = [0.24 0.24 0.24];
styles = {'-', '--', '-.'};
markers = {'o', 's', '^'};
curveWidthHinf = 2.2;   % Emphasize H_inf against other curves and reference.
curveWidthBS = 1.4;     % Backstepping trajectories.
refWidth = 1.9;        % Thicker reference for overlap visibility.
markerSize = 5.6;
markerCount = 7;    % Per trajectory; sparse markers keep the curves readable.
markerLineWidth = 1.4;
zoomInsetLineWidthHinf = 1.4;
zoomInsetLineWidthBS = 1.2;
zoomX = [34.70 34.80];
zoomIndices = tr >= zoomX(1) & tr <= zoomX(2);
zoomValues = [y(zoomIndices, 1); y(zoomIndices, 4); r(zoomIndices)];
if isempty(zoomValues)
    zoomY = [0.54127 0.54146];
else
    zoomPad = max(0.12 * (max(zoomValues)-min(zoomValues)), 1e-6);
    zoomY = [min(zoomValues)-zoomPad, max(zoomValues)+zoomPad];
end
zoomRegionEmphasisFactor = 6.4 * sqrt(2); % Visual emphasis only; inset limits stay exact.
zoomRegionEmphasisFactorX = 2.0 * sqrt(2);
zoomRegionEdgeColor = [1 0 0];
zoomRegionLineWidth = 2.2;
zoomRegionMinXSpan = 0.005;
zoomRegionMinYSpan = 0.010;
zoomRegionFaceColor = [1 0.75 0.75];
zoomRegionFaceAlpha = 0.16;
zoomArrowColor = [1 0 0];
zoomArrowLineWidth = 1.0;
zoomArrowHeadLength = 7;
zoomArrowHeadWidth = 8;
zoomInsetDotRadius = 0;
fontName = 'Times New Roman';
if any(strcmpi(listfonts, 'STIXGeneral')), fontName = 'STIXGeneral'; end

fig = figure('Color', 'w', 'Units', 'centimeters', ...
    'Position', [3 3 18 6.8], 'Name', 'Cart-pendulum comparison', ...
    'NumberTitle', 'off');
tl = tiledlayout(fig, 1, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax = nexttile(tl);
hold(ax, 'on');

[tUnique, sampleIndex] = unique(t, 'stable');
markerIndices = cell(1, nTrajectories);
for controller = [2 1]  % Plot Backstepping first so H_inf overlays it.
    for model = 1:nModels
        col = nModels * (controller - 1) + model;
        phase = 0.15 + 0.70 * (col - 1) / (nTrajectories - 1);
        markerTimes = t(1) + ((0:markerCount-1) + phase) ...
            * (t(end) - t(1)) / markerCount;
        markerIndices{col} = unique(round(interp1(tUnique, sampleIndex, ...
            markerTimes, 'nearest')));
        plot(ax, t, y(:, col), ...
            'Color', colors(controller, :), 'LineStyle', styles{model}, ...
            'LineWidth', (controller == 1) * curveWidthHinf + ...
            (controller == 2) * curveWidthBS, 'HandleVisibility', 'off');
    end
end

for col = 1:nTrajectories
    controller = ceil(col / nModels);
    model = mod(col - 1, nModels) + 1;
    idx = markerIndices{col};
    markerFace = 'w';
    if controller == 1
        markerFace = colors(controller, :); % H_inf markers filled for stronger contrast
    end
    plot(ax, t(idx), y(idx, col), 'LineStyle', 'none', ...
        'Marker', markers{model}, 'MarkerSize', markerSize, ...
        'LineWidth', markerLineWidth, ...
        'MarkerFaceColor', markerFace, 'MarkerEdgeColor', colors(controller, :), ...
        'HandleVisibility', 'off');
end

if ~isempty(r)
    plot(ax, tr, r, '--', 'Color', refColor, 'LineWidth', refWidth, ...
    'HandleVisibility', 'off');
end

set(ax, 'FontName', fontName, 'FontSize', 8.5, ...
    'LineWidth', 0.6, 'TickDir', 'in', 'Box', 'on', ...
    'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 0.13, ...
    'TickLabelInterpreter', 'tex');
xlabel(ax, 't (s)', 'FontName', fontName, 'FontSize', 9);
ylabel(ax, 'y_1 (rad)', 'Interpreter', 'tex', ...
    'FontName', fontName, 'FontSize', 9);

inWindow = tr >= t(1) & tr <= t(end);
allY = [y(:); r(inWindow)];
allY = allY(isfinite(allY));
xlim(ax, [t(1), t(end)]); % Include the full logged run (175 s for the current data).
if ~isempty(allY)
    lo = min(allY);
    hi = max(allY);
    pad = 0.06 * (hi - lo);
    if pad == 0, pad = 0.05 * max(abs(lo), 1e-3); end
    ylim(ax, [lo - pad, hi + pad]);
end

hLegend = gobjects(1, 3 + nModels);
labels = {'H_\infty', 'Backstepping', 'Reference', ...
    'Nominal', 'P1', 'P2'};
for controller = 1:2
    hLegend(controller) = plot(ax, NaN, NaN, '-', ...
        'Color', colors(controller, :), 'LineWidth', ...
        (controller == 1) * curveWidthHinf + (controller == 2) * curveWidthBS);
end
hLegend(3) = plot(ax, NaN, NaN, '--', ...
    'Color', refColor, 'LineWidth', refWidth);
for model = 1:nModels
    hLegend(3 + model) = plot(ax, NaN, NaN, ...
        'Color', styleColor, 'LineStyle', styles{model}, 'LineWidth', curveWidthBS, ...
        'Marker', markers{model}, 'MarkerSize', 6, ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', styleColor);
end
lgd = legend(ax, hLegend, labels, 'Orientation', 'horizontal', ...
    'NumColumns', 3, 'AutoUpdate', 'off', 'Box', 'off', 'Interpreter', 'tex', ...
    'FontName', fontName, 'FontSize', 8);
lgd.Layout.Tile = 'south';
if isprop(lgd, 'IconColumnWidth')
    lgd.IconColumnWidth = 34;
elseif isprop(lgd, 'ItemTokenSize')
    lgd.ItemTokenSize = [34 12];
end

xZoom = sort(zoomX);
yZoom = sort(zoomY);
xBoxMin = xZoom(1);
xBoxMax = xZoom(2);
yBoxMin = yZoom(1);
yBoxMax = yZoom(2);
if xZoom(1) < xZoom(2) && yZoom(1) < yZoom(2) ...
        && xZoom(2) >= t(1) && xZoom(1) <= t(end) ...
        && yZoom(2) >= min(y(:)) && yZoom(1) <= max(y(:))
    axXLim = xlim(ax);
    axYLim = ylim(ax);
    if isfinite(diff(axXLim)) && diff(axXLim) > 0 && isfinite(diff(axYLim)) && diff(axYLim) > 0
        xHalf = max(0.5 * (xBoxMax - xBoxMin) * zoomRegionEmphasisFactorX, ...
            zoomRegionMinXSpan * diff(axXLim));
        yHalf = max(0.5 * (yBoxMax - yBoxMin) * zoomRegionEmphasisFactor, ...
            zoomRegionMinYSpan * diff(axYLim));
        xMid = 0.5 * (xBoxMin + xBoxMax);
        yMid = 0.5 * (yBoxMin + yBoxMax);
        xBoxMin = max(xMid - xHalf, axXLim(1));
        xBoxMax = min(xMid + xHalf, axXLim(2));
        yBoxMin = max(yMid - yHalf, axYLim(1));
        yBoxMax = min(yMid + yHalf, axYLim(2));
    else
        xBoxMin = xZoom(1);
        xBoxMax = xZoom(2);
        yBoxMin = yZoom(1);
        yBoxMax = yZoom(2);
    end
    if xBoxMax <= xBoxMin || yBoxMax <= yBoxMin
        xBoxMin = xZoom(1);
        xBoxMax = xZoom(2);
        yBoxMin = yZoom(1);
        yBoxMax = yZoom(2);
    end
    rectangle(ax, 'Position', [xBoxMin, yBoxMin, diff([xBoxMin, xBoxMax]), diff([yBoxMin, yBoxMax])], ...
        'LineStyle', '-', 'LineWidth', zoomRegionLineWidth, ...
        'EdgeColor', zoomRegionEdgeColor, ...
        'FaceColor', zoomRegionFaceColor, 'FaceAlpha', zoomRegionFaceAlpha, ...
        'HandleVisibility', 'off');

    drawnow;
    figUnits = get(fig, 'Units');
    axUnits = get(ax, 'Units');
    set(fig, 'Units', 'pixels');
    set(ax, 'Units', 'pixels');
    figPosPix = get(fig, 'Position');
    mainPosPix = get(ax, 'Position');
    set(ax, 'Units', axUnits);
    set(fig, 'Units', figUnits);
    mainPos = [ ...
        mainPosPix(1) / figPosPix(3), ...
        mainPosPix(2) / figPosPix(4), ...
        mainPosPix(3) / figPosPix(3), ...
        mainPosPix(4) / figPosPix(4) ];

    baseZoomInsetW = min(0.32, max(0.18, 0.36 * mainPos(3)));
    baseZoomInsetH = min(0.28, max(0.14, 0.40 * mainPos(4)));
    zoomInsetScale = 1.0;
    zoomInsetW = min(mainPos(3) - 0.025, zoomInsetScale * baseZoomInsetW);
    zoomInsetH = min(mainPos(4) - 0.025, zoomInsetScale * baseZoomInsetH);
    rightMargin = 0.005;
    zoomInsetX = mainPos(1) + mainPos(3) - zoomInsetW - rightMargin;
    lowerMiddleOffset = 0.08;
    zoomInsetY = mainPos(2) + lowerMiddleOffset * mainPos(4);
    zoomInsetY = min(zoomInsetY, mainPos(2) + mainPos(4) - zoomInsetH - 0.03);
    zoomInsetY = max(zoomInsetY, mainPos(2) + 0.02);
    zoomInsetPos = [zoomInsetX, zoomInsetY, zoomInsetW, zoomInsetH];

    axZoom = axes('Parent', fig, 'Units', 'normalized', ...
        'Position', zoomInsetPos, 'Color', 'white', 'LineWidth', 0.8, ...
        'Tag', 'zoomInset', 'HitTest', 'off');
    hold(axZoom, 'on');
    title(axZoom, 'Nominal', 'FontName', fontName, 'FontSize', 6, ...
        'FontWeight', 'normal');

    for col = [1 4] % Only the two nominal responses in the inset.
        controller = ceil(col / nModels);
        model = mod(col - 1, nModels) + 1;
        plot(axZoom, t, y(:, col), ...
            'Color', colors(controller, :), ...
            'LineStyle', styles{model}, ...
            'LineWidth', (controller == 1) * zoomInsetLineWidthHinf + ...
            (controller == 2) * zoomInsetLineWidthBS, ...
            'HandleVisibility', 'off');
    end

    if ~isempty(r)
        plot(axZoom, tr, r, '--', 'Color', refColor, 'LineWidth', 1.3, ...
            'HandleVisibility', 'off');
    end

    xlim(axZoom, xZoom);
    ylim(axZoom, yZoom);
    set(axZoom, 'FontName', fontName, 'FontSize', 6, ...
        'LineWidth', 0.8, 'TickDir', 'in', ...
        'Box', 'on', 'XGrid', 'on', 'YGrid', 'on', ...
        'GridAlpha', 0.2, 'TickLabelInterpreter', 'tex', ...
        'XColor', [0 0 0], 'YColor', [0 0 0], ...
        'XLimMode', 'manual', 'YLimMode', 'manual');
    set(axZoom, 'XTick', [34.70 34.75 34.80], ...
        'YTick', [0.54130 0.54135 0.54140], ...
        'FontSize', 6, 'XMinorGrid', 'off', 'YMinorGrid', 'off');
    xtickformat(axZoom, '%.2f');
    ytickformat(axZoom, '%.5f');

    set(axZoom, 'Layer', 'top');
    uistack(axZoom, 'top');

    axPos = get(ax, 'Position');
    axXLim = xlim(ax);
    axYLim = ylim(ax);
    if diff(axXLim) > 0 && diff(axYLim) > 0 && xBoxMin >= axXLim(1) ...
            && xZoom(2) <= axXLim(2) && yZoom(1) >= axYLim(1) ...
            && yZoom(2) <= axYLim(2)
        zoomPos = get(axZoom, 'Position');
        fromNormX = axPos(1) + axPos(3) * (xBoxMax - axXLim(1)) / diff(axXLim);
        fromNormY = axPos(2) + axPos(4) * (yBoxMax - axYLim(1)) / diff(axYLim);
        toNormX = zoomPos(1) + 0.02 * zoomPos(3);
        toNormY = zoomPos(2) + 0.02 * zoomPos(4);
        annotation(fig, 'arrow', [fromNormX, toNormX], [fromNormY, toNormY], ...
            'Color', zoomArrowColor, 'LineWidth', zoomArrowLineWidth, ...
            'LineStyle', '-', 'HeadStyle', 'plain', ...
            'HeadLength', zoomArrowHeadLength, 'HeadWidth', zoomArrowHeadWidth);
    if zoomInsetDotRadius > 0
        annotation(fig, 'ellipse', [fromNormX - zoomInsetDotRadius, ...
            fromNormY - zoomInsetDotRadius, 2 * zoomInsetDotRadius, ...
            2 * zoomInsetDotRadius], 'LineStyle', '-', ...
            'LineWidth', 2.2, 'Color', zoomArrowColor, ...
            'FaceColor', zoomArrowColor, 'EdgeColor', zoomArrowColor);
        annotation(fig, 'ellipse', [toNormX - 1.2 * zoomInsetDotRadius, ...
            toNormY - 1.2 * zoomInsetDotRadius, 2.4 * zoomInsetDotRadius, ...
            2.4 * zoomInsetDotRadius], 'LineStyle', '-', ...
            'LineWidth', 2.2, 'Color', zoomArrowColor, ...
            'FaceColor', zoomArrowColor, 'EdgeColor', zoomArrowColor);
    end
    end
end

if ~isempty(exportBase)
    exportBase = char(exportBase);
    exportgraphics(fig, [exportBase '.pdf'], ...
        'ContentType', 'vector', 'BackgroundColor', 'white');
end
end
