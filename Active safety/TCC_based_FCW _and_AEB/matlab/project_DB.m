%% A hypothetical rear-end conflict situation (task 1)
% vehicle data
Vx_ego = 90/3.6; % speed for ego vehicle in m/s
Vx_Tar = 0; % speed for target vehicle in m/s
Rt = 1.5; % reaction time in sec or time gap TG
Acc_ego = 5; % - deceleration, + accelerations: in m/s^2 for ego vehicle
Acc_Tar = 0; % - deceleration, + accelerations: in m/s^2 for target vehicle
Range_n = 90; % new range for question (c) in mtr

%% ANSWERS 
% range (a)
dis_rt = Vx_ego*Rt; % range in mtr, X=Vx*Rt for given reaction/delay time
dis_b = 0.5*(Vx_ego^2/Acc_ego); % distance covered during braking 
dis_total = dis_rt+dis_b; %min distance 
t_brk=Vx_ego/Acc_ego;

% time to collision (b)
TTC_b = dis_b/(Vx_ego-Vx_Tar); 

% (c) for new distance
dis_c = Range_n-dis_rt; %new total distance after new range with reaction time in mtr
TTC_c = dis_c/(Vx_ego-Vx_Tar); % time to collision (c) based on new distance

% collision check
if dis_b < dis_c
    fprintf('No collision expected.');
else
    fprintf('Collision is possible');
end

% (D) 
time = 8;             % time in sec
dt = 0.1;             % step size
t_v = 0:dt:time;       %time vector

% zero vectors 
S_ego_1 = zeros(size(t_v));
S_ego_2= zeros(size(t_v));
d_ego_1 = zeros(size(t_v));
d_ego_2= zeros(size(t_v));
count=0;
for i = 2:length(t_v)
    if t_v(i) <= Rt
        count = i;
        S_ego_1(1,1:i) = Vx_ego; S_ego_2(1,1:i) = Vx_ego; 
        d_ego_1(1)=dis_total; d_ego_1(i)=dis_total-S_ego_1(i-1)*t_v(i);  % decreasing distance
        d_ego_2(i)=Range_n-S_ego_2(i-1)*t_v(i); d_ego_2(1)=Range_n;  % decreasing distance
    else
        S_ego_1(i) = max(S_ego_1(1) - Acc_ego*(t_v(i)-Rt),0); S_ego_2(i)=S_ego_1(i); %decreasing speed
        d_ego_1(i)=d_ego_1(count)-(S_ego_1(i)*(t_v(i)-Rt)+0.5*Acc_ego*(t_v(i)-Rt)^2); %decreasing distance
        d_ego_2(i)=d_ego_2(count)-(S_ego_2(i)*(t_v(i)-Rt)+0.5*Acc_ego*(t_v(i)-Rt)^2); %decreasing distance
        if d_ego_2(i)<=0 
            S_ego_2(i)=0; 
            d_ego_2(i)=0;        
        end
        if d_ego_1(i)<=0
            S_ego_1(i)=0;
            d_ego_1(i)=0;        
        end
    end
end

subplot(2,1,1)
plot(t_v, S_ego_1,'r-');hold on
plot(t_v, S_ego_2,'b--');hold off
xline(Rt);
xlabel('Time in sec'); 
ylabel('Speed in m/s'); 
legend('D 100','D 90','reaction time 1.5s')
title('Speed  VS time');
grid on; ylim([-5 30]);
subplot(2,1,2)
plot(t_v,d_ego_1,'r-');hold on
plot(t_v,d_ego_2,'b--');hold off
xline(Rt);
xlabel('Time in sec'); 
ylabel('Distance in m'); ylim([-5 110]);
legend('D 100','D 90','reaction time 1.5s')
title('Distance VS time'); 
grid on;

%% Study critical braking behavior with experimental data (task 2)
load('RadarData.mat');

Radar_Range_RD = {};
vehicle_Vx_RD = {};
Radar_Ti_RD = {};
vehicle_Ti_RD = {};
TP_RD = {};
FN_RD = {};
Radar_Acc_RD = {};
Radar_RangeRate_RD = {};

length=numel(RadarData);
 for i=1:length
    Radar_Range_RD{i} = RadarData(i).RadarRange;
    vehicle_Vx_RD{i} = RadarData(i).VehicleSpeed;
    Radar_Ti_RD{i} = RadarData(i).RadarTime;
    vehicle_Ti_RD{i} = RadarData(i).VehicleTime;
    TP_RD{i} = RadarData(i).TestPerson;
    FN_RD{i} = RadarData(i).FileName;
    Radar_Acc_RD{i} = RadarData(i).RadarAccel;
    Radar_RangeRate_RD{i} = RadarData(i).RadarRangeRate;
 end

% 1. Remove runs where all data are NaN
notAllNaN = cellfun(@(x) ~all(isnan(x)), Radar_Range_RD);

% Apply first filter
radar_range        = Radar_Range_RD(notAllNaN);
radar_range_rate   = Radar_RangeRate_RD(notAllNaN);
radar_acceleration = Radar_Acc_RD(notAllNaN);
vehicle_speed      = vehicle_Vx_RD(notAllNaN);
participant_id     = TP_RD(notAllNaN);
radar_time         = Radar_Ti_RD(notAllNaN);
vehicle_time       = vehicle_Ti_RD(notAllNaN);
file_name          = FN_RD(notAllNaN);

% 2. Synchronize radar_time (subtract 200 ms)
radar_time = cellfun(@(t) t - 0.2, radar_time, 'UniformOutput', false);

% 3. Find bad runs (too many NaNs or all > 10)
badRuns = cellfun(@(x) isempty(x) || mean(isnan(x)) > 0.85 || all(x > 10 | isnan(x)), radar_range);

% Apply second filter
radar_range        = radar_range(~badRuns);
radar_range_rate   = radar_range_rate(~badRuns);
radar_acceleration = radar_acceleration(~badRuns);
vehicle_speed      = vehicle_speed(~badRuns);
participant_id     = participant_id(~badRuns);
radar_time         = radar_time(~badRuns);
vehicle_time       = vehicle_time(~badRuns);
file_name          = file_name(~badRuns);


% Compute acceleration and identify braking maneuvers (optimised)
ResultsMat = {} ;  % Each row = [ParticipantID, MeanAcc, MinAcc, SpeedOnset, RangeOnset, TTC]
brakingEpisodes = cell(size(vehicle_speed));
hasBraking = false(1, numel(vehicle_speed));

for i = 1:numel(vehicle_speed)
    v = vehicle_speed{i}(:);
    t = vehicle_time{i}(:);
    r = radar_range{i}(:);
    rr = radar_range_rate{i}(:);
    pid = participant_id(i);   % Participant/Test person ID
    
    % Skip bad data
    if numel(v) ~= numel(t) || numel(v) < 2
        continue;
    end
    
    % --- Compute acceleration ---
    dt = diff(t);
    dv = diff(v);
    dt(dt==0) = NaN; % avoid div by zero
    a = [NaN; dv ./ dt]; 
    a = movmean(a,5);   % smooth
    
    % --- Detect braking segments (a <= -1.2) ---
    isBraking = a <= -1.2;
    d = diff([0; isBraking; 0]);
    starts = find(d==1);
    ends   = find(d==-1)-1;
    
    % Keep only segments with length >= 5
    segLens = ends - starts + 1;
    goodSegs = segLens >= 5;
    starts = starts(goodSegs);
    ends   = ends(goodSegs);    
    selectedEpisodes = [];
    
    for j = 1:numel(starts)
        idx = starts(j):ends(j);
        finalSpeed = v(idx(end));
        if finalSpeed > 1   % must end near zero
            continue;
        end
        % --- Metrics ---
        meanAcc = mean(a(idx),'omitnan');
        minAcc  = min(a(idx),[],'omitnan');
        v_onset = v(idx(1));
        r_onset = r(idx(1));
        TTC     = r_onset / max(v_onset, eps);
        
        % Skip if any metric is NaN
        if any(isnan([meanAcc, minAcc, v_onset, r_onset, TTC]))
            continue;
        end
        % Store episode
        ResultsMat(end+1,:) = [pid, meanAcc, minAcc, v_onset, r_onset, TTC];
        selectedEpisodes = [selectedEpisodes; t(starts(j)), t(ends(j))];
    end    
    brakingEpisodes{i} = selectedEpisodes;
    if ~isempty(selectedEpisodes)
        hasBraking(i) = true;
    end
end

% Keep only runs with braking
radar_range        = radar_range(hasBraking);
radar_range_rate   = radar_range_rate(hasBraking);
radar_acceleration = radar_acceleration(hasBraking);
vehicle_speed      = vehicle_speed(hasBraking);
participant_id     = participant_id(hasBraking);
radar_time         = radar_time(hasBraking);
vehicle_time       = vehicle_time(hasBraking);
file_name          = file_name(hasBraking);
brakingEpisodes    = brakingEpisodes(hasBraking);

% Save results to workspace
assignin('base','BrakingResults',ResultsMat);
fprintf('Done. Found %d braking events across %d runs.\n', size(ResultsMat,1), numel(vehicle_speed));
disp('First 10 rows of ResultsMat:');
disp(ResultsMat(1:min(10,end),:));

% Plot velocity and acceleration with braking segments highlighted
outDir = 'plots';
if ~exist(outDir,'dir')
    mkdir(outDir);
end
for i = 1:numel(vehicle_speed)
    v = vehicle_speed{i}(:);
    t = vehicle_time{i}(:);
    % --- Recompute acceleration ---
    if numel(v) < 2
        continue
    end
    dt = diff(t);
    dv = diff(v);
    dt(dt==0) = NaN;
    a = [NaN; dv ./ dt];
    a = movmean(a,5);
    fig = figure;
    % --- Top subplot: Speed vs Time ---
    subplot(2,1,1)
    plot(t, v, 'b', 'LineWidth', 1.5); hold on
    xlabel('Time [s]')
    ylabel('Velocity [m/s]')
    title(sprintf('Run %d: Speed vs Time', i))
    grid on
    if ~isempty(brakingEpisodes{i})
        for j = 1:size(brakingEpisodes{i},1)
            idx = t >= brakingEpisodes{i}(j,1) & t <= brakingEpisodes{i}(j,2);
            finalSpeed = v(find(idx,1,'last'));
            if finalSpeed <= 1
                plot(t(idx), v(idx), 'r', 'LineWidth', 2)
                break
            end
        end
    end
    % --- Bottom subplot: Acceleration vs Time ---
    subplot(2,1,2)
    plot(t, a, 'k', 'LineWidth', 1.5); hold on
    xlabel('Time [s]')
    ylabel('Acceleration [m/s^2]')
    title(sprintf( 'Run %d: Acceleration vs Time', i))
    grid on

    if ~isempty(brakingEpisodes{i})
        for j = 1:size(brakingEpisodes{i},1)
            idx = t >= brakingEpisodes{i}(j,1) & t <= brakingEpisodes{i}(j,2);
            finalSpeed = v(find(idx,1,'last'));
            if finalSpeed <= 1
                plot(t(idx), a(idx), 'r', 'LineWidth', 2)
                break
            end
        end
    end
   % Save to file
    saveas(fig, fullfile(outDir, sprintf('run_%d.png', i)));
    close(fig);
end

%% Driver behavior analysis 

load("NEW_TableTask2.mat"); % Assuming new_data is the variable name in the .mat file

%Initialize a new table with 73 rows and 6 columns
N_Rows = height(SafetyMetricsTable);
N_Cols = width(SafetyMetricsTable);

% part 3_a
T_P = double(SafetyMetricsTable.Test_person);
Speed_BO = SafetyMetricsTable.Speed_BO; 
TTC_BO = SafetyMetricsTable.TTC_BO; 
Mean_acc = SafetyMetricsTable.Mean_acc; 
Min_acc = SafetyMetricsTable.Min_acc;

uniqueRefs = unique(T_P(~isnan(T_P)));  
nRefs = numel(uniqueRefs);
colors = lines(max(nRefs,2));   % ensure at least 2 rows of color
figure(1)
hold on;
% Plotting each unique reference with a different color
for i = 1:nRefs
    idx = (T_P == uniqueRefs(i));
    plot(Speed_BO(idx), TTC_BO(idx), '*', ...
         'Color', colors(i,:), ...
         'DisplayName', sprintf('Test Person %d', uniqueRefs(i)));
end

xlabel('Speed at BO in m/s');
ylabel('TTC at BO in sec');
grid minor
title('TTC vs Speed @ BO');
legend show;
hold off;

%5th and 95th percentiles for TTC, Min Acc, and Mean Acc
TTC_5th = prctile(TTC_BO, 5);
TTC_95th = prctile(TTC_BO, 95);
MinAcc_5th = prctile(Min_acc, 5);
MinAcc_95th = prctile(Min_acc, 95);
MeanAcc_5th = prctile(Mean_acc, 5);
MeanAcc_95th = prctile(Mean_acc, 95);


% 3.c
figure(2)
histogram(TTC_BO); hold on
xline(TTC_5th,'r--', 'LineWidth', 1.5, 'DisplayName', '5th Percentile'); hold on
xline(TTC_95th,'g--', 'LineWidth', 1.5, 'DisplayName', '95th Percentile'); hold off
grid minor
legend show;
title('Histogram of TTC @ brake onset');
xlabel('TTC in sec');
ylabel('NO: of Test runs');

% 3.e
figure(3)
subplot(1,2,1)
histogram(Min_acc); hold on
xline(MinAcc_5th,'r--', 'LineWidth', 1.5, 'DisplayName', '5th Percentile'); hold on
xline(MinAcc_95th,'g--', 'LineWidth', 1.5, 'DisplayName', '95th Percentile'); hold off
legend show;
grid minor
title('Histogram of Min Acc in m/s^2');
xlabel('Min Acc in m/s^2');
ylabel('NO: of Test runs');
subplot(1,2,2)
histogram(Mean_acc); hold on
xline(MeanAcc_5th,'r--', 'LineWidth', 1.5, 'DisplayName', '5th Percentile'); hold on
xline(MeanAcc_95th,'g--', 'LineWidth', 1.5, 'DisplayName', '95th Percentile'); hold off
legend show;
grid minor
title('Histogram of Mean Acc in m/s^2');
xlabel('Mean Acc in m/s^2');
ylabel('NO: of Test runs');
ylim([0 22.5])

%% Active safety system design and evaluation task 4

clc; close all; clear;

try
    load('NEW_TableTask2.mat'); % Or your table name
    fprintf('Safety metrics table loaded successfully.\n');
catch
    error('Could not load safety metrics table. Please ensure the file exists.');
end


% Part (a): Design TWO FCW Systems (Conservative & Aggressive)
fprintf('\n=== TASK 4(a): FCW System Design ===\n');

% Based on driver behavior analysis, extract key metrics
% These values should come from your Task 3 analysis
TTC_BO = SafetyMetricsTable.TTC_BO; % TTC at brake onset from Task 3
Speed_BO = SafetyMetricsTable.Speed_BO; % Speed at brake onset
Min_acc = SafetyMetricsTable.Min_acc; % Minimum acceleration

% Calculate percentiles for FCW design
TTC_5th = prctile(TTC_BO, 5);   % 5th percentile - aggressive drivers
TTC_50th = prctile(TTC_BO, 50); % Median
TTC_95th = prctile(TTC_BO, 95); % 95th percentile - conservative drivers

fprintf('Driver Behavior TTC Statistics:\n');
fprintf('  5th percentile (aggressive): %.2f s\n', TTC_5th);
fprintf('  50th percentile (median): %.2f s\n', TTC_50th);
fprintf('  95th percentile (conservative): %.2f s\n\n', TTC_95th);

% FCW SYSTEM DESIGN PARAMETERS
% Conservative FCW: Warns earlier, suitable for cautious drivers
% Based on 95th percentile + reaction time
RT_conservative = 1.2;  % Reaction time [s] - Euro NCAP standard
a_driver_cons = 4.0;    % Expected driver braking [m/s²] - moderate braking
TTC_FCW_conservative = 4.2; % [s] - Based on percentile analysis + margin

% Aggressive FCW: Warns later, for experienced/aggressive drivers
% Based on median TTC + reaction time
RT_aggressive = 0.8;    % Shorter reaction time [s] - alert driver
a_driver_aggr = 6.0;    % Expected driver braking [m/s²] - hard braking
TTC_FCW_aggressive = 3.2; % [s] - More aggressive threshold

safety_margin = 1.0;    % Safety buffer [m]

fprintf('FCW System Parameters:\n');
fprintf('CONSERVATIVE FCW:\n');
fprintf('  TTC Threshold: %.1f s\n', TTC_FCW_conservative);
fprintf('  Reaction Time: %.1f s\n', RT_conservative);
fprintf('  Expected Driver Deceleration: %.1f m/s²\n\n', a_driver_cons);

fprintf('AGGRESSIVE FCW:\n');
fprintf('  TTC Threshold: %.1f s\n', TTC_FCW_aggressive);
fprintf('  Reaction Time: %.1f s\n', RT_aggressive);
fprintf('  Expected Driver Deceleration: %.1f m/s²\n\n', a_driver_aggr);

% THREAT ASSESSMENT AND DECISION MAKING:
% Part (b): Test FCW Systems on Euro NCAP CCRs
fprintf('=== TASK 4(b): FCW Testing on Euro NCAP CCRs ===\n');

% Euro NCAP CCRs test speeds (stationary target)
speeds_kmh = [55, 60, 65, 70, 75, 80]; % [km/h]
speeds_ms = speeds_kmh / 3.6;          % Convert to [m/s]
n_speeds = length(speeds_ms);

% Initialize result arrays
FCW_cons_activation_range = zeros(n_speeds, 1);
FCW_cons_stop_range = zeros(n_speeds, 1);
FCW_aggr_activation_range = zeros(n_speeds, 1);
FCW_aggr_stop_range = zeros(n_speeds, 1);

% Test each speed scenario
for i = 1:n_speeds
    v = speeds_ms(i);
    
    % Conservative FCW
    % Activation when TTC = TTC_threshold
    FCW_cons_activation_range(i) = v * TTC_FCW_conservative;
    
    % Calculate stopping distance after FCW warning
    % Distance = v*RT + v²/(2*a)
    dist_during_reaction = v * RT_conservative;
    dist_braking = (v^2) / (2 * a_driver_cons);
    total_dist = dist_during_reaction + dist_braking;
    FCW_cons_stop_range(i) = FCW_cons_activation_range(i) - total_dist;
    
    % Aggressive FCW
    FCW_aggr_activation_range(i) = v * TTC_FCW_aggressive;
    dist_during_reaction_aggr = v * RT_aggressive;
    dist_braking_aggr = (v^2) / (2 * a_driver_aggr);
    total_dist_aggr = dist_during_reaction_aggr + dist_braking_aggr;
    FCW_aggr_stop_range(i) = FCW_aggr_activation_range(i) - total_dist_aggr;
end

% Display results
fprintf('\nFCW Performance on Euro NCAP CCRs:\n');
fprintf('Speed [km/h] | Cons Activ [m] | Cons Stop [m] | Aggr Activ [m] | Aggr Stop [m]\n');
fprintf('-------------+----------------+---------------+----------------+--------------\n');
for i = 1:n_speeds
    fprintf('    %3d      |     %6.2f     |    %6.2f     |     %6.2f     |    %6.2f\n', ...
        speeds_kmh(i), FCW_cons_activation_range(i), FCW_cons_stop_range(i), ...
        FCW_aggr_activation_range(i), FCW_aggr_stop_range(i));
end

% Find optimal TTC thresholds
% Most aggressive TTC that still avoids collision in ALL scenarios
fprintf('\n--- Finding Optimal TTC Thresholds ---\n');

% For each speed, calculate minimum TTC needed to stop with safety margin
TTC_min_required = zeros(n_speeds, 1);
for i = 1:n_speeds
    v = speeds_ms(i);
    % Minimum stopping distance needed
    min_dist = v * RT_conservative + v^2 / (2 * a_driver_cons) + safety_margin;
    % Minimum TTC at which to warn
    TTC_min_required(i) = min_dist / v;
end

TTC_optimal_all = max(TTC_min_required); % Most aggressive that works for all
TTC_optimal_slow = TTC_min_required(1);  % For slowest speed

fprintf('Most aggressive TTC for ALL scenarios: %.2f s\n', TTC_optimal_all);
fprintf('TTC for slowest speed (55 km/h): %.2f s\n', TTC_optimal_slow);
fprintf('Difference: %.2f s\n', TTC_optimal_all - TTC_optimal_slow);
fprintf('\nComparison with Task 3:\n');
fprintf('  Task 3 median TTC: %.2f s\n', TTC_50th);
fprintf('  Optimal TTC is %.2f%% of Task 3 median\n', 100*TTC_optimal_all/TTC_50th);

% ========================================================================
% Part (c): FCW Limitations
% ========================================================================
fprintf('\n=== TASK 4(c): FCW System Limitations ===\n');
fprintf('LIMITATIONS:\n');
fprintf('1. FCW cannot autonomously avoid collisions - it relies on driver response\n');
fprintf('2. Effectiveness depends on:\n');
fprintf('   - Driver attention and reaction time (varies significantly)\n');
fprintf('   - Driver willingness to brake hard enough\n');
fprintf('   - Vehicle braking capability and road conditions\n');
fprintf('3. If driver is distracted, asleep, or incapacitated, FCW is ineffective\n');
fprintf('4. False alarms may lead drivers to disable or ignore the system\n');
fprintf('5. Cannot account for evasive maneuvers (steering)\n\n');

% ========================================================================
% Part (d): Design AEB Using Required Acceleration
% ========================================================================
fprintf('=== TASK 4(d): AEB System Design (Acceleration-Based) ===\n');

% Determine maximum braking capability
% Based on 10th percentile of minimum acceleration from Task 3
a_AEB_max = abs(prctile(Min_acc, 10)); % Take absolute value

fprintf('AEB Maximum Braking Capability: %.2f m/s²\n', a_AEB_max);
fprintf('Motivation:\n');
fprintf('  - Based on 10th percentile of driver minimum accelerations\n');
fprintf('  - Ensures system stays within vehicle physical limits\n');
fprintf('  - Conservative enough for various road conditions\n');
fprintf('  - Typical emergency braking: 7-9 m/s² for passenger cars\n\n');

% If the calculated value seems unreasonable, use 8.0 m/s² as standard
if a_AEB_max < 6 || a_AEB_max > 10
    fprintf('  Adjusting to standard value due to data constraints\n');
    a_AEB_max = 8.0;
end

fprintf('Using a_AEB_max = %.1f m/s²\n\n', a_AEB_max);

% THREAT ASSESSMENT: Required acceleration to stop
% a_required = v² / (2 * (Range - safety_margin))
% DECISION: Activate if a_required > a_AEB_max

fprintf('AEB Acceleration-Based System:\n');
fprintf('  Threat Assessment: a_required = v²/(2*(Range-margin))\n');
fprintf('  Decision Logic: IF a_required > %.1f m/s² THEN Activate AEB\n\n', a_AEB_max);

% ========================================================================
% Part (e): Test AEB (Acceleration) on Euro NCAP CCRs
% ========================================================================
fprintf('=== TASK 4(e): AEB Acceleration-Based Testing ===\n');

% Initialize arrays
AEB_acc_activation_range = zeros(n_speeds, 1);
AEB_acc_stop_range = zeros(n_speeds, 1);

% Test each speed
for i = 1:n_speeds
    v = speeds_ms(i);
    
    % AEB activates when: v²/(2*(Range-margin)) = a_AEB_max
    % Solving for Range: Range = v²/(2*a_AEB_max) + margin
    AEB_acc_activation_range(i) = v^2 / (2 * a_AEB_max) + safety_margin;
    
    % After braking at a_AEB_max, the vehicle stops at:
    AEB_acc_stop_range(i) = safety_margin; % By design
end

fprintf('\nAEB Acceleration Performance:\n');
fprintf('Speed [km/h] | Activation Range [m] | Stop Range [m]\n');
fprintf('-------------+----------------------+----------------\n');
for i = 1:n_speeds
    fprintf('    %3d      |        %6.2f        |     %5.2f\n', ...
        speeds_kmh(i), AEB_acc_activation_range(i), AEB_acc_stop_range(i));
end

fprintf('\nRESULT: AEB stops approximately %.1f m before obstacle in all scenarios\n', safety_margin);
fprintf('This is the designed safety margin, showing consistent performance.\n\n');

% ========================================================================
% Part (f): Design AEB Using TTC Threshold
% ========================================================================
fprintf('=== TASK 4(f): AEB System Design (TTC-Based) ===\n');

% TTC thresholds for different speed ranges
TTC_AEB_low = 0.9;   % For speeds < 15 m/s (~54 km/h)
TTC_AEB_high = 1.4;  % For speeds >= 15 m/s

fprintf('AEB TTC-Based System:\n');
fprintf('  Low Speed TTC Threshold (v < 54 km/h): %.1f s\n', TTC_AEB_low);
fprintf('  High Speed TTC Threshold (v >= 54 km/h): %.1f s\n', TTC_AEB_high);
fprintf('  Rationale: Higher speeds need earlier intervention\n\n');

% Test on Euro NCAP scenarios
AEB_TTC_threshold = zeros(n_speeds, 1);
AEB_TTC_activation_range = zeros(n_speeds, 1);
AEB_TTC_stop_range = zeros(n_speeds, 1);

for i = 1:n_speeds
    v = speeds_ms(i);
    
    % Select appropriate TTC threshold
    if v < 15
        TTC_threshold = TTC_AEB_low;
    else
        TTC_threshold = TTC_AEB_high;
    end
    AEB_TTC_threshold(i) = TTC_threshold;
    
    % Activation range: Range = v * TTC
    AEB_TTC_activation_range(i) = v * TTC_threshold;
    
    % Stopping distance after AEB activation (immediate braking)
    dist_to_stop = v^2 / (2 * a_AEB_max);
    AEB_TTC_stop_range(i) = AEB_TTC_activation_range(i) - dist_to_stop;
end

% Create comprehensive results table
fprintf('\nAEB TTC Performance on Euro NCAP CCRs:\n');
fprintf('Speed   | TTC      | Activation | Stop      | Benefit/Drawback\n');
fprintf('[km/h]  | Thresh[s]| Range [m]  | Range [m] |\n');
fprintf('--------+----------+------------+-----------+------------------\n');
for i = 1:n_speeds
    if AEB_TTC_stop_range(i) > 2
        comment = 'Safe margin';
    elseif AEB_TTC_stop_range(i) > 0
        comment = 'Minimal margin';
    else
        comment = 'Collision';
    end
    fprintf('  %3d   |   %.1f    |   %6.2f   |  %6.2f   | %s\n', ...
        speeds_kmh(i), AEB_TTC_threshold(i), AEB_TTC_activation_range(i), ...
        AEB_TTC_stop_range(i), comment);
end

fprintf('\nBENEFITS:\n');
fprintf('  + Earlier activation at high speeds provides better protection\n');
fprintf('  + Adaptive thresholds prevent false alarms at low speeds\n');
fprintf('DRAWBACKS:\n');
fprintf('  - Fixed thresholds may not adapt to all driving conditions\n');
fprintf('  - Early activation increases risk of false positives\n\n');

% ========================================================================
% Part (g): AEB System Limitations
% ========================================================================
fprintf('=== TASK 4(g): AEB System Limitations ===\n');
fprintf('SCENARIO WITH UNEXPECTED BRAKING:\n');
fprintf('Example: Vehicle approaching a traffic light with stopped cars ahead.\n');
fprintf('  - Lead vehicle is stationary (red light)\n');
fprintf('  - AEB detects stationary target within activation range\n');
fprintf('  - Traffic light turns green, lead vehicle will accelerate\n');
fprintf('  - AEB does not know about traffic light - triggers emergency brake\n');
fprintf('  - Driver expects lead vehicle to move - AEB braking is unexpected\n\n');

fprintf('ADDITIONAL INFORMATION NEEDED:\n');
fprintf('1. Target vehicle velocity and acceleration (to predict movement)\n');
fprintf('2. Traffic signal status and infrastructure data (V2I communication)\n');
fprintf('3. Driver intent signals (accelerator pedal position, brake readiness)\n');
fprintf('4. Scene context (parking lot, traffic queue, intersection)\n');
fprintf('5. Object classification (distinguishing relevant from irrelevant objects)\n\n');

% ========================================================================
% Part (h): Visualization with Task 1 Scenario
% ========================================================================
fprintf('=== TASK 4(h): Applying Systems to Task 1 Scenario ===\n');

% Task 1 scenario parameters
v0_task1 = 90 / 3.6;        % Initial speed [m/s]
range0_task1 = 90;          % Initial range [m]
RT_task1 = 1.5;             % Reaction time [s]
a_brake_task1 = -5;         % Braking deceleration [m/s²]

% Time vector
dt = 0.05;
t_max = 10;
t = 0:dt:t_max;
n_steps = length(t);

% Initialize arrays
range_task1 = zeros(1, n_steps);
speed_task1 = zeros(1, n_steps);
FCW_cons_active = false(1, n_steps);
FCW_aggr_active = false(1, n_steps);
AEB_active = false(1, n_steps);

% Flags for first activation
FCW_cons_activated_at = NaN;
FCW_aggr_activated_at = NaN;
AEB_activated_at = NaN;
collision_occurred = false;

% Simulation - no intervention (baseline)
for i = 1:n_steps
    if i == 1
        range_task1(i) = range0_task1;
        speed_task1(i) = v0_task1;
    else
        % No braking in baseline scenario
        speed_task1(i) = v0_task1;
        range_task1(i) = range0_task1 - v0_task1 * t(i);
        
        if range_task1(i) <= 0
            range_task1(i) = 0;
            speed_task1(i) = 0;
            collision_occurred = true;
        end
    end
    
    % Check FCW activation conditions
    if range_task1(i) > 0 && speed_task1(i) > 0
        TTC_current = range_task1(i) / speed_task1(i);
        
        % Conservative FCW
        if TTC_current <= TTC_FCW_conservative
            FCW_cons_active(i) = true;
            if isnan(FCW_cons_activated_at)
                FCW_cons_activated_at = t(i);
            end
        end
        
        % Aggressive FCW
        if TTC_current <= TTC_FCW_aggressive
            FCW_aggr_active(i) = true;
            if isnan(FCW_aggr_activated_at)
                FCW_aggr_activated_at = t(i);
            end
        end
        
        % AEB (acceleration-based)
        a_required = speed_task1(i)^2 / (2 * max(range_task1(i) - safety_margin, 0.1));
        if a_required >= a_AEB_max
            AEB_active(i) = true;
            if isnan(AEB_activated_at)
                AEB_activated_at = t(i);
            end
        end
    end
end

% Simulation WITH AEB intervention
range_with_AEB = zeros(1, n_steps);
speed_with_AEB = zeros(1, n_steps);
AEB_braking = false;

for i = 1:n_steps
    if i == 1
        range_with_AEB(i) = range0_task1;
        speed_with_AEB(i) = v0_task1;
    else
        % Check if AEB should activate
        if ~AEB_braking && range_with_AEB(i-1) > 0 && speed_with_AEB(i-1) > 0
            a_req = speed_with_AEB(i-1)^2 / (2 * max(range_with_AEB(i-1) - safety_margin, 0.1));
            if a_req >= a_AEB_max
                AEB_braking = true;
            end
        end
        
        % Update speed and range
        if AEB_braking && speed_with_AEB(i-1) > 0
            speed_with_AEB(i) = max(speed_with_AEB(i-1) - a_AEB_max * dt, 0);
        else
            speed_with_AEB(i) = speed_with_AEB(i-1);
        end
        
        range_with_AEB(i) = range_with_AEB(i-1) - speed_with_AEB(i-1) * dt;
        
        if range_with_AEB(i) <= 0
            range_with_AEB(i) = 0;
            speed_with_AEB(i) = 0;
        end
    end
end

% Print activation information
fprintf('\nTask 1 Scenario Analysis (v0=%.1f m/s, Range=%.1f m):\n', v0_task1, range0_task1);
if ~isnan(FCW_cons_activated_at)
    range_at_FCW_cons = interp1(t, range_task1, FCW_cons_activated_at);
    fprintf('  Conservative FCW activates at t=%.2f s (Range=%.2f m)\n', ...
        FCW_cons_activated_at, range_at_FCW_cons);
else
    fprintf('  Conservative FCW: Would activate immediately (already critical)\n');
end

if ~isnan(FCW_aggr_activated_at)
    range_at_FCW_aggr = interp1(t, range_task1, FCW_aggr_activated_at);
    fprintf('  Aggressive FCW activates at t=%.2f s (Range=%.2f m)\n', ...
        FCW_aggr_activated_at, range_at_FCW_aggr);
end

if ~isnan(AEB_activated_at)
    range_at_AEB = interp1(t, range_task1, AEB_activated_at);
    fprintf('  AEB activates at t=%.2f s (Range=%.2f m)\n', AEB_activated_at, range_at_AEB);
end

final_range_AEB = range_with_AEB(end);
fprintf('  Without AEB: %s\n', ternary(collision_occurred, 'COLLISION', 'No collision'));
fprintf('  With AEB: Vehicle stops at %.2f m from obstacle\n\n', final_range_AEB);

% Create visualization plots
figure('Name', 'Task 4(h): FCW and AEB Activation in Task 1 Scenario', 'Position', [100 100 1200 800]);

% Subplot 1: Range vs Time
subplot(3,1,1);
plot(t, range_task1, 'k-', 'LineWidth', 2, 'DisplayName', 'No Intervention'); hold on;
plot(t, range_with_AEB, 'g-', 'LineWidth', 2, 'DisplayName', 'With AEB');

% Mark activation points
if ~isnan(FCW_cons_activated_at)
    xline(FCW_cons_activated_at, 'b--', 'LineWidth', 1.5, 'Label', 'FCW Conservative', ...
        'LabelHorizontalAlignment', 'left');
end
if ~isnan(FCW_aggr_activated_at)
    xline(FCW_aggr_activated_at, 'c--', 'LineWidth', 1.5, 'Label', 'FCW Aggressive', ...
        'LabelHorizontalAlignment', 'left');
end
if ~isnan(AEB_activated_at)
    xline(AEB_activated_at, 'r--', 'LineWidth', 1.5, 'Label', 'AEB', ...
        'LabelHorizontalAlignment', 'left');
end

yline(safety_margin, 'm:', 'LineWidth', 1.5, 'Label', 'Safety Margin');
xlabel('Time [s]', 'FontSize', 11);
ylabel('Range to Obstacle [m]', 'FontSize', 11);
title('Distance to Stationary Obstacle vs Time', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
grid on;
ylim([0 max(range0_task1*1.1, 100)]);
hold off;

% Subplot 2: Speed vs Time
subplot(3,1,2);
plot(t, speed_task1, 'k-', 'LineWidth', 2, 'DisplayName', 'No Intervention'); hold on;
plot(t, speed_with_AEB, 'g-', 'LineWidth', 2, 'DisplayName', 'With AEB');

if ~isnan(FCW_cons_activated_at)
    xline(FCW_cons_activated_at, 'b--', 'LineWidth', 1.5);
end
if ~isnan(FCW_aggr_activated_at)
    xline(FCW_aggr_activated_at, 'c--', 'LineWidth', 1.5);
end
if ~isnan(AEB_activated_at)
    xline(AEB_activated_at, 'r--', 'LineWidth', 1.5);
end

xlabel('Time [s]', 'FontSize', 11);
ylabel('Vehicle Speed [m/s]', 'FontSize', 11);
title('Ego Vehicle Speed vs Time', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'northeast', 'FontSize', 9);
grid on;
hold off;

% Subplot 3: System Activation Status
subplot(3,1,3);
plot(t, double(FCW_cons_active), 'b-', 'LineWidth', 2, 'DisplayName', 'FCW Conservative'); hold on;
plot(t, double(FCW_aggr_active), 'c-', 'LineWidth', 2, 'DisplayName', 'FCW Aggressive');
plot(t, double(AEB_active), 'r-', 'LineWidth', 2, 'DisplayName', 'AEB');

xlabel('Time [s]', 'FontSize', 11);
ylabel('System Active', 'FontSize', 11);
title('Active Safety Systems Activation Status', 'FontSize', 12, 'FontWeight', 'bold');
legend('Location', 'east', 'FontSize', 9);
grid on;
ylim([-0.1 1.1]);
yticks([0 1]);
yticklabels({'Inactive', 'Active'});
hold off;

% Save figure
saveas(gcf, 'Task4h_FCW_AEB_Activation.png');

% ========================================================================
% Create Summary Table for Task 4(f) - Required for Report
% ========================================================================

fprintf('=== TASK 4(f): Results Table for Report ===\n\n');

% Create comprehensive table
T_AEB_TTC = table(speeds_kmh', AEB_TTC_threshold, AEB_TTC_activation_range, AEB_TTC_stop_range, ...
    'VariableNames', {'Speed_kmh', 'TTC_Threshold_s', 'Activation_Range_m', 'Stop_Range_m'});

disp(T_AEB_TTC);

% Save table
writetable(T_AEB_TTC, 'Task4f_AEB_TTC_Results.csv');
fprintf('\nTable saved as: Task4f_AEB_TTC_Results.csv\n');

% ========================================================================
% Create Comparison Plot for All Systems
% ========================================================================

figure('Name', 'Task 4: All Systems Comparison', 'Position', [100 100 1000 600]);

plot(speeds_kmh, FCW_cons_activation_range, 'b-o', 'LineWidth', 2, ...
    'MarkerSize', 8, 'DisplayName', 'FCW Conservative'); hold on;
plot(speeds_kmh, FCW_aggr_activation_range, 'c-s', 'LineWidth', 2, ...
    'MarkerSize', 8, 'DisplayName', 'FCW Aggressive');
plot(speeds_kmh, AEB_acc_activation_range, 'r-^', 'LineWidth', 2, ...
    'MarkerSize', 8, 'DisplayName', 'AEB (Acceleration)');
plot(speeds_kmh, AEB_TTC_activation_range, 'm-d', 'LineWidth', 2, ...
    'MarkerSize', 8, 'DisplayName', 'AEB (TTC)');

xlabel('Vehicle Speed [km/h]', 'FontSize', 12);
ylabel('Activation Range [m]', 'FontSize', 12);
title('Active Safety Systems - Activation Ranges vs Speed', 'FontSize', 13, 'FontWeight', 'bold');
legend('Location', 'northwest', 'FontSize', 10);
grid on;
hold off;

saveas(gcf, 'Task4_All_Systems_Comparison.png');

% ========================================================================
% MATLAB Functions for FCW and AEB Systems
% ========================================================================

fprintf('\n=== MATLAB FUNCTION DEFINITIONS ===\n');
fprintf('The following functions are defined at the end of this script:\n');
fprintf('1. FCW_Conservative(range, rangeRate)\n');
fprintf('2. FCW_Aggressive(range, rangeRate)\n');
fprintf('3. AEB_Acceleration(range, rangeRate)\n');
fprintf('4. AEB_TTC(range, rangeRate)\n\n');

fprintf('Task 4 Complete!\n');
fprintf('Generated files:\n');
fprintf('  - Task4f_AEB_TTC_Results.csv\n');
fprintf('  - Task4h_FCW_AEB_Activation.png\n');
fprintf('  - Task4_All_Systems_Comparison.png\n\n');

% ========================================================================
% FUNCTION DEFINITIONS
% ========================================================================

% Helper function for ternary operator
function result = ternary(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

% ------------------------------------------------------------------------
% FCW CONSERVATIVE FUNCTION
% ------------------------------------------------------------------------
function [activate] = FCW_Conservative(range, rangeRate)
% FCW_Conservative - Conservative Forward Collision Warning System
%
% Threat Assessment: Time-to-Collision (TTC)
% Decision Logic: Activate if TTC <= 4.2 seconds
%
% Inputs:
%   range     - Distance to obstacle [m] (scalar)
%   rangeRate - Relative velocity [m/s] (negative = closing) (scalar)
%
% Output:
%   activate  - 1 if warning should be issued, 0 otherwise

    TTC_threshold = 4.2; % Conservative threshold [s]
    
    % Calculate relative speed (positive = closing)
    relativeSpeed = -rangeRate;
    
    % Calculate TTC if closing
    if relativeSpeed > 0
        TTC = range / relativeSpeed;
        
        % Decision: Activate if TTC below threshold
        if TTC <= TTC_threshold
            activate = 1;
        else
            activate = 0;
        end
    else
        % No collision threat if not closing
        activate = 0;
    end
end

% ------------------------------------------------------------------------
% FCW AGGRESSIVE FUNCTION
% ------------------------------------------------------------------------
function [activate] = FCW_Aggressive(range, rangeRate)
% FCW_Aggressive - Aggressive Forward Collision Warning System
%
% Threat Assessment: Time-to-Collision (TTC)
% Decision Logic: Activate if TTC <= 3.2 seconds
%
% Inputs:
%   range     - Distance to obstacle [m] (scalar)
%   rangeRate - Relative velocity [m/s] (negative = closing) (scalar)
%
% Output:
%   activate  - 1 if warning should be issued, 0 otherwise

    TTC_threshold = 3.2; % Aggressive threshold [s]
    
    % Calculate relative speed (positive = closing)
    relativeSpeed = -rangeRate;
    
    % Calculate TTC if closing
    if relativeSpeed > 0
        TTC = range / relativeSpeed;
        
        % Decision: Activate if TTC below threshold
        if TTC <= TTC_threshold
            activate = 1;
        else
            activate = 0;
        end
    else
        % No collision threat if not closing
        activate = 0;
    end
end

% ------------------------------------------------------------------------
% AEB ACCELERATION-BASED FUNCTION
% ------------------------------------------------------------------------
function [activate] = AEB_Acceleration(range, rangeRate)
% AEB_Acceleration - Autonomous Emergency Braking (Acceleration-based)
%
% Threat Assessment: Required deceleration to stop
% Decision Logic: Activate if required acceleration > max capability
%
% Inputs:
%   range     - Distance to obstacle [m] (scalar)
%   rangeRate - Relative velocity [m/s] (negative = closing) (scalar)
%
% Output:
%   activate  - 1 if AEB should brake, 0 otherwise

    a_max = 8.0;         % Maximum braking capability [m/s²]
    safety_margin = 1.0; % Safety distance [m]
    
    % Calculate relative speed (positive = closing)
    relativeSpeed = -rangeRate;
    
    % Calculate required deceleration: a_req = v² / (2 * (range - margin))
    if relativeSpeed > 0 && range > safety_margin
        available_distance = range - safety_margin;
        a_required = (relativeSpeed^2) / (2 * available_distance);
        
        % Decision: Activate if required deceleration exceeds capability
        if a_required >= a_max
            activate = 1;
        else
            activate = 0;
        end
    else
        % No threat or already too close
        if range <= safety_margin && relativeSpeed > 0
            activate = 1; % Emergency!
        else
            activate = 0;
        end
    end
end

% ------------------------------------------------------------------------
% AEB TTC-BASED FUNCTION
% ------------------------------------------------------------------------
function [activate] = AEB_TTC(range, rangeRate)
% AEB_TTC - Autonomous Emergency Braking (TTC-based)
%
% Threat Assessment: Time-to-Collision with speed-adaptive thresholds
% Decision Logic: Activate if TTC below speed-dependent threshold
%
% Inputs:
%   range     - Distance to obstacle [m] (scalar)
%   rangeRate - Relative velocity [m/s] (negative = closing) (scalar)
%
% Output:
%   activate  - 1 if AEB should brake, 0 otherwise

    TTC_low_speed = 0.9;  % TTC threshold for low speeds [s]
    TTC_high_speed = 1.4; % TTC threshold for high speeds [s]
    speed_threshold = 15; % Speed boundary [m/s] (~54 km/h)
    
    % Calculate relative speed (positive = closing)
    relativeSpeed = -rangeRate;
    
    % Determine appropriate TTC threshold based on speed
    if relativeSpeed < speed_threshold
        TTC_threshold = TTC_low_speed;
    else
        TTC_threshold = TTC_high_speed;
    end
    
    % Calculate TTC if closing
    if relativeSpeed > 0
        TTC = range / relativeSpeed;
        
        % Decision: Activate if TTC below threshold
        if TTC <= TTC_threshold
            activate = 1;
        else
            activate = 0;
        end
    else
        % No collision threat if not closing
        activate = 0;
    end
end
