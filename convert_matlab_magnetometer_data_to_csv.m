% mat_to_csv.m
% Convert a .mat file to CSV

% Input file
mat_file = '/Users/masonjp2/Downloads/suncet_tr_20260406_125316.mat';

% Load MAT file
data = load(mat_file);

% Get variable names
vars = fieldnames(data);

% If only one variable, export it directly
if length(vars) == 1
    var_data = data.(vars{1});
    
    % If struct, convert to table
    if isstruct(var_data)
        T = struct2table(var_data);
    else
        T = array2table(var_data);
    end
    
    writetable(T, 'output.csv');
    
else
    % Multiple variables — write each to its own CSV
    for i = 1:length(vars)
        var_name = vars{i};
        var_data = data.(var_name);
        
        if isstruct(var_data)
            T = struct2table(var_data);
        else
            T = array2table(var_data);
        end
        
        csv_name = [var_name '.csv'];
        writetable(T, csv_name);
    end
end

disp('Conversion complete.');