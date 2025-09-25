function cell_out = add_channel_dim_to_cell(cell_in)
    % This function loops through each 2D matrix in a cell array
    % and reshapes it to have an explicit 3rd dimension of size 1.
    num_samples = numel(cell_in);
    cell_out = cell(size(cell_in));
    for i = 1:num_samples
        sample = cell_in{i};
        cell_out{i} = reshape(sample, [size(sample, 1), size(sample, 2), 1]);
    end
end