%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CHAIN
% This is an adaptation of the chain method in popEye:
% https://github.com/sascha2schroeder/popEye/
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function fixation_XY = chain(fixation_XY, line_Y, x_thresh, y_thresh)

	if ~exist('x_thresh')
		x_thresh = 192;
	end
	if ~exist('y_thresh')
		y_thresh = 32;
	end

	n = size(fixation_XY, 1);
	dist_X = abs(diff(fixation_XY(:, 1)));
	dist_Y = abs(diff(fixation_XY(:, 2)));
	end_chain_indices = find(dist_X > x_thresh | dist_Y > y_thresh).';
	end_chain_indices = [end_chain_indices, n];
	start_of_chain = 1;
	for chain_i = 1 : length(end_chain_indices)
		end_of_chain = end_chain_indices(chain_i);
		mean_y = mean(fixation_XY(start_of_chain:end_of_chain, 2));
		[_, line_i] = min(abs(line_Y - mean_y));
		fixation_XY(start_of_chain:end_of_chain, 2) = line_Y(line_i);
		start_of_chain = end_of_chain + 1;
	end

end
