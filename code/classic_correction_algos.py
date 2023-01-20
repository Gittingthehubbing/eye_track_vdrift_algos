import numpy as np
import pandas as pd

def slice_for_dffix(dffix,trial, x_thresh=192, y_thresh=32, w_thresh=32, n_thresh=90):
    
    fixation_array = dffix.loc[:,["x","y"]].values
    y_diff = trial["y_diff"]

    if "y_char_unique" in trial:
        corrected_fix_y_vals = slice(fixation_array,trial["y_char_unique"],line_height=y_diff, x_thresh=x_thresh, y_thresh=y_thresh, w_thresh=w_thresh, n_thresh=n_thresh)
    else:
        corrected_fix_y_vals = slice(fixation_array,trial["y_midline"],line_height=y_diff, x_thresh=x_thresh, y_thresh=y_thresh, w_thresh=w_thresh, n_thresh=n_thresh)

    corrected_line_nums = [trial["y_char_unique"].index(y) for y in corrected_fix_y_vals]
    if "assigned_line" in dffix.columns:
        acc = sum(corrected_line_nums == dffix.assigned_line.values) / dffix.shape[0]
    df = pd.DataFrame()
    df["x_eyekit"] = fixation_array[:,0]
    df["y_eyekit"] = corrected_fix_y_vals
    df["line_num_eyekit"] = corrected_line_nums

    if "y_eyekit" in dffix.columns:
        for col in ["x_eyekit","y_eyekit","line_num_eyekit"]:
            dffix.loc[:,col] = df.loc[:,col]
    else:
        assert all(dffix.index.values == df.index.values), "Index mismatch for eyekit correction"
        dffix = pd.concat([dffix,df],axis=1)
    return dffix

def slice(fixation_XY, midlines,line_height:float, x_thresh=192, y_thresh=32, w_thresh=32, n_thresh=90):
    """
    Adapted from Eyekit(https://github.com/jwcarr/eyekit/blob/350d055eecaa1581b03db5a847424825ffbb10f6/eyekit/_snap.py)
    implementation

    Form a set of runs and then reduce the set to *m* by repeatedly merging
    those that appear to be on the same line. Merged sequences are then
    assigned to text lines in positional order. Default params:
    `x_thresh=192`, `y_thresh=32`, `w_thresh=32`, `n_thresh=90`. Requires
    NumPy. Original method by [Glandorf & Schroeder (2021)](https://doi.org/10.1016/j.procs.2021.09.069).
    """
    fixation_XY = np.array(fixation_XY, dtype=float)
    line_Y = np.array(midlines, dtype=float)
    proto_lines, phantom_proto_lines = {}, {}
    # 1. Segment runs
    dist_X = abs(np.diff(fixation_XY[:, 0]))
    dist_Y = abs(np.diff(fixation_XY[:, 1]))
    end_run_indices = list(
        np.where(np.logical_or(dist_X > x_thresh, dist_Y > y_thresh))[0] + 1
    )
    run_starts = [0] + end_run_indices
    run_ends = end_run_indices + [len(fixation_XY)]
    runs = [list(range(start, end)) for start, end in zip(run_starts, run_ends)]
    # 2. Determine starting run
    longest_run_i = np.argmax(
        [fixation_XY[run[-1], 0] - fixation_XY[run[0], 0] for run in runs]
    )
    proto_lines[0] = runs.pop(longest_run_i)
    # 3. Group runs into proto lines
    while runs:
        merger_on_this_iteration = False
        for proto_line_i, direction in [(min(proto_lines), -1), (max(proto_lines), 1)]:
            # Create new proto line above or below (depending on direction)
            proto_lines[proto_line_i + direction] = []
            # Get current proto line XY coordinates (if proto line is empty, get phanton coordinates)
            if proto_lines[proto_line_i]:
                proto_line_XY = fixation_XY[proto_lines[proto_line_i]]
            else:
                proto_line_XY = phantom_proto_lines[proto_line_i]
            # Compute differences between current proto line and all runs
            run_differences = np.zeros(len(runs))
            for run_i, run in enumerate(runs):
                y_diffs = [
                    y - proto_line_XY[np.argmin(abs(proto_line_XY[:, 0] - x)), 1]
                    for x, y in fixation_XY[run]
                ]
                run_differences[run_i] = np.mean(y_diffs)
            # Find runs that can be merged into this proto line
            merge_into_current = list(np.where(abs(run_differences) < w_thresh)[0])
            # Find runs that can be merged into the adjacent proto line
            merge_into_adjacent = list(
                np.where(
                    np.logical_and(
                        run_differences * direction >= w_thresh,
                        run_differences * direction < n_thresh,
                    )
                )[0]
            )
            # Perform mergers
            for index in merge_into_current:
                proto_lines[proto_line_i].extend(runs[index])
            for index in merge_into_adjacent:
                proto_lines[proto_line_i + direction].extend(runs[index])
            # If no, mergers to the adjacent, create phantom line for the adjacent
            if not merge_into_adjacent:
                average_x, average_y = np.mean(proto_line_XY, axis=0)
                adjacent_y = average_y + line_height * direction
                phantom_proto_lines[proto_line_i + direction] = np.array(
                    [[average_x, adjacent_y]]
                )
            # Remove all runs that were merged on this iteration
            for index in sorted(merge_into_current + merge_into_adjacent, reverse=True):
                del runs[index]
                merger_on_this_iteration = True
        # If no mergers were made, break the while loop
        if not merger_on_this_iteration:
            break
    # 4. Assign any leftover runs to the closest proto lines
    for run in runs:
        best_pl_distance = np.inf
        best_pl_assignemnt = None
        for proto_line_i in proto_lines:
            if proto_lines[proto_line_i]:
                proto_line_XY = fixation_XY[proto_lines[proto_line_i]]
            else:
                proto_line_XY = phantom_proto_lines[proto_line_i]
            y_diffs = [
                y - proto_line_XY[np.argmin(abs(proto_line_XY[:, 0] - x)), 1]
                for x, y in fixation_XY[run]
            ]
            pl_distance = abs(np.mean(y_diffs))
            if pl_distance < best_pl_distance:
                best_pl_distance = pl_distance
                best_pl_assignemnt = proto_line_i
        proto_lines[best_pl_assignemnt].extend(run)
    # 5. Prune proto lines
    while len(proto_lines) > len(line_Y):
        top, bot = min(proto_lines), max(proto_lines)
        if len(proto_lines[top]) < len(proto_lines[bot]):
            proto_lines[top + 1].extend(proto_lines[top])
            del proto_lines[top]
        else:
            proto_lines[bot - 1].extend(proto_lines[bot])
            del proto_lines[bot]
    # 6. Map proto lines to text lines
    for line_i, proto_line_i in enumerate(sorted(proto_lines)):
        fixation_XY[proto_lines[proto_line_i], 1] = line_Y[line_i]
    return fixation_XY[:, 1]