from simulation import ReadingScenario
import eyekit
import pandas as pd
import numpy as np
from PIL import Image
factor_dict = dict()
factor_dict['noise'] = 10
reading_scenario = ReadingScenario(
    **factor_dict,
    lines_per_passage=(8,12),
    max_characters_per_line=80,
    character_spacing=16,
    line_spacing=64,
    include_line_breaks=False
)

passage, fixation_XY, intended_I = reading_scenario.simulate()

average_duration = 80.0
fixation_times = []
for idx in range(len(fixation_XY)):
    if idx == 0:
        t_start = 10
    else:
        t_start = t_end
    t_end = np.round(t_start + (np.random.rand()+1.)*average_duration,1)
    fixation_times.append(dict(start=t_start,end=t_end))
fixation_times = pd.DataFrame(fixation_times)

fix_df = pd.DataFrame([{
    "x":fixation_XY[idx][0],
    "y":fixation_XY[idx][1],
    "assigned_line":intended_I[idx],
    "y_midline":passage.midlines[intended_I[idx]],
    "corrected_start_time":fixation_times.start.values[idx],
    "corrected_end_time":fixation_times.end.values[idx],
} for idx in range(len(fixation_XY))])


fixations_tuples = [
    (x[1]["x"],x[1]["y"],x[1]["corrected_start_time"],x[1]["corrected_end_time"]) 
    if x[1]["corrected_start_time"] < x[1]["corrected_end_time"] 
    else (x[1]["x"],x[1]["y"],x[1]["corrected_start_time"],x[1]["corrected_end_time"]+1)  for x in fix_df.iterrows()
]

seq = eyekit.FixationSequence(fixations_tuples)

box = passage.box
img = eyekit.vis.Image(box[2]+box[0], box[3]+box[1])
img.draw_text_block(passage)
img.draw_fixation_sequence(seq)
img.save('test_passage.png')

im = Image.open('test_passage.png')
im