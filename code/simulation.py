'''
Code for performing the fixation sequence simulations.
'''

from sys import stdout
import pickle
import numpy as np
import eyekit
import lorem
import algorithms
from classic_correction_algos import slice
import core
from tqdm.auto import tqdm
import pandas as pd
import os
import pathlib as pl
import json

class ReadingScenario:

	def __init__(self, noise=0, slope=0, shift=0, regression_within=0, regression_between=0, lines_per_passage=(8, 12), max_characters_per_line=80, character_spacing=16, line_spacing=64):
		# Distortion parameters
		self.noise = noise
		self.slope = slope
		self.shift = shift
		# Regression parameters
		self.regression_within = regression_within
		self.regression_between = regression_between
		# Passage parameters
		self.min_lines, self.max_lines = lines_per_passage
		self.max_characters_per_line = max_characters_per_line
		self.character_spacing = character_spacing
		self.line_spacing = line_spacing

	def _generate_passage(self):
		n_lines = np.random.randint(self.min_lines, self.max_lines+1)
		lines = ['']
		has_paragraph_gap = False
		while len(lines) < n_lines:
			for word in lorem.sentence().split():
				if (len(lines[-1]) + len(word)) <= self.max_characters_per_line:
					lines[-1] += word + ' '
				elif len(lines) == n_lines//2 and np.random.rand() > 0.75 and not has_paragraph_gap:
					lines.append(' ')
					lines.append(word + ' ')
					has_paragraph_gap = True
				elif len(lines) > 2 and np.random.rand() > 0.8 and not has_paragraph_gap:
					lines.append(' ')
					lines.append(word + ' ')
					has_paragraph_gap = True
					if np.random.rand() > 0.7:
						words_last_line = lines[-1].split(' ')
						keep_len = max(int(len(words_last_line) * np.random.rand()),2)
						lines[-1] = ' '.join(words_last_line[:keep_len])
				else:
					lines.append(word + ' ')
			if len(lines) == n_lines and len(lines[-1].split()) == 1:
				# Final line only contains one word, so add in an extra word
				# because a one-word final line can be problematic for merge
				# since it cannot create sequences with one fixation.
				lines[-1] = 'lorem ' + lines[-1]
		font_size=round(26.667+(5*(1-np.random.rand())))
		line_height=round(64.0+(10*(1-np.random.rand())))
		return eyekit.TextBlock(lines, position=(font_size, font_size), font_face='Courier New', font_size=font_size, line_height=line_height,anchor="left")

	def _generate_line_sequence(self, passage, line_i, partial_reading=False, inherited_line_y_for_shift=None):
		x_margin, y_margin = passage.x_tl, passage.y_tl
		max_line_width = passage.width
		if partial_reading:
			start_point = np.random.randint(0, max_line_width//2) + x_margin
			end_point = np.random.randint(max_line_width//2, max_line_width) + x_margin
		else:
			start_point = x_margin
			end_point = max_line_width + x_margin
		line_X = []
		for line_n in range(passage.n_rows):
			for word_i, word in enumerate(passage.words(line_n=line_n)):
				if line_n != line_i:
					continue
				x_word_center = word.center[0]
				if x_word_center < start_point or x_word_center > end_point:
					continue
				x_value = int(np.random.triangular(word[0].x, x_word_center, word[-1].x+1))
				line_X.append(x_value)
				if word_i > 0 and np.random.random() < self.regression_within:
					x_regression = int(np.random.triangular(x_margin, word[0].x+1, word[0].x+1))
					line_X.append(x_regression)
		line_X = np.array(line_X, dtype=int) - x_margin
		line_y = passage.midlines[line_i] - y_margin
		line_Y = np.random.normal(line_y, self.noise, len(line_X))
		line_Y += line_X * self.slope
		if inherited_line_y_for_shift:
			line_Y += (inherited_line_y_for_shift - y_margin) * self.shift
		else:
			line_Y += line_y * self.shift
		line_Y = np.array(list(map(round, line_Y)), dtype=int)
		return line_X + x_margin, line_Y + y_margin, [line_i]*len(line_Y)

	def _generate_fixation_sequence(self, passage):
		X, Y, intended_I = [], [], []
		for line_i, line_y in enumerate(passage.midlines):
			line_X, line_Y, line_I = self._generate_line_sequence(passage, line_i)
			X.extend(line_X)
			Y.extend(line_Y)
			intended_I.extend(line_I)
			if line_i > 0 and np.random.random() < self.regression_between:
				rand_prev_line = int(np.random.triangular(0, line_i, line_i))
				rand_insert_point = np.random.randint(1, len(line_X))
				regression = self._generate_line_sequence(passage, rand_prev_line, partial_reading=True, inherited_line_y_for_shift=line_y)
				for rx, ry, ri in zip(*regression):
					X.insert(-rand_insert_point, rx)
					Y.insert(-rand_insert_point, ry)
					intended_I.insert(-rand_insert_point, ri)
		return np.column_stack([X, Y]), np.array(intended_I, dtype=int)

	def simulate(self, passage=None):
		'''
		Generates a fixation sequence over a passage using the distortion
		and regression parameters of the reading scenario. If no passsage
		is provided, a random one is generated according to the passage
		parameters of the reading scenario. Returns the passage, fixation
		sequence, and "correct" lines numbers.
		'''
		if passage is None:
			passage = self._generate_passage()
		fixation_XY, intended_I = self._generate_fixation_sequence(passage)
		return passage, fixation_XY, intended_I


def simulate_factor(factor, n_gradations, n_sims):
	'''
	Performs some number of simulations for each gradation in the factor
	space. A reading scenario is created for each factor value, and then,
	for each simulation, a passage and fixation sequence are generated
	and corrected by each algorithm. Results are returned as a 3D numpy
	array.
	'''
	results = np.zeros((len(core.algorithms), n_gradations, n_sims), dtype=float)
	_, (factor_min, factor_max) = core.factors[factor]
	for gradation_i, factor_value in enumerate(np.linspace(factor_min, factor_max, n_gradations)):
		print('%s = %f' % (factor, factor_value))
		reading_scenario = ReadingScenario(**{factor:factor_value})
		for sim_i in range(n_sims):
			passage, fixation_XY, intended_I = reading_scenario.simulate()
			for method_i, method in enumerate(core.algorithms):
				corrected_I = algorithms.correct_drift(method, fixation_XY, passage, return_line_assignments=True)
				matches = intended_I == corrected_I
				results[method_i][gradation_i][sim_i] = sum(matches) / len(matches)
			proportion_complete = (sim_i+1) / n_sims
			stdout.write('\r')
			stdout.write(f"[{'=' * int(100 * proportion_complete):{100}s}] {int(100 * proportion_complete)}%")
			stdout.flush()
		stdout.write('\n')
	return results

def save_sim_data(factor1,factor2,num_sims:int = 1,savedir="data/saved_data",lines_per_passage=(8, 14), max_characters_per_line=(50,130), character_spacing=(10,18), line_spacing=(15,130)):
	os.makedirs("data/saved_data",exist_ok=True)
	_, (factor1_min, factor1_max) = core.factors[factor1]
	_, (factor2_min, factor2_max) = core.factors[factor2]
	# max_characters_per_line_choices = np.arange(max_characters_per_line[0],max_characters_per_line[1],1)
	# character_spacing_choices = np.arange(character_spacing[0],character_spacing[1],1)
	# line_spacing_choices = np.arange(line_spacing[0],line_spacing[1],1)
	enum1 = np.linspace(factor2_min,factor2_max,num_sims)
	enum2 = np.linspace(factor1_min,factor1_max,num_sims)
	if factor1 == factor2:
		enum2 = [0]
		factor2 = ''
	if num_sims > 4 and factor2 != '':
		if enum1[0] == 0:
			enum1 = enum1[3:]
		if enum2[0] == 0:
			enum2 = enum2[3:]
	for factor_value2 in enum1:
		for factor_value1 in enum2:
			max_characters_per_line_choice = np.random.randint(max_characters_per_line[0],max_characters_per_line[1])
			character_spacing_choice = np.random.randint(character_spacing[0],character_spacing[1])
			line_spacing_choice = np.random.randint(line_spacing[0],line_spacing[1])

			if factor2 == '':
				fname = f"{factor1}_{factor_value1:.3f}_cs{max_characters_per_line_choice}_cSp{character_spacing_choice}_lS{line_spacing_choice}"
				factor_dict = {factor1:factor_value1}
			else:
				fname = f"{factor1}_{factor_value1:.3f}{factor2}_{factor_value2:.3f}_cs{max_characters_per_line_choice}_cSp{character_spacing_choice}_lS{line_spacing_choice}"
				factor_dict = {factor1:factor_value1,factor2:factor_value2}
			reading_scenario = ReadingScenario(
				**factor_dict,
				lines_per_passage=lines_per_passage,
				max_characters_per_line=max_characters_per_line_choice,
				character_spacing=character_spacing_choice,
				line_spacing=line_spacing_choice
			)
			try:
				passage, fixation_XY, intended_I = reading_scenario.simulate()
				if len(fixation_XY)>500:
					continue
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
				# fixation_times = pd.DataFrame([dict(start=t,end=t+average_duration) for t in np.arange(10,average_duration*len(fixation_XY),average_duration)])
				# random_vec = np.random.randn(len(fixation_XY))
				# fixation_times = fixation_times * random_vec
				corrected_fix_y_vals = slice(fixation_XY,passage.midlines,passage.line_height)
				corrected_line_nums = [passage.midlines.index(y) for y in corrected_fix_y_vals]

				pupil_sizes = pd.DataFrame([dict(pupil_size=np.random.randn()) for _ in range(len(fixation_XY))])
				pupil_sizes = pupil_sizes.applymap(lambda x: round(abs(x) * 721.0,2)) 
				fix_df = pd.DataFrame([{
					"x":fixation_XY[idx][0],
					"y":fixation_XY[idx][1],
					"assigned_line":intended_I[idx],
					"y_midline":passage.midlines[intended_I[idx]],
					"corrected_start_time":fixation_times.start.values[idx],
					"corrected_end_time":fixation_times.end.values[idx],
					"pupil_size":pupil_sizes.pupil_size.values[idx],
					"x_eyekit":fixation_XY[idx][0],
					"y_eyekit":corrected_fix_y_vals[idx],
					"line_num_eyekit":corrected_line_nums[idx],
				} for idx in range(len(fixation_XY))])

						
				trial = dict(
					filename = str(fname),
					y_midline = passage.midlines,
					y_char_unique = passage.midlines,
					num_char_lines = passage.n_lines,
					y_diff=passage.line_height,
					trial_id=fname,
					text = passage.text,
					display_coords = passage.box,
					font_size = passage.font_size,
					font= passage.font_face,
					line_heights = passage.line_height,
					chars_list= [
						{
							"char_xmin" : x.x_tl,
							"char_ymin" : x.y_tl,
							"char_xmax" : x.x_br,
							"char_ymax" : x.y_br,
							"char_x_center" : x.x,
							"char_y_center" : x.y,
							"char":x.display_text,
							"assigned_line": passage.midlines.index(x.y)
						} for x in passage.characters(alphabetical_only=False)
					],
					words_list= [
						{
							"word_xmin" : x.x_tl,
							"word_ymin" : x.y_tl,
							"word_xmax" : x.x_br,
							"word_ymax" : x.y_br,
							"word_x_center" : x.x,
							"word_y_center" : x.y,
							"word":x.display_text,
							"assigned_line": passage.midlines.index(x.y)
						} for x in passage.words(alphabetical_only=False)
					]
				)
				# fix_df.to_csv(f"{savedir}/fixations_{fname}.csv")
				fix_df.to_csv(f"{savedir}/{fname}_fixations.csv")
				
				with open(f"{savedir}/{fname}_trial.json",'w') as f:
					json_string = json.dumps(trial,indent=4)
					f.write(json_string)
				# eyekit.io.save(passage,f"{savedir}/passage_{fname}.json")
			except Exception as e:
				print(e)
		# print(factor1," Done")
	# print(factor2," Done")


if __name__ == '__main__':

	# import argparse
	# parser = argparse.ArgumentParser()
	# parser.add_argument('factor', action='store', type=str, help='factor to simulate')
	# parser.add_argument('output_dir', action='store', type=str, help='directory to write results to')
	# parser.add_argument('--n_gradations', action='store', type=int, default=50, help='number of gradations in factor')
	# parser.add_argument('--n_sims', action='store', type=int, default=100, help='number of simulations per gradation')
	# args = parser.parse_args()

	factor = "all"
	n_gradations = 4
	n_sims = 2
	lines_per_passage = (12,14)
	drive = ["/media/fssd","F:/","../.."][-1]
	output_dir = f"{drive}/pydata/Eye_Tracking/Simulation_data_{factor}/processed_data"
	pl.Path(output_dir).parent.mkdir(exist_ok=True)
	pl.Path(output_dir).mkdir(exist_ok=True)
	do_save_sim_data = True
	do_sim_correction = False
	factors_available = list(core.factors.keys())
	print(f"Factors available {factors_available}")
	#['noise', 'slope', 'shift', 'regression_within', 'regression_between']
	factors_available.remove("slope")
	# factors_available = [factor]
	print(f"Running factors {factors_available}")
	if do_save_sim_data:
		for factor2_idx,_ in tqdm(enumerate(factors_available),desc="outer"):

			for factor1 in tqdm(factors_available,desc=f"Factor {factors_available[factor2_idx]}"):
				factor2 = factors_available[factor2_idx]
				if factor2 == factor1:
					if factor2_idx == len(factors_available)-1:
						factor2 = factors_available[factor2_idx-1]
					else:
						factor2 = factors_available[factor2_idx+1]
				save_sim_data(factor1,factor2,n_sims,savedir=output_dir,lines_per_passage=lines_per_passage)

	if do_sim_correction:
		results = simulate_factor(factor, n_gradations, n_sims)
		with open('%s/%s.pkl' % (output_dir, factor), mode='wb') as file:
			pickle.dump(results, file)
