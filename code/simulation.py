'''
Code for performing the fixation sequence simulations.
'''

from sys import stdout
import pickle
import numpy as np
import eyekit
import algorithms
from classic_correction_algos import slice
# import core
from tqdm.auto import tqdm
import pandas as pd
import os
import pathlib as pl
import json
from PIL import Image
import add_fonts

FACTORS = {
           'regression_within':('Probability of within-line regression', (0, 1)),
           'regression_between':('Probability of between-line regression', (0, 1)),
		   'noise':('Noise distortion', (0, 40)),
           'slope':('Slope distortion', (-0.1, 0.1)),
           'shift':('Shift distortion', (-0.2, 0.2)),
		   }
ALGORITHMS = ['attach', 'chain', 'cluster', 'compare', 'merge', 'regress', 'segment', 'slice', 'split', 'stretch', 'warp']

class ReadingScenario:

	def __init__(self, noise=0, slope=0, shift=0, regression_within=0, regression_between=0, lines_per_passage=(8, 12), max_characters_per_line=80, character_spacing=16,include_line_breaks=False,text:str=None,text_gen_cfg:dict=None):
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
		self.include_line_breaks = include_line_breaks
		self.text_gen_cfg = text_gen_cfg
		if text is not None:
			self.text = text

	def _generate_passage(self):
		n_lines = np.random.randint(self.min_lines, self.max_lines+1)
		lines = ['']
		has_paragraph_gap = False
		add_short_sentences = True if np.random.rand() < self.text_gen_cfg["short_sentence_in_trial_probability"] else False
		while len(lines) < n_lines:
			# for word in lorem.sentence().split():
			if self.text is None:
				import lorem
				words = lorem.sentence().split()
			else:
				words = self.text.split(" ")
			for word in words:
				if len(lines) > n_lines:
					break
				if add_short_sentences and len(lines[-1]) > 0 and np.random.rand() < self.text_gen_cfg["short_sentence_break_after_word_probability"] :
					lines.append(word + ' ')
				elif (len(lines[-1]) + len(word)) <= self.max_characters_per_line:
					lines[-1] += word + ' '
				elif self.include_line_breaks and len(lines) == n_lines//2 and np.random.rand() < 0.25 and not has_paragraph_gap:
					lines.append(' ')
					lines.append(word + ' ')
					has_paragraph_gap = True
				elif self.include_line_breaks and len(lines) > 1 and np.random.rand() < 0.2 and not has_paragraph_gap:
					lines.append(' ')
					lines.append(word + ' ')
					has_paragraph_gap = True
					if np.random.rand() < 0.3:
						words_last_line = lines[-1].split(' ')
						keep_len = max(int(len(words_last_line) * np.random.rand()),2)
						lines[-1] = ' '.join(words_last_line[:keep_len])
				else:
					lines.append(word + ' ')
			if len(lines) == n_lines and len(lines[-1].split()) == 1:
				# Final line only contains one word, so add in an extra word
				# because a one-word final line can be problematic for merge
				# since it cannot create sequences with one fixation.
				w = words[-1] #np.random.randint(1,len(words)+1)
				lines[-1] = f'{w} ' + lines[-1]
		font_size=round(self.text_gen_cfg["base_font_size"]+(self.text_gen_cfg["font_size_range"]*(1-np.random.rand())))
		line_height=round(self.text_gen_cfg["base_line_height"]+(self.text_gen_cfg["line_height_range"]*(1-np.random.rand())))
		return eyekit.TextBlock(lines, position=(font_size, font_size), font_face=self.text_gen_cfg["font_face"], font_size=font_size, line_height=line_height,anchor="left")

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
				if len(word) == 1:
					x_value = int(word[0].x + (np.random.rand(1)[0] - 0.5)*2)
				else:
					x_value = int(np.random.triangular(word[0].x, x_word_center, word[-1].x+1))
				line_X.append(x_value)
				if word_i > 0 and np.random.random() < self.regression_within:
					x_regression = int(np.random.triangular(x_margin, word[0].x+1, word[0].x+1))
					line_X.append(x_regression)
		line_X = np.array(line_X) - x_margin
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
			if line_i > 0 and np.random.random() < self.regression_between and len(line_X)>1:
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
	results = np.zeros((len(ALGORITHMS), n_gradations, n_sims), dtype=float)
	_, (factor_min, factor_max) = FACTORS[factor]
	for gradation_i, factor_value in enumerate(np.linspace(factor_min, factor_max, n_gradations)):
		print('%s = %f' % (factor, factor_value))
		reading_scenario = ReadingScenario(**{factor:factor_value})
		for sim_i in range(n_sims):
			passage, fixation_XY, intended_I = reading_scenario.simulate()
			for method_i, method in enumerate(ALGORITHMS):
				corrected_I = algorithms.correct_drift(method, fixation_XY, passage, return_line_assignments=True)
				matches = intended_I == corrected_I
				results[method_i][gradation_i][sim_i] = sum(matches) / len(matches)
			proportion_complete = (sim_i+1) / n_sims
			stdout.write('\r')
			stdout.write(f"[{'=' * int(100 * proportion_complete):{100}s}] {int(100 * proportion_complete)}%")
			stdout.flush()
		stdout.write('\n')
	return results

def eyekit_plot(fix_df,passage,savename):

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
	img.save(f'{savename}.png')
	im = Image.open(f'{savename}.png')
	im = im.resize((1024//2,768//2))
	im.save(f'{savename}.png')
	return 0

def save_sim_data(
		factor1,factor2,num_sims:int = 1,
		savedir="data/saved_data",
		lines_per_passage=(8, 14), 
		max_characters_per_line=(50,130), 
		character_spacing=(10,18), 
		include_line_breaks=False,
		always_apply_small_noise=False,
		max_num_fixations=500,
		do_eyekit_plot=False,
		texts:str=None,
		text_num=0,
		text_gen_cfg:dict=None
):
	os.makedirs("data/saved_data",exist_ok=True)
	_, (factor1_min, factor1_max) = FACTORS[factor1]
	_, (factor2_min, factor2_max) = FACTORS[factor2]
	enum1 = np.linspace(factor2_min,factor2_max,num_sims)
	enum2 = np.linspace(factor1_min,factor1_max,num_sims)
	if factor1 == factor2:
		enum2 = [0]
		factor2 = ''
	if num_sims > 4 and factor2 != '':
		if enum1[0] == 0 and factor2 not in ["regression_within","regression_between"]:
			enum1 = enum1[3:]
		if enum2[0] == 0 and factor1 not in ["regression_within","regression_between"]:
			enum2 = enum2[3:]
		if factor1 not in ["regression_within","regression_between"]:
			enum2 = enum2[enum2 != 0]
		if factor2 not in ["regression_within","regression_between"]:
			enum1 = enum1[enum1 != 0]

	for factor_value2 in enum1:
		for factor_value1 in enum2:
			max_characters_per_line_choice = np.random.randint(max_characters_per_line[0],max_characters_per_line[1])
			character_spacing_choice = np.random.randint(character_spacing[0],character_spacing[1])

			if factor2 == '':
				fname = f"{factor1}_{factor_value1:.3f}_cs{max_characters_per_line_choice}_cSp{character_spacing_choice}"
				factor_dict = {factor1:factor_value1}
			else:
				fname = f"{factor1}_{factor_value1:.3f}{factor2}_{factor_value2:.3f}_cs{max_characters_per_line_choice}_cSp{character_spacing_choice}"
				factor_dict = {factor1:factor_value1,factor2:factor_value2}
			if always_apply_small_noise and 'noise' not in factor_dict:
				factor_dict['noise'] = 6
			elif 'noise' in factor_dict and factor_dict['noise'] < 6:
				factor_dict['noise'] = 6

			reading_scenario = ReadingScenario(
				**factor_dict,
				lines_per_passage=lines_per_passage,
				max_characters_per_line=max_characters_per_line_choice,
				character_spacing=character_spacing_choice,
				include_line_breaks=include_line_breaks,
				text=texts[text_num],
				text_gen_cfg=text_gen_cfg,
			)
			try:
				passage, fixation_XY, intended_I = reading_scenario.simulate()
				if fixation_XY.shape[0] > max_num_fixations:
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
				pupil_sizes = pupil_sizes.map(lambda x: round(abs(x) * 721.0,2)) 
				fix_df = pd.DataFrame([{
					"x":fixation_XY[idx][0],
					"y":fixation_XY[idx][1],
					"assigned_line":intended_I[idx],
					"y_midline":passage.midlines[intended_I[idx]],
					"corrected_start_time":fixation_times.start.values[idx],
					"corrected_end_time":fixation_times.end.values[idx],
					"pupil_size":pupil_sizes.pupil_size.values[idx],
					"y_eyekit":corrected_fix_y_vals[idx],
					"line_num_eyekit":corrected_line_nums[idx],
				} for idx in range(len(fixation_XY))])

				if do_eyekit_plot:
					eyekit_plot(fix_df,passage,pl.Path(output_dir).parent.joinpath("plots").joinpath(f"eyekitPlot_{fname}"))
				unique_midlines = passage.midlines
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

				chars_list = []
				for word in words_list:
					letter_width =(word["word_xmax"] - word["word_xmin"])/ (len(word["word"])+1)
					for i_w, letter in enumerate(word["word"]):
						xmin = round(word["word_xmin"] + i_w * letter_width,2)
						xmax = round(word["word_xmin"]+ (i_w+1) * letter_width,2)
						if xmax > word["word_xmax"]:
							xmax = round(word["word_xmax"],2)
						char_dict = dict(
							char_xmin = xmin,
							char_xmax = xmax,
							char_ymin = word["word_ymin"],
							char_ymax = word["word_ymax"],
							char = letter,
						)
						
						char_dict["char_x_center"] = (char_dict["char_xmax"] - char_dict["char_xmin"])/2 + char_dict["char_xmin"]
						char_dict["char_y_center"] = (word["word_ymin"] - word["word_ymax"])/2 + word["word_ymax"]

						if i_w >= len(word["word"])+1:
							break
						char_dict["assigned_line"] = unique_midlines.index(char_dict["char_y_center"])
						chars_list.append(char_dict)

				chars_list_eyekit= [
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
					words_list=words_list,
					chars_list=chars_list,
				)

				
				fix_df.to_csv(f"{savedir}/{fname}_fixations.csv")
				
				with open(f"{savedir}/{fname}_trial.json",'w') as f:
					json_string = json.dumps(trial,indent=4)
					f.write(json_string)
				# eyekit.io.save(passage,f"{savedir}/passage_{fname}.json")
			except Exception as e:
				print(e)
			text_num += 1
			if text_num == len(texts) -1:
				text_num = 0
		# print(factor1," Done")
	# print(factor2," Done")
	return text_num

if __name__ == '__main__':

	factor = [
		"all_noLineBreaks_alwaysNoise_wikitext",
		"all_noLineBreaks_shortSent_alwaysNoise_wikitext",
		"all_noLineBreaks_shortSentEverytrial_alwaysNoise_wikitext",
		"all_noLineBreaks_shortSent_withGaps_alwaysNoise_wikitext",
	][3]
	n_gradations = 4
	n_sims = 20
	lines_per_passage = (8,14) #(12,14)
	max_characters_per_line = (50,130)
	include_line_breaks = False
	always_apply_small_noise = True
	drive = ["/media/fssd","F:/","../..","/media/ubulappi/Extreme SSD/all","H:/all"][-1]
	output_dir = f"{drive}/pydata/Eye_Tracking/Sim_{factor}/processed_data"
	text_source = f"{drive}/pydata/Text_Data_General/wikitext-2/wiki.train.tokens"
	pl.Path(output_dir).parent.mkdir(exist_ok=True)
	pl.Path(output_dir).parent.joinpath("plots").mkdir(exist_ok=True)
	pl.Path(output_dir).mkdir(exist_ok=True)
	do_save_sim_data = True
	do_eyekit_plot = False
	do_sim_correction = False
	factors_available = list(FACTORS.keys())

	text_gen_cfg = dict(
		short_sentence_in_trial_probability = 0.25, #was 0.25
		short_sentence_break_after_word_probability = 0.01,
		base_font_size=25, #was 26.667
		font_size_range=10, #was 5
		base_line_height = 64, #was 64
		line_height_range=15, # was 10
		font_face = "Consola Mono", # was Courier New
	)

	print(f"Factors available {factors_available}")
	#['noise', 'slope', 'shift', 'regression_within', 'regression_between']
	factors_available.remove("slope")
	# factors_available = [factor]
	print(f"Running factors {factors_available}")

	with open(text_source,"r",encoding="utf-8") as f:
		text = f.read()
	texts = text.split("\n \n ")
	texts = [x.replace(" .",".").replace("<unk>","").replace("\n","").replace("@-@ ","-").replace("@.@ ",".").replace("@,@ ",",") for x in texts if len(x)>100 and "= =" not in x and "Note :" not in x]
	texts = [x.encode('ascii',errors='ignore').decode() for x in texts]
	texts = [x.replace(" .",".").replace(" ,",",").replace(" )",")").replace(" (","(").replace(" :",":").replace(" ;",";").replace(" '","'").replace(' "','"') for x in texts]
	texts = [x.replace("( ","(").replace("  "," ").replace("' ","'").replace('"','" ').replace(" ]","]").replace(" [","[").replace("' ","'").replace('" ','"') for x in texts]
	texts = [x.replace("$ ","$").replace(" ?","?").replace(" !","!").replace(" ,",",") for x in texts]
	np.random.seed(42)
	np.random.shuffle(texts)
	text_num = 0
	if do_save_sim_data:
		for factor2_idx,_ in tqdm(enumerate(factors_available),desc="outer"):

			for factor1 in tqdm(factors_available,desc=f"Factor {factors_available[factor2_idx]}"):
				factor2 = factors_available[factor2_idx]
				if factor2 == factor1:
					if factor2_idx == len(factors_available)-1:
						factor2 = factors_available[factor2_idx-1]
					else:
						factor2 = factors_available[factor2_idx+1]
				
				text_num = save_sim_data(
					factor1,
					factor2,
					n_sims,
					savedir=output_dir,
					lines_per_passage=lines_per_passage,
					max_characters_per_line=max_characters_per_line,
					include_line_breaks=include_line_breaks,
					always_apply_small_noise=always_apply_small_noise,
					texts=texts,
					do_eyekit_plot=do_eyekit_plot,
					text_num=text_num,
					text_gen_cfg=text_gen_cfg
				)

	if do_sim_correction:
		results = simulate_factor(factor, n_gradations, n_sims)
		with open('%s/%s.pkl' % (output_dir, factor), mode='wb') as file:
			pickle.dump(results, file)
