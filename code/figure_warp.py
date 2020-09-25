import eyekit
import algorithms
import globals

data = eyekit.io.read('../data/fixations/sample.json')
passages = eyekit.io.read('../data/passages.json')

for fixation in data['trial_5']['fixations']:
	fixation.duration = 100
corrected_fixation_sequence, solution = algorithms.correct_drift('warp', data['trial_5']['fixations'].XYarray(), passages['1B'], return_solution=True)
corrected_fixation_sequence = eyekit.FixationSequence(corrected_fixation_sequence)
word_XY = passages['1B'].word_centers

diagram = eyekit.Image(1920, 1080)
diagram.draw_text_block(passages['1B'], color='gray')
diagram.draw_fixation_sequence(eyekit.FixationSequence(word_XY), color=globals.illustration_colors[1])
diagram.draw_fixation_sequence(data['trial_5']['fixations'], color=globals.illustration_colors[2])
for fixation, mapped_words in zip(data['trial_5']['fixations'], solution):
	for word_i in mapped_words:
		word_x, word_y = word_XY[word_i]
		diagram.draw_line(fixation.xy, (word_x, word_y), 'black', dashed=True)

fig = eyekit.Figure()
fig.add_image(diagram)
fig.set_crop_margin(3)
fig.set_auto_letter(False)
fig.save('../visuals/_illustration_warp.pdf', 83)
# fig.save('../manuscript/figs/fig03_single_column_.eps', 83)
