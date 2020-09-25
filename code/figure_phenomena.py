import eyekit
import simulation
import globals


def run_and_visualize(passage, label, **params):
	reading_scenario = simulation.ReadingScenario(**params)
	_, fixation_sequence, _ = reading_scenario.simulate(passage)
	fixation_sequence = eyekit.FixationSequence(fixation_sequence)
	diagram = eyekit.Image(1920, 1080)
	diagram.draw_text_block(passage, color='gray')
	diagram.draw_fixation_sequence(fixation_sequence, color='black')
	diagram.set_caption(label)
	return diagram


passage = eyekit.TextBlock(globals.lorem_ipsum_text, position=(360, 161), font_face='Courier New', font_size=26.667, line_height=64)

fig = eyekit.Figure(5, 1)
fig.add_image(run_and_visualize(passage, 'Noise', noise=20.0))
fig.add_image(run_and_visualize(passage, 'Slope (downward)', noise=2.0, slope=0.1))
fig.add_image(run_and_visualize(passage, 'Shift (downward)', noise=2.0, shift=0.2))
fig.add_image(run_and_visualize(passage, 'Within-line regression', noise=2.0, regression_within=0.5))
fig.add_image(run_and_visualize(passage, 'Between-line regression', noise=2.0, regression_between=0.5))
fig.set_crop_margin(5)
fig.set_caption_font('Helvetica')
fig.save('../visuals/_illustration_phenomena.pdf', 83)
# fig.save('../manuscript/figs/fig04_single_column.eps', 83)
