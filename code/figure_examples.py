import eyekit

eyekit.vis.set_default_font('Helvetica Neue', 8)

data = eyekit.io.read('../data/fixations/sample.json')
passages = eyekit.io.read('../data/passages.json')

adult_fixation_sequence = data['trial_5']['fixations']  #   8, 1B
child_fixation_sequence = data['trial_30']['fixations'] # 204, 4A

adult = eyekit.vis.Image(1920, 1080)
adult.draw_text_block(passages['1B'], color='gray')
adult.draw_fixation_sequence(adult_fixation_sequence)
adult.set_caption('Reading trial by an adult')

child = eyekit.vis.Image(1920, 1080)
child.draw_text_block(passages['4A'], color='gray')
child.draw_fixation_sequence(child_fixation_sequence)
child.set_caption('Reading trial by a child')

fig = eyekit.vis.Figure(1, 2)
fig.add_image(adult)
fig.add_image(child)
fig.set_crop_margin(3)
fig.save('../visuals/illustration_examples.pdf', 174)
# fig.save('../manuscript/figs/fig01_double_column.eps', 174)
