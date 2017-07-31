#!/usr/bin/env python3

import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import os
from random import shuffle

fonts = [os.path.join('/Library/Fonts/', x) for x in os.listdir('/Library/Fonts') if x.endswith('.ttf')]

bad_fonts = [19, 33, 35, 44, 47, 51, 57, 58, 87, 88, 89, 90, 91]
bad_fonts.reverse()
for x in bad_fonts:
	del fonts[x]

#print(fonts)

letters = 'abcdefghijklmnopqrstuvwxyz'

for letter in letters:
	if not os.path.exists('train/' + letter):
    		os.makedirs('train/' + letter)
	if not os.path.exists('validate/' + letter):
		os.makedirs('validate/' + letter)


font_split = int(len(fonts) * 3 * 0.8)

for letter in letters:
	count = 0
	train_count = 0
	for x_off in range(3):
		shuffle(fonts)
		for x in range(len(fonts)):
			try:
				img = Image.new("RGB", (32, 32))
				font = ImageFont.truetype(fonts[x], 28)
				draw = ImageDraw.Draw(img)
				draw.text((x_off, 0), letter, (255,255,255), font=font)

				if count < font_split:
					num = str(count * 2)
					while len(num) < 3:
						num = '0' + num
					img.save("train/" + letter + '/' + letter + num + ".png")
					train_count += 1
				else:
					num = str((count - train_count) * 2)
					while len(num) < 3:
						num = '0' + num
					img.save('validate/' + letter + '/' + letter + num + '.png')

				img = Image.new("RGB", (32, 32))
				font = ImageFont.truetype(fonts[x], 28)
				draw = ImageDraw.Draw(img)
				draw.text((x_off, 0), letter.upper(), (255,255,255), font=font)

				if count < font_split:
					num = str(count * 2 + 1)
					while len(num) < 3:
						num = '0' + num
					img.save("train/" + letter + '/' + letter + num + ".png")
				else:
					num = str((count - train_count) * 2 + 1)
					while len(num) < 3:
						num = '0' + num
					img.save('validate/' + letter + '/' + letter + num + '.png')
				count += 1
			except OSError:
				print('Bad font')
