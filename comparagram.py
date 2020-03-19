import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

images = []

def load_imgs():
	for i in range(12):
		img = cv2.imread(f"averaged/v{i+1}.jpg")
		rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		rgb = rgb.transpose(2,0,1).reshape(3,-1)
		images.append(rgb)


def response_func(q, s, a, c):
	return (1/(1+(np.exp(-a) * q)) ** c) * s

# Make the scatter plot
def plot():
	if not os.path.exists('plots'):
		os.mkdir('plots')
	for i in range(11):
		fig, ax = plt.subplots()
		ax.scatter(images[i][0], images[i+1][0],color='r', alpha = 0.3, edgecolors='none', label='r channel', s=8)
		ax.scatter(images[i][1], images[i+1][1],color='g', alpha = 0.3, edgecolors='none', label='g channel', s=8)
		ax.scatter(images[i][2], images[i+1][2],color='b', alpha = 0.3, edgecolors='none', label='b channel', s=8)
		ax.legend(loc='upper left')
		plt.savefig(f"plots/v{i}_v{i+1}.png")
		plt.cla()

# Fit the response function into the plot
def fit_response_func():
	if not os.path.exists('plots_fit'):
		os.mkdir('plots_fit')
	for i in range(10,11):
		fig, ax = plt.subplots()
		ax.scatter(images[i][0], images[i+1][0],color='r', alpha = 0.3, edgecolors='none', label='r channel', s=8)
		ax.scatter(images[i][1], images[i+1][1],color='g', alpha = 0.3, edgecolors='none', label='g channel', s=8)
		ax.scatter(images[i][2], images[i+1][2],color='b', alpha = 0.3, edgecolors='none', label='b channel', s=8)

		#plot line of fit for r
		x = images[i][0]
		y = images[i+1][0]
		popt, pcov = curve_fit(response_func, x, y)
		plt.plot(x, response_func(x, *popt), 'r-',
				 label='fit: s=%5.3f, a=%5.3f, c=%5.3f' % tuple(popt))

		x = images[i][1]
		y = images[i+1][1]
		popt, pcov = curve_fit(response_func, x, y)
		plt.plot(x, response_func(x, *popt), 'g-',
				 label='fit: s=%5.3f, a=%5.3f, c=%5.3f' % tuple(popt))

		x = images[i][2]
		y = images[i+1][2]
		popt, pcov = curve_fit(response_func, x, y)
		plt.plot(x, response_func(x, *popt), 'b-',
				 label='fit: s=%5.3f, a=%5.3f, c=%5.3f' % tuple(popt))

		ax.legend(loc='upper left')
		plt.savefig(f"plots_fit/v{i}_v{i+1}.png")
		plt.cla()


def fit_linear_line():
	if not os.path.exists('plots_linear_fit'):
		os.mkdir('plots_linear_fit')
	for i in range(11):
		fig, ax = plt.subplots()
		ax.scatter(images[i][0], images[i + 1][0], color='r', alpha=0.3, edgecolors='none', label='r channel', s=8)
		ax.scatter(images[i][1], images[i + 1][1], color='g', alpha=0.3, edgecolors='none', label='g channel', s=8)
		ax.scatter(images[i][2], images[i + 1][2], color='b', alpha=0.3, edgecolors='none', label='b channel', s=8)
		mr, br = np.polyfit(images[i][0], images[i + 1][0], 1)
		plt.plot(images[i][0], mr * images[i][0] + br, 'r-', label='fit: mr=%5.3f' % mr)
		mg, bg = np.polyfit(images[i][1], images[i + 1][1], 1)
		plt.plot(images[i][1], mg * images[i][1] + bg, 'g-', label='fit: mg=%5.3f' % mg)
		mb, bb = np.polyfit(images[i][2], images[i + 1][2], 1)
		plt.plot(images[i][2], mb * images[i][2] + bb, 'b-', label='fit: mb=%5.3f' % mb)
		ax.legend(loc='upper left')
		plt.savefig(f"plots_linear_fit/v{i}_v{i + 1}.png")
		plt.cla()

load_imgs()
fit_linear_line()




