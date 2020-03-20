import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

images = []
m_lst = []


def load_imgs():
    for i in range(12):
        img = cv2.imread(f"averaged/v{i + 1}_averaged.png")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = rgb.transpose(2, 0, 1).reshape(3, -1)
        images.append(rgb)


def response_func1(q, s, a, c):
    return (1 / (1 + (np.exp(-a) * q)) ** c) * s


def response_func2(q, a, b, g):
    return a + b * q ** g


def comparametric_equation(q, g):
    return q * 2 ** g


# Make the scatter plot
def plot():
    if not os.path.exists('plots'):
        os.mkdir('plots')

    for i in range(11):
        fig, ax = plt.subplots()
        ax.scatter(images[i][0], images[i + 1][0], color='r', alpha=0.3, edgecolors='none',
                   label='r channel', s=8)
        ax.scatter(images[i][1], images[i + 1][1], color='g', alpha=0.3, edgecolors='none',
                   label='g channel', s=8)
        ax.scatter(images[i][2], images[i + 1][2], color='b', alpha=0.3, edgecolors='none',
                   label='b channel', s=8)
        ax.legend(loc='lower left')
        plt.savefig(f"plots/v{i}_v{i + 1}.png")
        plt.cla()


# Fit the response function into the plot
def fit_response_func():
    if not os.path.exists('plots_fit'):
        os.mkdir('plots_fit')
    for i in range(10, 11):
        fig, ax = plt.subplots()
        ax.scatter(images[i][0], images[i + 1][0], color='r', alpha=0.3, edgecolors='none',
                   label='r channel', s=8)
        ax.scatter(images[i][1], images[i + 1][1], color='g', alpha=0.3, edgecolors='none',
                   label='g channel', s=8)
        ax.scatter(images[i][2], images[i + 1][2], color='b', alpha=0.3, edgecolors='none',
                   label='b channel', s=8)

        # plot line of fit for r
        x = images[i][0]
        y = images[i + 1][0]

        popt, pcov = curve_fit(response_func2, x, y)
        plt.plot(x, response_func2(x, *popt), 'r-',
                 label='fit: a=%5.3f, b=%5.3f, g=%5.3f' % tuple(popt))

        x = images[i][1]
        y = images[i + 1][1]
        popt, pcov = curve_fit(response_func2, x, y)
        plt.plot(x, response_func2(x, *popt), 'g-',
                 label='fit: a=%5.3f, b=%5.3f, g=%5.3f' % tuple(popt))

        x = images[i][2]
        y = images[i + 1][2]
        popt, pcov = curve_fit(response_func2, x, y)
        plt.plot(x, response_func2(x, *popt), 'k-',
                 label='fit: a=%5.3f, b=%5.3f, g=%5.3f' % tuple(popt))

        ax.legend(loc='lower left')
        plt.savefig(f"plots_fit_func/v{i}_v{i + 1}.png")
        plt.cla()


def fit_linear_line():
    if not os.path.exists('plots_fit_linear'):
        os.mkdir('plots_fit_linear')
    for i in range(11):
        fig, ax = plt.subplots()
        ax.scatter(images[i][0], images[i + 1][0], color='r', alpha=0.3, edgecolors='none',
                   label='r channel', s=8)
        ax.scatter(images[i][1], images[i + 1][1], color='g', alpha=0.3, edgecolors='none',
                   label='g channel', s=8)
        ax.scatter(images[i][2], images[i + 1][2], color='b', alpha=0.3, edgecolors='none',
                   label='b channel', s=8)
        mr, br = np.polyfit(images[i][0], images[i + 1][0], 1)
        plt.plot(images[i][0], mr * images[i][0] + br, 'r-', label='fit: mr=%5.3f' % mr)
        mg, bg = np.polyfit(images[i][1], images[i + 1][1], 1)
        plt.plot(images[i][1], mg * images[i][1] + bg, 'g-', label='fit: mg=%5.3f' % mg)
        mb, bb = np.polyfit(images[i][2], images[i + 1][2], 1)
        plt.plot(images[i][2], mb * images[i][2] + bb, 'k-', label='fit: mb=%5.3f' % mb)
        ax.legend(loc='lower left')
        plt.savefig(f"plots_fit_linear/v{i}_v{i + 1}.png")
        plt.cla()
        m_lst.append((mr, mg, mb))


def rgb_comparagram():
    load_imgs()
    if not os.path.exists('comparagrams'):
        os.mkdir('comparagrams')
    comparator = 3
    for img_index in range(12 - comparator):
        r1_component = images[img_index][0]
        g1_component = images[img_index][1]
        b1_component = images[img_index][2]

        r2_component = images[img_index + comparator][0]
        g2_component = images[img_index + comparator][1]
        b2_component = images[img_index + comparator][2]

        base_image = np.zeros([256, 256, 3])
        for r1, g1, b1, r2, g2, b2 in zip(r1_component, g1_component, b1_component, r2_component,
                                          g2_component, b2_component):
            # Image is BGR format.
            base_image[b2, b1, 0] += 1
            base_image[g2, g1, 1] += 1
            base_image[r2, r1, 2] += 1

        masked_array = np.ma.masked_greater(base_image, 255)
        base_image[masked_array.mask] = 255
        base_image = base_image.astype(np.uint8)
        base_image = cv2.flip(base_image, 0)
        cv2.imwrite(f"comparagrams/color_img_{img_index:02d}_{img_index + comparator:02d}.jpg",
                    base_image)


def plot_k2(threshold):
    img = cv2.imread(f"comparagrams/k_2_composite_comparagram.jpg")
    img = cv2.flip(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if not os.path.exists('plots_k2'):
        os.mkdir('plots_k2')

    dots = []
    fig, ax = plt.subplots()
    for col in range(256):
        for row in range(256):
            if img[col][row][0] >= threshold:
                ax.scatter(row, col, color='r', alpha=0.3, edgecolors='none', s=8)
                if not [col, row] in dots:
                    dots.append([col, row])

            if img[col][row][1] >= threshold:
                ax.scatter(row, col, color='g', alpha=0.3, edgecolors='none', s=8)
                if not [col, row] in dots:
                    dots.append([col, row])

            if img[col][row][2] >= threshold:
                ax.scatter(row, col, color='b', alpha=0.3, edgecolors='none', s=8)
                if not [col, row] in dots:
                    dots.append([col, row])

    dots = np.asarray(dots)
    print(dots.shape)
    print(dots)
    x = dots[:, 1]
    y = dots[:, 0]
    popt, pcov = curve_fit(comparametric_equation, x, y)
    plt.plot(x, comparametric_equation(x, *popt), 'k-',
             label='fit: g=%5.3f' % tuple(popt))
    plt.xlim([0, 255])
    plt.ylim([0, 255])
    ax.legend(loc='lower left')
    plt.show()


def full_composite_comparagram():
    comparator = 3
    pass_name = "third_pass"
    for i in range(12 - comparator):
        img = cv2.imread(f"comparagrams/{pass_name}/color_img_{i:02d}_{i + comparator:02d}.jpg")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(rgb)

    base_image = np.zeros([256, 256, 3])
    for img_index in range(12 - comparator):
        r1_component = images[img_index][:, :, 0]
        g1_component = images[img_index][:, :, 1]
        b1_component = images[img_index][:, :, 2]

        base_image[:, :, 0] += b1_component
        base_image[:, :, 1] += g1_component
        base_image[:, :, 2] += r1_component

    masked_array = np.ma.masked_greater(base_image, 255)
    base_image[masked_array.mask] = 255
    base_image = base_image.astype(np.uint8)
    cv2.imwrite(f"comparagrams/color_img_{pass_name}.jpg", base_image)


# full_composite_comparagram()
plot_k2(100)
