#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def generator(file_name):
    for line in open(file_name, 'r'):
        yield tuple(map(float, line.split()))

def to_list(file_name):
    return list(map(list, zip(*generator(file_name))))


label = ['IBM-M1', 'IBM-M2', 'IBM-M2-Rand(1)', 'IBM-M2-Rand(2)', 'IBM-M2-Rand(3)', 'IBM-M2-Uniform', 'IBM-M1-AddN', 'IBM-M1-SmoothHeavyNull', 'IBM-M1-AllImprove']
color = ['red', 'green', 'blue', 'black', 'yellow', 'orange', 'pink', 'grey', 'cyan']
perplexity = to_list(sys.argv[1])
likelihood = to_list(sys.argv[2])
recall = to_list(sys.argv[3])
precision = to_list(sys.argv[4])
aer = to_list(sys.argv[5])
print(len(aer), len(precision), len(recall), len(likelihood), len(perplexity))

iterations = [i for i in range(1, len(perplexity[0]) + 1)]
    

with PdfPages('perplexity_' + sys.argv[-1]) as pdf:
    fig, ax = plt.subplots()
    for perplexity_config, color_config, label_config in zip(perplexity, color, label):
        ax.plot(iterations, perplexity_config, color=color_config, label=label_config)
    plt.xlabel('Iteration')
    plt.ylabel('Perplexity')

    legend = ax.legend(loc='upper right', ncol=2, shadow=True, prop={'size':9})

    plt.grid(True)
    plt.figure(figsize=(8, 6))
    pdf.savefig(fig)
    plt.close()

with PdfPages('likelihood_' + sys.argv[-1]) as pdf:
    fig, ax = plt.subplots()
    for likelihood_config, color_config, label_config in zip(likelihood, color, label):
        ax.plot(iterations, likelihood_config, color=color_config, label=label_config)
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')

    legend = ax.legend(loc='lower right', ncol=2, shadow=True, prop={'size':9})

    plt.grid(True)
    plt.figure(figsize=(8, 6))
    pdf.savefig(fig)
    plt.close()

with PdfPages('recall_' + sys.argv[-1]) as pdf:
    fig, ax = plt.subplots()
    for recall_config, color_config, label_config in zip(recall, color, label):
        ax.plot(iterations, recall_config, color=color_config, label=label_config)
    plt.xlabel('Iteration')
    plt.ylabel('Recall')

    legend = ax.legend(loc='lower right', ncol=2, shadow=True, prop={'size':9})

    plt.grid(True)
    plt.figure(figsize=(8, 6))
    pdf.savefig(fig)
    plt.close()

with PdfPages('precision_' + sys.argv[-1]) as pdf:
    fig, ax = plt.subplots()
    for precision_config, color_config, label_config in zip(precision, color, label):
        ax.plot(iterations, precision_config, color=color_config, label=label_config)
    plt.xlabel('Iteration')
    plt.ylabel('Precision')

    legend = ax.legend(loc='lower right', ncol=2, shadow=True, prop={'size':9})


    plt.grid(True)
    plt.figure(figsize=(8, 6))
    pdf.savefig(fig)
    plt.close()

with PdfPages('aer_' + sys.argv[-1]) as pdf:
    fig, ax = plt.subplots()
    for aer_config, color_config, label_config in zip(aer, color, label):
        ax.plot(iterations, aer_config, color=color_config, label=label_config)
    plt.xlabel('Iteration')
    plt.ylabel('AER')

    legend = ax.legend(loc='center right', ncol=2, shadow=True, prop={'size':9})

    plt.grid(True)
    plt.figure(figsize=(8, 6))
    pdf.savefig(fig)
    plt.close()
