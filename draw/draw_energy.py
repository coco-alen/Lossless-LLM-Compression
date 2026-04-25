import torch
import matplotlib.pyplot as plt
import time
import matplotlib.ticker as mtick
import seaborn as sns

sns.set_palette("colorblind")

def draw_picture_flops(matrix_shape, data1, data2, data3, filename):
    fig, ax = plt.subplots(figsize=(9.5, 4))
    bar_width = 0.25
    
    r1 = range(len(matrix_shape))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    ax.bar(r1, data1, width=bar_width, edgecolor='white', color=sns.color_palette()[0], hatch="x", label='TVM FakeShift')
    ax.bar(r2, data2, width=bar_width, edgecolor='white', color=sns.color_palette()[1], hatch="//", label='TVM MatMul')
    ax.bar(r3, data3, width=bar_width, edgecolor='white', color=sns.color_palette()[2], hatch="\\", label=r'$\bf TVM$ $\bf MatShift$ $\bf (Ours)$')
    
    # plt.xticks(r2, matrix_shape, rotation=30, fontsize=30)
    # plt.yticks(fontsize=60)
    
    # plt.legend(loc='upper left', fontsize=60)

    # font = {'size' : 60}
    # plt.xlabel("Matrix size (batchSize,M,N,K)",font)
    # plt.ylabel('Latency(ms)', font)
    # # plt.title('Comparison of FP16 MM, HQ and LSS operator',fontsize=60)

    # Add labels and title
    # ax.set_xlabel('Matrix Dimensions', fontsize=18, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=18, fontweight='bold')
    legend = ax.legend(loc='upper left', ncol=3, bbox_to_anchor=(0.05, 1.05), 
                fontsize=14, columnspacing=1.2,
                handletextpad=0.5, handlelength=2)
    ax.tick_params(axis='both', labelsize=15, width=0)
    plt.xticks(r2, matrix_shape, rotation='horizontal', fontsize=12)
    plt.yticks(fontsize=14) #, weight='bold')
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))
    ax.set_ylim(0, 2.99)

    ax.grid(axis='y', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # i = 0
    for text in legend.get_texts():
        # text.set_fontstyle("italic")
        # i += 1
        # print(i, text)
        # if i > 2:
        #     text.set_fontweight("bold")
        text.set_fontsize(14)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    for dir in ["top", "bottom", "right", "left"]:
        ax.spines[dir].set_linewidth(2)
    
    plt.savefig(filename + '.pdf', bbox_inches='tight')


### Shift

shapeList = [[32, 3136, 64, 512],
            [32, 3136, 512, 64],
            [32, 784, 128, 1024],
            [32, 784, 1024, 128],
            [32, 196, 320, 1280],
            [32, 196, 1280, 320],
            [32, 50, 512, 2048],
            [32, 50, 2048, 512]]

shapeListStr = ["input:" + str([32, 3136, 64])+ "\n" + "weight:" + str([512, 64]),
            "input:" + str([32, 3136, 512])+ "\n" + "weight:" + str([64, 512]),
            "input:" + str([32, 784, 128])+ "\n" + "weight:" + str([1024, 128]),
            "input:" + str([32, 784, 1024])+ "\n" + "weight:" + str([128, 1024]),
            "input:" + str([32, 196, 320])+ "\n" + "weight:" + str([1280, 320]),
            "input:" + str([32, 196, 1280])+ "\n" + "weight:" + str([320, 1280]),
            "input:" + str([32, 50, 512])+ "\n" + "weight:" + str([2048, 512]),
            "input:" + str([32, 50, 2048])+ "\n" + "weight:" + str([512, 2048])]

tvmShiftInt32Time = [0.59044, 0.63438, 0.618137, 0.68529, 0.538836, 0.5340, 0.29725, 0.39330]
tvmFakeShiftFloat32Time = [1.96683, 1.96364, 2.02737, 2.6143, 1.72853, 1.96642, 1.06968, 1.343491]
tvmMult3Float32Time = [0.73340, 0.78623, 0.706168, 0.82368, 0.5599, 0.56861, 0.4247, 0.4225]

# select 1, 3, 5, 2, 4, 6
shapeListStr = [
    "[B, K, M, N] = " + "\n" + "[32, 64, 3136, 512]",
    "[B, K, M, N] = " + "\n" + "[32, 128, 784, 1024]",
    "[B, K, M, N] = " + "\n" + "[32, 512, 3136, 64]",
    "[B, K, M, N] = " + "\n" + "[32, 1024, 784, 128]",
    # "[B, K, M, N] = " + "\n" + "[32, 1280, 196, 320]",
]

tvmShiftInt32Time = [0.59044, 0.618137, 0.63438, 0.68529] #, 0.5340]
tvmFakeShiftFloat32Time = [1.96683, 2.02737, 1.96364, 2.6143] #, 1.96642]
tvmMult3Float32Time = [0.73340, 0.706168, 0.78623, 0.82368] #, 0.56861]

filename = "matshift"

draw_picture_flops(
    shapeListStr,
    tvmFakeShiftFloat32Time,
    tvmMult3Float32Time,
    tvmShiftInt32Time,
    filename
)