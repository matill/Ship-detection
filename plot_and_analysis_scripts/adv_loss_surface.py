import numpy as np
import matplotlib.pyplot as plt

RESOLUTION = 300
LINSPACE_MAX = 1.5
NUM_CONTOURS = 40

LINE_COLOR = "black"
VECTOR_COLOR = "#B85450"
LINE_WIDTH = 0.7
VECTOR_DOT_SIZE = 0.015

THETA_TRUE_DEGREES = 30
LAMBDA_1 = 0.5
LAMBDA_2 = 1.5


theta_true_radians = THETA_TRUE_DEGREES * 2 * 3.14159 / 360
y_true = np.array([np.sin(theta_true_radians), np.cos(theta_true_radians)])
print("y_true", y_true)

xlist = np.linspace(-LINSPACE_MAX, LINSPACE_MAX, RESOLUTION)
ylist = np.linspace(-LINSPACE_MAX, LINSPACE_MAX, RESOLUTION)

X, Y = np.meshgrid(xlist, ylist)

print(X.shape)
print(Y.shape)

X_Y_LEN = np.sqrt((X ** 2) + (Y ** 2))
X_NORM = X / X_Y_LEN
Y_NORM = Y / X_Y_LEN


loss_1_a = (X_NORM - y_true[0]) ** 2
loss_1_b = (Y_NORM - y_true[1]) ** 2
loss_1 = loss_1_a + loss_1_b
print("loss_1_a", loss_1_a.shape)  
print("loss_1_b", loss_1_b.shape)  
print("loss_1", loss_1.shape)  


loss_2 = (X - y_true[0]) ** 2 + (Y - y_true[1]) ** 2


loss_3_a = X * y_true[0] + Y * y_true[1]
loss_3_b = X * y_true[1] - Y * y_true[0]
loss_3 = LAMBDA_1 * (loss_3_a - 1) ** 2 + LAMBDA_2 * loss_3_b ** 2




Z = np.sqrt(X**2 + Y**2)
# print(Z)



def plot_loss(loss, name, ax):
    # fig = plt.figure(figsize=(6,5))
    # left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    # ax = fig.add_axes([left, bottom, width, height]) 


    # Hide axes 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # cp = ax.contour(X, Y, loss)
    # cp = ax.contour(X, Y, loss, NUM_CONTOURS)
    cp = ax.contourf(X, Y, loss, NUM_CONTOURS)
    ax.clabel(cp, inline=False, fontsize=10)
    if name:
        ax.set_title(name)

    # Unit circle and "cross hair"
    ax.add_artist(plt.Circle((0, 0), 1, fill=False, color=LINE_COLOR, linewidth=LINE_WIDTH))
    ax.add_artist(plt.Line2D([-LINSPACE_MAX, LINSPACE_MAX], [0, 0], color=LINE_COLOR, linewidth=LINE_WIDTH))
    ax.add_artist(plt.Line2D([0, 0], [-LINSPACE_MAX, LINSPACE_MAX], color=LINE_COLOR, linewidth=LINE_WIDTH))

    # Vector
    ax.add_artist(plt.Line2D([0, y_true[0]], [0, y_true[1]], color=VECTOR_COLOR, linewidth=LINE_WIDTH))
    ax.add_artist(plt.Circle(y_true, VECTOR_DOT_SIZE, fill=True, color=VECTOR_COLOR, linewidth=LINE_WIDTH))
    # ax.set_xlabel('x_1')
    # ax.set_ylabel('x_2')
    # plt.show()



from matplotlib.gridspec import GridSpec

# fig, axs = plt.subplots(1, 3, sharey=True, squeeze=False)

# Masters thesis figure
fig = plt.figure(figsize=(15, 5))
gs = GridSpec(nrows=1, ncols=3)
ax0 = fig.add_subplot(gs[:, 0])
ax1 = fig.add_subplot(gs[:, 1])
ax2 = fig.add_subplot(gs[:, 2])
plot_loss(loss_1, "Loss (unit)", ax0)
plot_loss(loss_2, "Loss (mse)", ax1)
plot_loss(loss_3, "Loss (mse-w)", ax2)
plt.savefig("out/adv-loss-surfaces.png", bbox_inches='tight')
plt.close("all")

# Paper figure
fig = plt.figure(figsize=(5, 5))
gs = GridSpec(nrows=1, ncols=1)
ax0 = fig.add_subplot(gs[:, 0])
plot_loss(loss_3, None, ax0)
# plot_loss(loss_2, "Loss (mse)", ax1)
# plot_loss(loss_3, "Loss (mse-w)", ax2)
plt.savefig("out/adv-loss-surfaces-paper.png", bbox_inches='tight')
plt.close("all")

# plt.savefig("test.png")
# plt.show()
