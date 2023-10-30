import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import *

random_seed = 0
torch.manual_seed(random_seed)

x = torch.arange(-0.5, 0.5, 0.01)
# sdf = torch.arange(-0.5, 0.5, 0.01)
pts = [-0.25, 0.25]
sdf = torch.cat([torch.arange(-0.25, 0.25, 0.01), -torch.arange(-0.25, 0.25, 0.01)], dim=0)

dsdf = torch.cat([torch.ones_like(sdf), -torch.ones_like(sdf)], dim=0)


sigma = 0.05
sigmoid_sdf = sigmoid(sdf, sigma)

# gaussian_pos = torch.arange(-0.5, 0.5, 0.1).requires_grad_(True)
gaussian_pos = torch.cat(
    [-0.25 + torch.randn(5) / 10.0, 0.25 + torch.randn(5) / 10.0], dim=0
).requires_grad_(True)
# gaussian_pos = (0.0 + torch.randn(10) / 10.0).requires_grad_(True)

gaussian_sigma = torch.rand_like(gaussian_pos).requires_grad_(True)
gaussian_amplitude = torch.rand_like(gaussian_pos).requires_grad_(True)


activate_a = torch.tanh
# activate_a = torch.sigmoid
# activate_a = torch.abs
# def activate_a(x):
#     return x


# activate_sigma = torch.exp
def activate_sigma(x):
    return x


sup_sigma = 2 * sigma
sup_mask = ((x > (pts[0] - sup_sigma)) & (x < (pts[0] + sup_sigma))) | (
    (x > (pts[1] - sup_sigma)) & (x < (pts[1] + sup_sigma))
)

gt = sdf
# gt = sigmoid_sdf
sup_gt = gt[sup_mask]
sup_x = x[sup_mask]
# sup_gt = gt
# sup_x = x
# gt = sdf
optimizer = torch.optim.Adam([gaussian_pos, gaussian_sigma, gaussian_amplitude], lr=0.1)
for iter in tqdm(range(1000)):
    pred_sigmoid_sdf = composition(
        gaussian,
        sup_x,
        gaussian_pos,
        activate_sigma(gaussian_sigma),
        activate_a(gaussian_amplitude),
    )
    loss = torch.nn.functional.mse_loss(pred_sigmoid_sdf, sup_gt)

    # pred_sigmoid_sdf_n25 = composition(
    #     gaussian, -0.25, gaussian_pos, activate_sigma(gaussian_sigma), activate_a(gaussian_amplitude)
    # )
    # loss = torch.nn.functional.mse_loss(pred_sigmoid_sdf_n25, torch.tensor(0.5))
    # pred_sigmoid_sdf_p25 = composition(
    #     gaussian, 0.25, gaussian_pos, activate_sigma(gaussian_sigma), activate_a(gaussian_amplitude)
    # )
    # loss += torch.nn.functional.mse_loss(pred_sigmoid_sdf_p25, torch.tensor(0.5))

    # pred_grad = composition(
    #     dgaussian_dx,
    #     sup_x,
    #     gaussian_pos,
    #     activate_sigma(gaussian_sigma),
    #     activate_a(gaussian_amplitude),
    # )

    # loss_eikonal = torch.nn.functional.mse_loss(pred_grad, sup_gt * (1 - sup_gt))

    # pred_grad = pred_dsdf_dx(
    #     sup_x, gaussian_pos, activate_sigma(gaussian_sigma),
    #     activate_a(gaussian_amplitude), sigma, pred_sigmoid_sdf
    # )
    # loss_eikonal = torch.nn.functional.mse_loss(pred_grad, torch.ones_like(pred_grad))
    # loss += 0.0001 * loss_eikonal
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


with torch.no_grad():
    plot_num = 4
    plot_count = 0

    # plot_count += 1
    # plt.subplot(plot_num, 1, plot_count)
    # plt.plot(x, shape(gaussian, 0.0, 0.05), label="Gaussian(0,0.05)", color="red")
    # plt.plot(x, shape(dgaussian_dx, 0.0, 0.05), label="dGaussian_dx", color="blue")
    # plt.grid()
    # plt.legend()

    plot_count += 1
    plt.subplot(plot_num, 1, plot_count)
    # https://finthon.com/matplotlib-color-list/
    plt.plot(x, sdf, label="SDF", color="black")
    plt.plot(x, torch.ones_like(sdf), label="|dSDF_dx|", color="grey")
    plt.plot(x, sigmoid_sdf, label="Sigmoid(SDF)", color="red")
    # derivative of sigmoid = sigmoid*(1-sigmoid)
    plt.plot(x, sigmoid_sdf * (1 - sigmoid_sdf), label="dSigmoid_dx)", color="blue")

    plt.xlabel("x")
    plt.grid()
    plt.legend()

    plot_count += 1
    plt.subplot(plot_num, 1, plot_count)
    # https://finthon.com/matplotlib-color-list/
    plt.plot(x, gt, label="SDF", color="red", linewidth=3.5)
    plt.plot(
        x,
        composition(
            gaussian,
            x,
            gaussian_pos,
            activate_sigma(gaussian_sigma),
            activate_a(gaussian_amplitude),
        ).detach(),
        label="PredSDF",
        color="blue",
        linewidth=2.5,
    )

    for i in range(gaussian_pos.size(0)):
        plt.plot(
            x,
            gaussian(
                gaussian_pos[i],
                activate_sigma(gaussian_sigma[i]),
                activate_a(gaussian_amplitude[i]),
                x,
            ).detach(),
            color="cyan",
            linewidth=1.0,
        )

    # show gaussian centers
    # plt.plot(gaussian_pos, activate_a(gaussian_amplitude), "o", color="green")
    plt.xlabel("x")
    plt.grid()
    plt.legend()

    plot_count += 1
    plt.subplot(plot_num, 1, plot_count)
    # derivative of sigmoid = sigmoid*(1-sigmoid)
    # plt.plot(x, sigmoid_sdf * (1 - sigmoid_sdf), label="dSigmoid_dx", color="red")
    # plt.plot(
    #     x,
    #     composition(
    #         dgaussian_dx, x, gaussian_pos, activate_sigma(gaussian_sigma), activate_a(gaussian_amplitude)
    #     ).detach(),
    #     label="PreddSigmoid_dx",
    #     color="blue",
    # )

    plt.plot(x, sigmoid(gt, sigma), label="Sigmoid(SDF)", color="red")
    plt.plot(
        x,
        sigmoid(
            composition(
                gaussian,
                x,
                gaussian_pos,
                activate_sigma(gaussian_sigma),
                activate_a(gaussian_amplitude),
            ).detach(),
            sigma,
        ),
        label="Sigmoid(PredSDF)",
        color="blue",
    )

    # for i in range(gaussian_pos.size(0)):
    #     plt.plot(
    #         x,
    #         dgaussian_dx(
    #             gaussian_pos[i],
    #             activate_sigma(gaussian_sigma[i]),
    #             activate_a(gaussian_amplitude[i]),
    #             x,
    #         ).detach(),
    #         color="cyan",
    #     )

    plt.xlabel("x")
    plt.grid()
    plt.legend()

    plot_count += 1
    plt.subplot(plot_num, 1, plot_count)
    # derivative of sigmoid = sigmoid*(1-sigmoid)
    # plt.plot(x, sigmoid_sdf * (1 - sigmoid_sdf), label="dSigmoid_dx", color="red")
    # plt.plot(
    #     x,
    #     composition(
    #         dgaussian_dx, x, gaussian_pos, activate_sigma(gaussian_sigma), activate_a(gaussian_amplitude)
    #     ).detach(),
    #     label="PreddSigmoid_dx",
    #     color="blue",
    # )

    plt.plot(x, torch.ones_like(sdf), label="|dsdf_dx|", color="red")

    plt.plot(
        x,
        composition(
            dgaussian_dx,
            x,
            gaussian_pos,
            activate_sigma(gaussian_sigma),
            activate_a(gaussian_amplitude),
        ),
        label="|PreddSDF_dx|",
        color="blue",
    )

    for i in range(gaussian_pos.size(0)):
        plt.plot(
            x,
            dgaussian_dx(
                gaussian_pos[i],
                activate_sigma(gaussian_sigma[i]),
                activate_a(gaussian_amplitude[i]),
                x,
            ).detach(),
            color="cyan",
        )

    plt.xlabel("x")
    # plt.ylim(0, 2)
    plt.grid()
    plt.legend()

    plt.show()
