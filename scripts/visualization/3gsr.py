import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

random_seed = 0
torch.manual_seed(random_seed)

x = torch.arange(-0.5, 0.5, 0.01)
# sdf = torch.arange(-0.5, 0.5, 0.01)
sdf = torch.cat([torch.arange(-0.5, 0.5, 0.02), -torch.arange(-0.5, 0.5, 0.02)], dim=0)

dsdf = torch.cat([torch.ones_like(sdf), -torch.ones_like(sdf)], dim=0)


def sigmoid(x, sigma=1):
    return 1 / (1 + torch.exp(-x / sigma))


def inv_sigmoid(x, sigma=1):
    return sigma * torch.log(x / (1 - x))


sigma = 5.0
sigmoid_sdf = sigmoid(sdf, sigma)

# gaussian_pos = torch.arange(-0.5, 0.5, 0.1).requires_grad_(True)
gaussian_pos = torch.cat(
    [-0.25 + torch.randn(5) / 10.0, 0.25 + torch.randn(5) / 10.0], dim=0
).requires_grad_(True)
# gaussian_pos = (0.0 + torch.randn(10) / 10.0).requires_grad_(True)

gaussian_sigma = torch.rand_like(gaussian_pos).requires_grad_(True)
gaussian_amplitude = torch.rand_like(gaussian_pos).requires_grad_(True)


# gaussian_amplitude = torch.ones_like(gaussian_pos).requires_grad_(False)
# activate_a = torch.tanh
# activate_a = torch.sigmoid
def activate_a(x):
    return x


def gaussian(x, sigma=1, amplitude=1, query_x=0):
    return amplitude * torch.exp(-((x - query_x) ** 2) / (2 * sigma**2))


# derivative of gaussian = -x/sigma**2 * gaussian
def dgaussian_dx(x, sigma=1, amplitude=1, query_x=0):
    return (
        -amplitude
        * (x - query_x)
        / (sigma**2)
        * gaussian(x, query_x, sigma, amplitude)
    )


def composition(f, pos, gaussian_pos_, gaussian_sigma_, gaussian_amplitude_):
    comp = 0
    for i in range(gaussian_pos.size(0)):
        comp += f(
            gaussian_pos_[i],
            gaussian_sigma_[i],
            activate_a(gaussian_amplitude_[i]),
            pos,
        )
    return comp


def pred_dsdf_dx(
    x, gaussian_pos, gaussian_sigma, gaussian_amplitude, sigma, pred_sigmoid_sdf_
):
    return (
        (
            sigma
            / (pred_sigmoid_sdf_ * (1 - pred_sigmoid_sdf_))
            * composition(
                dgaussian_dx,
                x,
                gaussian_pos,
                gaussian_sigma,
                activate_a(gaussian_amplitude),
            )
        )
        .reshape(-1, 1)
        .norm(dim=-1)
        .squeeze()
    )


gt = sigmoid_sdf
# gt = sdf
optimizer = torch.optim.Adam([gaussian_pos, gaussian_sigma, gaussian_amplitude], lr=0.01)
for iter in tqdm(range(1000)):
    pred_sigmoid_sdf = composition(
        gaussian, x, gaussian_pos, gaussian_sigma, activate_a(gaussian_amplitude)
    )
    loss = torch.nn.functional.mse_loss(pred_sigmoid_sdf, gt)

    # pred_sigmoid_sdf_n25 = composition(
    #     gaussian, -0.25, gaussian_pos, gaussian_sigma, activate_a(gaussian_amplitude)
    # )
    # loss = torch.nn.functional.mse_loss(pred_sigmoid_sdf_n25, torch.tensor(0.5))
    # pred_sigmoid_sdf_p25 = composition(
    #     gaussian, 0.25, gaussian_pos, gaussian_sigma, activate_a(gaussian_amplitude)
    # )
    # loss += torch.nn.functional.mse_loss(pred_sigmoid_sdf_p25, torch.tensor(0.5))

    # pred_grad = composition(
    #     dgaussian_dx,
    #     x,
    #     gaussian_pos,
    #     gaussian_sigma,
    #     activate_a(gaussian_amplitude),
    # )

    # loss_eikonal = torch.nn.functional.mse_loss(pred_grad, gt * (1 - gt))

    # pred_grad = pred_dsdf_dx(
    #     x, gaussian_pos, gaussian_sigma, gaussian_amplitude, sigma, pred_sigmoid_sdf
    # )
    # loss_eikonal = torch.nn.functional.mse_loss(pred_grad, torch.ones_like(pred_grad))
    # loss += 0.0001 * loss_eikonal
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


def shape(f, x, sigma=1):
    query_x = x + torch.arange(-0.5, 0.5, 0.01)
    return f(x, sigma, 1, query_x)


with torch.no_grad():
    plt.subplot(5, 1, 1)
    plt.plot(x, shape(gaussian, 0.0, 0.05), label="Gaussian(0,0.05)", color="red")
    plt.plot(x, shape(dgaussian_dx, 0.0, 0.05), label="dGaussian_dx", color="blue")
    plt.grid()
    plt.legend()

    plt.subplot(5, 1, 2)
    # https://finthon.com/matplotlib-color-list/
    plt.plot(x, sdf, label="SDF", color="black")
    plt.plot(x, torch.ones_like(sdf), label="|dSDF_dx|", color="grey")
    plt.plot(x, sigmoid_sdf, label="Sigmoid(SDF)", color="red")
    # derivative of sigmoid = sigmoid*(1-sigmoid)
    plt.plot(x, sigmoid_sdf * (1 - sigmoid_sdf), label="dSigmoid_dx)", color="blue")

    plt.xlabel("x")
    plt.grid()
    plt.legend()

    plt.subplot(5, 1, 3)
    # https://finthon.com/matplotlib-color-list/
    plt.plot(x, gt, label="Sigmoid(SDF)", color="red", linewidth=3)
    plt.plot(
        x,
        composition(
            gaussian, x, gaussian_pos, gaussian_sigma, activate_a(gaussian_amplitude)
        ).detach(),
        label="PredSigmoid(SDF)",
        color="blue",
    )

    for i in range(gaussian_pos.size(0)):
        plt.plot(
            x,
            gaussian(
                gaussian_pos[i],
                gaussian_sigma[i],
                activate_a(gaussian_amplitude[i]),
                x,
            ).detach(),
            color="cyan",
        )

    # plt.plot(gaussian_pos, activate_a(gaussian_amplitude), "o", color="green")
    plt.xlabel("x")
    plt.grid()
    plt.legend()

    plt.subplot(5, 1, 4)
    # derivative of sigmoid = sigmoid*(1-sigmoid)
    # plt.plot(x, sigmoid_sdf * (1 - sigmoid_sdf), label="dSigmoid_dx", color="red")
    # plt.plot(
    #     x,
    #     composition(
    #         dgaussian_dx, x, gaussian_pos, gaussian_sigma, activate_a(gaussian_amplitude)
    #     ).detach(),
    #     label="PreddSigmoid_dx",
    #     color="blue",
    # )

    plt.plot(x, sdf, label="SDF", color="red")
    plt.plot(
        x,
        inv_sigmoid(
            composition(
                gaussian,
                x,
                gaussian_pos,
                gaussian_sigma,
                activate_a(gaussian_amplitude),
            ).detach(),
            sigma,
        ),
        label="PredSDF",
        color="blue",
    )

    # for i in range(gaussian_pos.size(0)):
    #     plt.plot(
    #         x,
    #         dgaussian_dx(
    #             gaussian_pos[i],
    #             gaussian_sigma[i],
    #             activate_a(gaussian_amplitude[i]),
    #             x,
    #         ).detach(),
    #         color="cyan",
    #     )

    plt.xlabel("x")
    plt.grid()
    plt.legend()

    plt.subplot(5, 1, 5)
    # derivative of sigmoid = sigmoid*(1-sigmoid)
    # plt.plot(x, sigmoid_sdf * (1 - sigmoid_sdf), label="dSigmoid_dx", color="red")
    # plt.plot(
    #     x,
    #     composition(
    #         dgaussian_dx, x, gaussian_pos, gaussian_sigma, activate_a(gaussian_amplitude)
    #     ).detach(),
    #     label="PreddSigmoid_dx",
    #     color="blue",
    # )

    plt.plot(x, torch.ones_like(sdf), label="|dsdf_dx|", color="red")
    
    pred_sigmoid_sdf = composition(
        gaussian, x, gaussian_pos, gaussian_sigma, activate_a(gaussian_amplitude)
    )
    plt.plot(
        x,
        pred_dsdf_dx(
            x, gaussian_pos, gaussian_sigma, gaussian_amplitude, sigma, pred_sigmoid_sdf
        ).detach(),
        label="|PreddSDF_dx|",
        color="blue",
    )

    # for i in range(gaussian_pos.size(0)):
    #     plt.plot(
    #         x,
    #         dgaussian_dx(
    #             gaussian_pos[i],
    #             gaussian_sigma[i],
    #             activate_a(gaussian_amplitude[i]),
    #             x,
    #         ).detach(),
    #         color="cyan",
    #     )

    plt.xlabel("x")
    # plt.ylim(0, 10)
    plt.grid()
    plt.legend()

    plt.show()
