import torch


def gaussian(u, sigma=1, amplitude=1, query_x=0):
    return amplitude * torch.exp(-((u - query_x) ** 2) / (2 * sigma**2))


# derivative of gaussian = -x/sigma**2 * gaussian
def dgaussian_dx(u_, sigma_=1, amplitude_=1, query_x_=0):
    return (u_ - query_x_) / (sigma_**2) * gaussian(u_, sigma_, amplitude_, query_x_)


def composition(f, pos, gaussian_pos_, gaussian_sigma_, gaussian_amplitude_):
    comp = 0
    for i in range(gaussian_pos_.size(0)):
        comp += f(
            gaussian_pos_[i],
            gaussian_sigma_[i],
            gaussian_amplitude_[i],
            pos,
        )
    return comp


def pred_dsdf_dx(
    x_, gaussian_pos_, gaussian_sigma_, gaussian_amplitude_, sigma_, pred_
):
    return (sigma_ / (pred_ * (1 - pred_))) * composition(
        dgaussian_dx, x_, gaussian_pos_, gaussian_sigma_, gaussian_amplitude_
    )


def sigmoid(x, sigma=1):
    return 1 / (1 + torch.exp(-x / sigma))


def inv_sigmoid(x, sigma=1):
    return sigma * torch.log(x / (1 - x))


def shape(f, x, sigma=1):
    query_x = x + torch.arange(-0.5, 0.5, 0.01)
    return f(x, sigma, 1, query_x)

def ssdf_to_alpha(ssdf_, eps_=1e-5):
    dist_intvs = ssdf_[..., 1:, 0] - ssdf_[..., :-1, 0]  # [B,R,N]
    est_prev_sdf = ssdf_ - iter_cos * dist_intvs * 0.5  # [B,R,N]
    est_next_sdf = ssdf_ + iter_cos * dist_intvs * 0.5  # [B,R,N]
    prev_cdf = (est_prev_sdf * inv_s).sigmoid()  # [B,R,N]
    next_cdf = (est_next_sdf * inv_s).sigmoid()  # [B,R,N]
    return max((ssdf_ - ssdf_ip1_) / (ssdf_ + eps_), 0)
