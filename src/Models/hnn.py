import torch
import torch.nn as nn


class HamiltonianNetwork(nn.Module):
    """
    Scalar Hamiltonian model.

    Input shape:
        [B, 2 * feature_dim]  -> concatenated [q, p]

    Output shape:
        [B, 1] -> scalar Hamiltonian energy
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class BidirectionalSymplecticRollout(nn.Module):
    """
    Midpoint-anchored bidirectional symplectic rollout.

    Takes a temporal feature sequence x [B, W, F], initializes:
        q_mid = x[:, mid]
        p_mid = x[:, mid] - x[:, mid - 1]

    Then performs symmetric forward/backward leapfrog rollout and returns:
        [q_b, p_b, q_mid, p_mid, q_f, p_f]

    Output shape:
        [B, 6 * feature_dim]
    """

    def __init__(self, feature_dim: int, steps: int, dt: float, hamiltonian_net: nn.Module):
        super().__init__()
        if steps % 2 != 0:
            raise ValueError("steps must be even for symmetric bidirectional rollout")
        self.feature_dim = feature_dim
        self.steps = steps
        self.dt = dt
        self.hamiltonian_net = hamiltonian_net

    def get_gradients(self, q: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state = torch.cat([q, p], dim=-1).requires_grad_(True)
        H = self.hamiltonian_net(state)
        dH = torch.autograd.grad(
            H.sum(),
            state,
            create_graph=True,
            retain_graph=True,
        )[0]
        dq = dH[:, self.feature_dim:]
        dp = -dH[:, :self.feature_dim]
        return dq, dp

    def leapfrog_step(self, q: torch.Tensor, p: torch.Tensor, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        dq, dp = self.get_gradients(q, p)
        p = p + 0.5 * dt * dp
        q = q + dt * dq
        _, dp_final = self.get_gradients(q, p)
        p = p + 0.5 * dt * dp_final
        return q, p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B, W, F] input, got {tuple(x.shape)}")
        if x.shape[1] < 2:
            raise ValueError("window length must be at least 2")
        if x.shape[2] != self.feature_dim:
            raise ValueError(
                f"feature_dim mismatch: expected {self.feature_dim}, got {x.shape[2]}"
            )

        mid_idx = x.shape[1] // 2
        q_mid = x[:, mid_idx, :]
        p_mid = x[:, mid_idx, :] - x[:, mid_idx - 1, :]

        q_f, p_f = q_mid, p_mid
        for _ in range(self.steps // 2):
            q_f, p_f = self.leapfrog_step(q_f, p_f, self.dt)

        q_b, p_b = q_mid, p_mid
        for _ in range(self.steps // 2):
            q_b, p_b = self.leapfrog_step(q_b, p_b, -self.dt)

        return torch.cat([q_b, p_b, q_mid, p_mid, q_f, p_f], dim=-1)


class HNNEncoderCore(nn.Module):
    """
    Baseline-faithful HNN encoder core:
        temporal features -> Hamiltonian network -> symplectic rollout readout
    """

    def __init__(self, feature_dim: int, steps: int, dt: float, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or (2 * feature_dim)
        self.hamiltonian = HamiltonianNetwork(
            state_dim=2 * feature_dim,
            hidden_dim=hidden_dim,
        )
        self.rollout = BidirectionalSymplecticRollout(
            feature_dim=feature_dim,
            steps=steps,
            dt=dt,
            hamiltonian_net=self.hamiltonian,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rollout(x)


class HNNLatentHead(nn.Module):
    """
    Maps rollout features [B, 6 * F] to the branch latent code [B, latent_dim].
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HNNDecoder(nn.Module):
    """
    Baseline-faithful HNN decoder path.

    Shape flow:
        z_sys                : [B, latent_dim]
        dense expansion      : [B, window_size * hidden_dim]
        reshape              : [B, window_size, hidden_dim]
        output projection    : [B, window_size, output_dim]
    """

    def __init__(self, latent_dim: int, window_size: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.expand = nn.Linear(latent_dim, window_size * hidden_dim)
        self.activation = nn.Tanh()
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.expand(z)
        x = self.activation(x)
        x = x.view(z.shape[0], self.window_size, self.hidden_dim)
        x = self.output_proj(x)
        return x


class HNNBranch(nn.Module):
    """
    Single entry point for the HNN branch.

    Input:
        x: [B, W, F]

    Outputs:
        z_sys  : [B, latent_dim]
        recon  : [B, W, F]
    """

    def __init__(
        self,
        input_dim: int,
        window_size: int,
        latent_dim: int,
        steps: int,
        dt: float,
        hamiltonian_hidden_dim: int | None = None,
        decoder_hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.steps = steps
        self.dt = dt

        self.encoder_core = HNNEncoderCore(
            feature_dim=input_dim,
            steps=steps,
            dt=dt,
            hidden_dim=hamiltonian_hidden_dim,
        )
        self.latent_head = HNNLatentHead(
            input_dim=6 * input_dim,
            latent_dim=latent_dim,
        )
        self.decoder = HNNDecoder(
            latent_dim=latent_dim,
            window_size=window_size,
            hidden_dim=decoder_hidden_dim,
            output_dim=input_dim,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"expected [B, W, F] input, got {tuple(x.shape)}")
        if x.shape[1] != self.window_size:
            raise ValueError(
                f"window_size mismatch: expected {self.window_size}, got {x.shape[1]}"
            )
        if x.shape[2] != self.input_dim:
            raise ValueError(
                f"input_dim mismatch: expected {self.input_dim}, got {x.shape[2]}"
            )

        rollout_features = self.encoder_core(x)
        z_sys = self.latent_head(rollout_features)
        recon = self.decoder(z_sys)
        return z_sys, recon
