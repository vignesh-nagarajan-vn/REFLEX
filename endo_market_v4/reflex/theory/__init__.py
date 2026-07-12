"""Closed-form math-theory modules (research priorities 1.1-1.5).

Each module implements, as pure functions of :class:`reflex.config.Config`
(numpy/scipy only -- no torch), one derivation from ``theory/`` (the copies of
``research/math-theory/`` shipped with this package):

* :mod:`.analytic_boundary` -- 1.1: closed-form ``gamma``, ``beta``,
  ``epsilon``, the fixed point ``h*`` and the modulus ``m = epsilon*beta/gamma``.
* :mod:`.perfgd` -- 1.2: the analytic PerfGD correction
  ``Delta = -beta*(h-psi)*epsilon(h)``, the performative optimum ``h_PO``,
  ``gamma_PO``, and the echo-chamber gap.
* :mod:`.multi_dealer` -- 1.3: the ``N``-dealer boundary
  ``epsilon < gamma/(N_eff*beta)``, the joint Jacobian, and mean-field limits.
* :mod:`.robust` -- 1.4: the ambiguity radius, the robust stability
  certificate, and the ``O(1/sqrt(n))`` rate machinery.
* :mod:`.factor_scaling` -- 1.5: the ``d x d`` modulus matrix
  ``M = beta*Gamma^{-1}*E``, the Woodbury reduction, and the truncation bound.
* :mod:`.lazy_deploy` -- 1.6 (v4): the K-step RGD deployment map
  ``mu(K) = -m + c^K (1+m)``, the effective curvature ``gamma_eff(K)``, the
  deadbeat / max-stable step counts, and the ``c``-fit for the CRN K-probe.
"""

from .analytic_boundary import (
    AnalyticBoundary,
    ReferenceState,
    analytic_boundary,
    best_response,
    beta,
    blind_gradient,
    epsilon,
    frozen_gradient,
    gamma,
    gate_means,
    reference_state,
    solve_fixed_point,
    tau,
)
from .perfgd import (
    EchoChamberGap,
    PerfGDResult,
    analyze_perfgd,
    echo_chamber_gap,
    gamma_po,
    perfgd_correction,
    perfgd_gradient,
    run_perfgd,
    run_rrm_cobweb,
    solve_performative_optimum,
)
from .multi_dealer import (
    CommonModeProbe,
    MeanFieldLimit,
    MultiDealerBoundary,
    common_mode_probe,
    effective_config,
    identify_kappa,
    joint_jacobian,
    mean_field_boundary,
    measure_common_mode_modulus,
    measure_differential_modulus,
    multi_dealer_boundary,
    n_eff,
    run_joint_rrm,
    strong_coupling_limit,
    sweep_dealer_count,
)
from .robust import (
    RadiusCalibration,
    RobustBoundaryResult,
    RobustCertificate,
    calibrate_radius,
    empirical_radius,
    finite_sample_radius,
    loglog_rate,
    measure_modulus_estimates,
    rate_check,
    robust_boundary,
    robust_certificate,
    sample_complexity,
)
from .factor_scaling import (
    FactorModulus,
    PerBondConstants,
    TruncationBound,
    factor_modulus,
    factorize_corr,
    gamma_inverse_woodbury,
    modulus_matrix,
    per_bond_constants,
    spectral_radius,
    spectral_radius_woodbury,
    sweep_global_factor,
    systemic_spectral_radius,
    truncation_error_bound,
)
from .lazy_deploy import (
    LazyDeployCurve,
    deadbeat_k,
    effective_modulus,
    equal_modulus_k,
    fit_inner_contraction,
    gamma_eff,
    k_step_slope,
    lazy_deploy_curve,
    lazy_weight,
    max_stable_k,
)

__all__ = [
    # 1.1 analytic stability boundary
    "AnalyticBoundary",
    "ReferenceState",
    "analytic_boundary",
    "best_response",
    "beta",
    "blind_gradient",
    "epsilon",
    "frozen_gradient",
    "gamma",
    "gate_means",
    "reference_state",
    "solve_fixed_point",
    "tau",
    # 1.2 PerfGD correction
    "EchoChamberGap",
    "PerfGDResult",
    "analyze_perfgd",
    "echo_chamber_gap",
    "gamma_po",
    "perfgd_correction",
    "perfgd_gradient",
    "run_perfgd",
    "run_rrm_cobweb",
    "solve_performative_optimum",
    # 1.3 multi-dealer systemic risk
    "CommonModeProbe",
    "MeanFieldLimit",
    "MultiDealerBoundary",
    "common_mode_probe",
    "effective_config",
    "identify_kappa",
    "joint_jacobian",
    "mean_field_boundary",
    "measure_common_mode_modulus",
    "measure_differential_modulus",
    "multi_dealer_boundary",
    "n_eff",
    "run_joint_rrm",
    "strong_coupling_limit",
    "sweep_dealer_count",
    # 1.4 distributionally robust boundary
    "RadiusCalibration",
    "RobustBoundaryResult",
    "RobustCertificate",
    "calibrate_radius",
    "empirical_radius",
    "finite_sample_radius",
    "loglog_rate",
    "measure_modulus_estimates",
    "rate_check",
    "robust_boundary",
    "robust_certificate",
    "sample_complexity",
    # 1.5 factor-model scaling
    "FactorModulus",
    "PerBondConstants",
    "TruncationBound",
    "factor_modulus",
    "factorize_corr",
    "gamma_inverse_woodbury",
    "modulus_matrix",
    "per_bond_constants",
    "spectral_radius",
    "spectral_radius_woodbury",
    "sweep_global_factor",
    "systemic_spectral_radius",
    "truncation_error_bound",
    # 1.6 lazy deployment
    "LazyDeployCurve",
    "deadbeat_k",
    "effective_modulus",
    "equal_modulus_k",
    "fit_inner_contraction",
    "gamma_eff",
    "k_step_slope",
    "lazy_deploy_curve",
    "lazy_weight",
    "max_stable_k",
]
