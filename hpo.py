"""
HPO (Hyperparameter Optimization) 모듈
optuna Bayesian Optimization으로 선택 모델의 최적 파라미터 탐색
최적화 목표: DCA 에이전트의 Sharpe Ratio 최대화
"""
import numpy as np

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# 모델별 탐색 파라미터 공간 정의
SEARCH_SPACE = {
    "Tabular Q-Learning": {
        "lr":    ("float", 0.001, 0.05, True),   # (type, min, max, log_scale)
        "gamma": ("float", 0.70,  0.99, False),
        "eps":   ("float", 0.1,   0.9,  False),
    },
    "DQN": {
        "lr":            ("float", 1e-4, 1e-2, True),
        "gamma":         ("float", 0.70, 0.99, False),
        "eps":           ("float", 0.1,  0.9,  False),
        "target_update": ("int",   5,    50,   False),
    },
    "Double DQN": {
        "lr":            ("float", 1e-4, 1e-2, True),
        "gamma":         ("float", 0.70, 0.99, False),
        "eps":           ("float", 0.1,  0.9,  False),
        "target_update": ("int",   5,    50,   False),
    },
    "Dueling DQN": {
        "lr":            ("float", 1e-4, 1e-2, True),
        "gamma":         ("float", 0.70, 0.99, False),
        "eps":           ("float", 0.1,  0.9,  False),
        "target_update": ("int",   5,    50,   False),
    },
    "PPO": {
        "lr":            ("float", 1e-4, 1e-2, True),
        "gamma":         ("float", 0.70, 0.99, False),
        "clip_eps":      ("float", 0.05, 0.4,  False),
        "rollout_len":   ("int",   32,   128,  False),
        "update_epochs": ("int",   2,    8,    False),
    },
    "Rainbow": {
        "lr":            ("float", 1e-4, 1e-2, True),
        "gamma":         ("float", 0.70, 0.99, False),
        "eps":           ("float", 0.1,  0.9,  False),
        "target_update": ("int",   5,    50,   False),
        "per_alpha":     ("float", 0.3,  0.8,  False),   # PER 우선순위 강도
        "per_beta_start":("float", 0.2,  0.6,  False),   # IS 보정 초기값
    },
}

# 각 설정값의 권장 기준
HPO_SETTINGS_GUIDE = {
    "n_trials": {
        "fast":     {"value": 10,  "label": "빠름 (~3분)",    "desc": "대략적인 방향만 파악"},
        "balanced": {"value": 20,  "label": "균형 (~8분)",    "desc": "권장. 충분한 탐색"},
        "thorough": {"value": 50,  "label": "정밀 (~20분)",   "desc": "최고 정확도"},
    },
    "episodes": {
        "fast":     {"value": 100, "label": "빠름",  "desc": "노이즈 많음. trial 수 늘릴 때"},
        "balanced": {"value": 200, "label": "균형",  "desc": "권장. 속도와 정확도 균형"},
        "thorough": {"value": 500, "label": "정밀",  "desc": "실제 학습과 동일한 조건"},
    },
    "pretrain": {
        "fast":     {"value": 0, "label": "없음",  "desc": "신경망 모델 (DQN 계열) 권장 안 함"},
        "balanced": {"value": 1, "label": "1회",   "desc": "권장. 안정적 초기화"},
        "thorough": {"value": 2, "label": "2회",   "desc": "Tabular Q에서 효과적"},
    },
}


def _suggest_params(trial, model_name):
    """optuna trial에서 모델별 파라미터 샘플링"""
    space  = SEARCH_SPACE.get(model_name, {})
    params = {}
    for name, spec in space.items():
        ptype, lo, hi, log = spec
        if ptype == "float":
            params[name] = trial.suggest_float(name, lo, hi, log=log)
        elif ptype == "int":
            params[name] = trial.suggest_int(name, int(lo), int(hi))
    return params


def _build_agent(env, model_name, use_constraints, params, AgentClass):
    """파라미터 딕셔너리로 에이전트 생성"""
    kwargs = dict(env=env, use_constraints=use_constraints, **params)
    if "eps" not in kwargs:
        kwargs["eps"] = 0.5
    return AgentClass(**kwargs)


def run_single_trial(env, model_name, AgentClass, params, episodes, pretrain=1):
    """
    단일 trial 실행 — Sharpe Ratio 반환
    HPO objective 함수에서 호출
    """
    try:
        agent = _build_agent(env, model_name, True, params, AgentClass)

        for _ in range(pretrain):
            for i in range(20, 20 + episodes):
                _, _, r, s, a, _, _ = agent.select_action(i)
                agent.learn(s, a, r, min(i + 1, len(env.data) - 1))
                agent.decay_epsilon(i - 20, episodes)

        for i in range(20, 20 + episodes):
            _, _, r, s, a, _, _ = agent.select_action(i)
            agent.learn(s, a, r, min(i + 1, len(env.data) - 1))
            agent.decay_epsilon(i - 20, episodes)

        sharpe = agent.get_sharpe()
        return sharpe if np.isfinite(sharpe) else -999.0

    except Exception:
        return -999.0


def _get_best_val(trial):
    """완료된 trial 중 최고 Sharpe 안전하게 반환"""
    completed = [t for t in trial.study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.value is not None]
    return max(t.value for t in completed) if completed else None


def optimize(env, model_name, AgentClass, episodes, n_trials=20,
             pretrain=1, callback=None):
    """
    단일 모델 HPO
    반환: (best_params, study)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("pip install optuna")

    def objective(trial):
        params = _suggest_params(trial, model_name)
        sharpe = run_single_trial(env, model_name, AgentClass,
                                  params, episodes, pretrain)
        if callback:
            best_val = _get_best_val(trial)
            callback(trial.number + 1, best_val if best_val is not None else sharpe, params)
        return sharpe

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study


def optimize_all_models(env, agent_registry, episodes, n_trials=20,
                        pretrain=1, callback=None):
    """
    전체 모델 HPO — 모델별로 독립적인 study를 실행하고 결과를 비교
    callback(model_name, trial_num, total_trials, best_sharpe, best_params)
    반환: {model_name: {"best_params": ..., "best_sharpe": ..., "study": ...}}
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("pip install optuna")

    results = {}
    model_names = list(agent_registry.keys())
    total_models = len(model_names)

    for model_idx, model_name in enumerate(model_names):
        AgentClass = agent_registry[model_name]

        # 이 모델에 대한 탐색 공간이 없으면 스킵
        if model_name not in SEARCH_SPACE:
            continue

        model_best = [-999.0]
        model_best_params = [{}]

        def make_objective(mname, mcls):
            def objective(trial):
                params = _suggest_params(trial, mname)
                sharpe = run_single_trial(env, mname, mcls,
                                          params, episodes, pretrain)
                if np.isfinite(sharpe) and sharpe > model_best[0]:
                    model_best[0]        = sharpe
                    model_best_params[0] = params

                if callback:
                    callback(
                        model_name   = mname,
                        model_idx    = model_idx + 1,
                        total_models = total_models,
                        trial_num    = trial.number + 1,
                        total_trials = n_trials,
                        best_sharpe  = model_best[0],
                        best_params  = model_best_params[0],
                    )
                return sharpe
            return objective

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(make_objective(model_name, AgentClass),
                       n_trials=n_trials, show_progress_bar=False)

        results[model_name] = {
            "best_params": study.best_params,
            "best_sharpe": study.best_value,
            "study":       study,
        }

    return results