/**
 * 带有状态保存功能的伪随机数生成器
 */
class Random {
	constructor(seed) {
		this.state = seed % 2147483647;
	};
	next() {
		this.state = (this.state * 16807) % 2147483647;
		return (this.state - 1) / 2147483646;
	};
	gaussian() {
		let u = 0;
		while (u <= 0) {
			u = this.next();
		}
		const v = this.next();
		return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
	};
};

function cloneCategoricalValue(value) {
	if (Array.isArray(value)) {
		return value.map(cloneCategoricalValue);
	}
	if (value && typeof value === 'object') {
		const cloned = {};
		for (const key of Object.keys(value)) {
			cloned[key] = cloneCategoricalValue(value[key]);
		}
		return cloned;
	}
	return value;
};

function isPrimitiveCategoricalValue(value) {
	return value === null || (typeof value !== 'object' && typeof value !== 'function');
};

/**
 * Trial 用于在 objective 中建议参数
 */
class Trial {
	constructor(study, trialParams) {
		this.study = study;
		this.trialParams = trialParams;
		this.usedParams = {};
	};

	registerMeta(name, meta) {
		const current = this.study.state.meta[name] || {};
		const currentChoices = current.choices;
		const nextChoices = meta.choices;
		const sameChoices = currentChoices === nextChoices || (
			Array.isArray(currentChoices)
			&& Array.isArray(nextChoices)
			&& currentChoices.length === nextChoices.length
			&& currentChoices.every((choice, index) => Object.is(choice, nextChoices[index]))
		);
		if (current.min === meta.min && current.max === meta.max && current.type === meta.type && sameChoices) {
			return;
		}
		this.study.state.meta[name] = { ...current, ...meta };
	};

	/**
	 * 初始化参数状态, 如果参数是首次出现, 则设定其初始中心值, 步长 (sigma) 和动量
	 * @param {string} name - 参数名称
	 * @param {number} min - 参数允许的最小值
	 * @param {number} max - 参数允许的最大值
	 */
	initParam(name, min, max, type = 'float') {
		// 如果该参数在之前的批次或当前初始化中未出现, 则初始化它
		if (!(name in this.study.state.params)) {
			this.study.state.params[name] = (min + max) / 2;
			this.study.state.sigmas[name] = (max - min) * 0.1;
			this.study.state.moments[name] = 0;
			if (!(name in this.trialParams)) {
				this.trialParams[name] = this.study.state.params[name];
			}
		}
		this.registerMeta(name, { min, max, type });
	};

	/**
	 * 建议一个浮点数参数
	 * @param {string} name - 参数名称
	 * @param {number} min - 最小值
	 * @param {number} max - 最大值
	 * @returns {number} 裁剪到 [min, max] 范围内的浮点值
	 */
	suggest_float(name, min, max) {
		this.initParam(name, min, max, 'float');
		const val = Math.max(min, Math.min(max, this.trialParams[name]));
		return this.usedParams[name] = val;
	};

	/**
	 * 建议一个整数参数
	 * @param {string} name - 参数名称
	 * @param {number} min - 最小值
	 * @param {number} max - 最大值
	 * @returns {number} 四舍五入后并裁剪到 [min, max] 范围内的整数值
	 */
	suggest_int(name, min, max) {
		this.initParam(name, min, max, 'int');
		const val = Math.max(min, Math.min(max, this.trialParams[name]));
		return this.usedParams[name] = Math.round(val);
	};

	/**
	 * 建议一个布尔值参数
	 * 内部将 0.5 作为阈值进行判定
	 * @param {string} name - 参数名称
	 * @returns {boolean} 建议的布尔值
	 */
	suggest_bool(name) {
		this.initParam(name, 0, 1, 'bool');
		return this.usedParams[name] = this.trialParams[name] > 0.5;
	};

	/**
	 * 从提供的数组中建议一个分类值
	 * @param {string} name - 参数名称
	 * @param {any[]} choices - 备选项数组
	 * @returns {any} 选定的分类值
	 */
	suggest_categorical(name, choices) {
		this.initParam(name, 0, choices.length - 1, 'categorical');
		const primitiveChoices = choices.every(isPrimitiveCategoricalValue);
		const safeChoices = primitiveChoices ? [...choices] : choices.map(cloneCategoricalValue);
		this.registerMeta(name, {
			min: 0,
			max: choices.length - 1,
			type: 'categorical',
			choices: safeChoices,
		});
		const raw = Math.max(0, Math.min(choices.length - 1, this.trialParams[name]));
		const index = Math.round(raw);
		const selectedChoice = safeChoices[index];
		if (primitiveChoices) {
			return this.usedParams[name] = selectedChoice;
		}
		this.usedParams[name] = cloneCategoricalValue(selectedChoice);
		return cloneCategoricalValue(selectedChoice);
	};
};

function valuesEqual(left, right) {
	if (Object.is(left, right)) return true;
	if (Array.isArray(left) && Array.isArray(right)) {
		if (left.length !== right.length) return false;
		for (let i = 0; i < left.length; i++) {
			if (!valuesEqual(left[i], right[i])) return false;
		}
		return true;
	}
	if (left && right && typeof left === 'object' && typeof right === 'object') {
		const leftKeys = Object.keys(left);
		const rightKeys = Object.keys(right);
		if (leftKeys.length !== rightKeys.length) return false;
		for (const key of leftKeys) {
			if (!(key in right) || !valuesEqual(left[key], right[key])) return false;
		}
		return true;
	}
	return false;
};

/**
 * 简单的进化优化器类, 用于通过进化算法寻找最优参数
 */
export default class HyperparameterOptimizer {
	constructor() {
		this.best_params = null;
		this.best_loss = Infinity;
		this.trial_count = 0;
		// 内部演化状态
		this.state = {
			params: {},		// 连续空间中心点
			sigmas: {},		// 独立维度步长
			moments: {},	// 独立维度动量
			meta: {},		// 参数元信息
			boolBias: {},	// 布尔参数偏置
			stagnation: 0,	// 停滞轮次
			rngState: null,	// 随机数生成器状态
		};
	};

	/**
	 * 导出当前内部状态, 用于恢复进度
	 */
	save() {
		return structuredClone({
			best_params: this.best_params,
			best_loss: this.best_loss,
			state: this.state,
		});
	};

	/**
	 * 执行进化优化流程, 寻找使目标函数值最小的参数组合
	 * @param {Function} objective - 目标函数
	 *  接收一个 Trial 实例作为参数, 并返回损失值 (越小越好)
	 *  示例: async (trial) => { const x = trial.suggest_float('x', 0, 1); return x*x; }
	 * @param {Object} [options={}]		- 配置选项
	 * @param {number} [options.n=100]	- 迭代次数
	 * @param {number} [options.seed=Math.random()]	- 随机数种子, 用于结果复现
	 * @param {Object|null} [options.cache=null]	- 状态快照 (通过 .save() 导出), 用于恢复进度
	 * @returns {Promise<void>}
	 */	async optimize(objective, { n = 100, seed = Math.random(), cache = null } = {}) {
		// 1. 恢复缓存（如果提供）
		if (cache) {
			for (const key in cache) {
				if (key in this) this[key] = structuredClone(cache[key]);
			}
		}

		// 2. 初始化随机数生成器
		// 如果缓存中有 rngState, 则继续随机序列; 否则重新开始
		const rng = new Random(seed);
		if (this.state.rngState !== null) {
			rng.state = this.state.rngState;
		}
		for (let i = 0; i < n; i++) {
			this.trial_count += 1;
			const baseLoss = this.best_loss;
			let currentTrialParams = {};
			let noiseVec = {};

			// 3. 基于当前内部状态生成参数
			for (let key in this.state.params) {
				const meta = this.state.meta[key] || { min: -Infinity, max: Infinity, type: 'float' };
				noiseVec[key] = rng.gaussian();
				let candidate = this.state.params[key] + this.state.moments[key] + (this.state.sigmas[key] * noiseVec[key]);

				if (meta.type === 'bool') {
					const bias = this.state.boolBias[key] || 0;
					candidate = candidate + bias > 0.5 ? 1 : 0;
					if (rng.next() < 0.12) candidate = candidate > 0.5 ? 0 : 1;
				} else if (meta.type === 'categorical') {
					candidate = Math.round(candidate);
					if (rng.next() < 0.22) {
						candidate = Math.round(meta.min + rng.next() * (meta.max - meta.min));
					}
				} else if (meta.type === 'int') {
					candidate = Math.round(candidate);
					if (rng.next() < 0.12) {
						candidate += rng.next() > 0.5 ? 1 : -1;
					}
				}

				currentTrialParams[key] = Math.max(meta.min, Math.min(meta.max, candidate));
			}

			const trial = new Trial(this, currentTrialParams);
			let value = await objective(trial);

			if (Number.isFinite(baseLoss) && value >= baseLoss) {
				const retryTrialParams = {};
				for (let key in this.state.params) {
					const meta = this.state.meta[key] || { min: -Infinity, max: Infinity, type: 'float' };
					const retryNoise = 0.5 * noiseVec[key] + 0.5 * rng.gaussian();
					let retryCandidate = this.state.params[key] + (this.state.sigmas[key] * retryNoise);
					if (meta.type === 'bool') {
						const bias = this.state.boolBias[key] || 0;
						retryCandidate = retryCandidate + bias > 0.5 ? 1 : 0;
					} else if (meta.type === 'categorical') {
						retryCandidate = Math.round(retryCandidate);
					} else if (meta.type === 'int') {
						retryCandidate = Math.round(retryCandidate);
					}
					retryTrialParams[key] = Math.max(meta.min, Math.min(meta.max, retryCandidate));
				}
				const retryTrial = new Trial(this, retryTrialParams);
				const retryValue = await objective(retryTrial);
				if (retryValue < value) {
					value = retryValue;
					currentTrialParams = retryTrialParams;
					for (let key in retryTrial.usedParams) {
						trial.usedParams[key] = retryTrial.usedParams[key];
					}
				}
			}

			const bestDelta = Number.isFinite(baseLoss)
				? Math.abs(baseLoss - this.best_loss)
				: Infinity;
			if (Number.isFinite(baseLoss) && value < baseLoss && bestDelta < Math.max(1e-9, Math.abs(baseLoss) * 0.08)) {
				const confirmTrial = new Trial(this, currentTrialParams);
				const confirmValue = await objective(confirmTrial);
				value = (value + confirmValue) / 2;
			}

			// 4. 进化逻辑
			if (value < this.best_loss) {
				this.state.stagnation = 0;
				const relativeGain = Number.isFinite(baseLoss) ? Math.min(1, Math.max(0, (baseLoss - value) / Math.max(1e-12, Math.abs(baseLoss)))) : 1;
				this.best_loss = value;
				this.best_params = { ...trial.usedParams };

				for (let key in this.state.params) {
					const delta = currentTrialParams[key] - this.state.params[key];
					const used = key in trial.usedParams;
					const metaType = this.state.meta[key]?.type;
					const effectiveChanged = metaType === 'bool' || metaType === 'int' || metaType === 'categorical'
						? currentTrialParams[key] !== this.state.params[key]
						: Math.abs(delta) > 1e-12;
					if (used && effectiveChanged) {
						this.state.moments[key] = 0.7 * this.state.moments[key] + 0.3 * delta;
						this.state.params[key] = currentTrialParams[key];
						this.state.sigmas[key] *= 1.15;
					}
					if (metaType === 'bool' && used && effectiveChanged) {
						const selected = currentTrialParams[key] > 0.5 ? 1 : 0;
						const direction = selected > 0.5 ? 1 : -1;
						const biasStep = 0.05 + 0.07 * relativeGain;
						this.state.boolBias[key] = Math.max(-0.35, Math.min(0.35, (this.state.boolBias[key] || 0) * 0.7 + direction * biasStep));
					}
				}
			} else {
				this.state.stagnation += 1;
				for (let key in this.state.sigmas) {
					const metaType = this.state.meta[key]?.type;
					this.state.sigmas[key] *= 0.85;
					this.state.moments[key] *= 0.5;
					if (metaType === 'bool') {
						this.state.boolBias[key] = (this.state.boolBias[key] || 0) * 0.8;
					}
					if (this.state.stagnation > 12 && metaType !== 'bool') {
						this.state.sigmas[key] *= 1.35;
					}
				}

				if (this.state.stagnation > 18) {
					for (let key in this.state.params) {
						const meta = this.state.meta[key] || { min: -Infinity, max: Infinity, type: 'float' };
						const resetCandidate = this.best_params && key in this.best_params ? this.best_params[key] : this.state.params[key];
						if (meta.type === 'bool') {
							this.state.params[key] = resetCandidate ? 1 : 0;
						} else if (meta.type === 'categorical' && Array.isArray(meta.choices)) {
							const resetIndex = meta.choices.findIndex((choice) => valuesEqual(choice, resetCandidate));
							const fallbackIndex = Math.max(meta.min, Math.min(meta.max, Number(this.state.params[key])));
							this.state.params[key] = resetIndex >= 0
								? Math.max(meta.min, Math.min(meta.max, resetIndex))
								: fallbackIndex;
						} else {
							this.state.params[key] = Math.max(meta.min, Math.min(meta.max, Number(resetCandidate)));
						}
						this.state.moments[key] *= 0.2;
					}
					this.state.stagnation = 0;
				}
			}
			
			// 5. 实时保存随机数状态, 确保即使分批执行, 随机序列也是连续不重复的
			this.state.rngState = rng.state;
		}
	};
};
