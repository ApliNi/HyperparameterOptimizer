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
		const u = this.next(), v = this.next();
		return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
	};
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

	/**
	 * 初始化参数状态, 如果参数是首次出现, 则设定其初始中心值, 步长 (sigma) 和动量
	 * @param {string} name - 参数名称
	 * @param {number} min - 参数允许的最小值
	 * @param {number} max - 参数允许的最大值
	 */
	initParam(name, min, max) {
		// 如果该参数在之前的批次或当前初始化中未出现, 则初始化它
		if (!(name in this.study.state.params)) {
			this.study.state.params[name] = (min + max) / 2;
			this.study.state.sigmas[name] = (max - min) * 0.1;
			this.study.state.moments[name] = 0;
			if (!(name in this.trialParams)) {
				this.trialParams[name] = this.study.state.params[name];
			}
		}
	};

	/**
	 * 建议一个浮点数参数
	 * @param {string} name - 参数名称
	 * @param {number} min - 最小值
	 * @param {number} max - 最大值
	 * @returns {number} 裁剪到 [min, max] 范围内的浮点值
	 */
	suggest_float(name, min, max) {
		this.initParam(name, min, max);
		const val = Math.max(min, Math.min(max, this.trialParams[name]));
		return (this.usedParams[name] = val);
	};

	/**
	 * 建议一个整数参数
	 * @param {string} name - 参数名称
	 * @param {number} min - 最小值
	 * @param {number} max - 最大值
	 * @returns {number} 四舍五入后并裁剪到 [min, max] 范围内的整数值
	 */
	suggest_int(name, min, max) {
		this.initParam(name, min, max);
		const val = Math.max(min, Math.min(max, this.trialParams[name]));
		return (this.usedParams[name] = Math.round(val));
	};

	/**
	 * 建议一个布尔值参数
	 * 内部将 0.5 作为阈值进行判定
	 * @param {string} name - 参数名称
	 * @returns {boolean} 建议的布尔值
	 */
	suggest_bool(name) {
		this.initParam(name, 0, 1);
		return (this.usedParams[name] = this.trialParams[name] > 0.5);
	};

	/**
	 * 从提供的数组中建议一个分类值
	 * @param {string} name - 参数名称
	 * @param {any[]} choices - 备选项数组
	 * @returns {any} 选定的分类值
	 */
	suggest_categorical(name, choices) {
		const index = this.suggest_int(name, 0, choices.length - 1);
		return (this.usedParams[name] = choices[index]);
	};
};

/**
 * 简单的进化优化器类, 用于通过进化算法寻找最优参数
 */
export default class HyperparameterOptimizer {
	constructor() {
		this.best_params = null;
		this.best_loss = Infinity;
		// 内部演化状态
		this.state = {
			params: {},		// 连续空间中心点
			sigmas: {},		// 独立维度步长
			moments: {},	// 独立维度动量
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
				if (key in this) this[key] = cache[key];
			}
		}

		// 2. 初始化随机数生成器
		// 如果缓存中有 rngState, 则继续随机序列; 否则重新开始
		const rng = new Random(seed);
		if (this.state.rngState !== null) {
			rng.state = this.state.rngState;
		}

		for (let i = 0; i < n; i++) {
			let currentTrialParams = {};
			let noiseVec = {};

			// 3. 基于当前内部状态生成参数
			for (let key in this.state.params) {
				noiseVec[key] = rng.gaussian();
				currentTrialParams[key] =	this.state.params[key] +
											this.state.moments[key] +
											(this.state.sigmas[key] * noiseVec[key]);
			}

			const trial = new Trial(this, currentTrialParams);
			const value = await objective(trial);

			// 4. 进化逻辑
			if (value < this.best_loss) {
				this.best_loss = value;
				this.best_params = { ...trial.usedParams };

				for (let key in this.state.params) {
					const delta = currentTrialParams[key] - this.state.params[key];
					this.state.moments[key] = 0.7 * this.state.moments[key] + 0.3 * delta;
					this.state.params[key] = currentTrialParams[key];
					this.state.sigmas[key] *= 1.15;	// 成功则尝试步长扩大
				}
			} else {
				for (let key in this.state.sigmas) {
					this.state.sigmas[key] *= 0.85;	// 失败则收缩步长
					this.state.moments[key] *= 0.5;	// 动量衰减
				}
			}
			
			// 5. 实时保存随机数状态, 确保即使分批执行, 随机序列也是连续不重复的
			this.state.rngState = rng.state;
		}
	};
};
